"""
Kronos time-series encoder service.

Quick setup (because Kronos is source-code based, not a pip package):
1) Clone Kronos locally:
    git clone https://github.com/shiyu-coder/Kronos
2) Install Kronos dependencies:
    pip install -r <PATH_TO_KRONOS>/requirements.txt
3) Point this handler to the cloned repo:
    export KRONOS_REPO_PATH=/path/to/Kronos
4) Optional model ids (env overrides defaults):
    export KRONOS_TOKENIZER_ID=NeoQuasar/Kronos-Tokenizer-base
    export KRONOS_MODEL_ID=NeoQuasar/Kronos-small

Main API:
- KronosService.encode(batch) -> Tensor[B, D]
- KronosService.encode_timeseries_batch(batch) -> Tensor[B, D]  (alias)

Notes:
- Univariate input (T,) is mapped to pseudo-OHLCVA:
  [open=high=low=close=ts, volume=0, amount=0].
- Multivariate input (T, 5) is treated as OHLCV; (T, 6) as OHLCVA.
- Output dimensionality D is FIXED:
    - Kronos backend  → raw d_model (512 for Kronos-small, etc.)
    - Fallback backend → projected up to `output_dim` (default 512)
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)

ArrayLikeSeries = Union[Sequence[float], np.ndarray, torch.Tensor]

# d_model per model variant — used to set output_dim default
_KRONOS_D_MODEL = {
    "NeoQuasar/Kronos-mini": 256,
    "NeoQuasar/Kronos-small": 512,
    "NeoQuasar/Kronos-base": 832,
    "NeoQuasar/Kronos-large": 1664,
}


class KronosService:
    """
    Thin service wrapper around the Kronos foundation model.

    Features:
      - Uses local Kronos repo + Hugging Face weights when available.
      - Deterministic fallback encoder for environments without Kronos.
      - Raw Kronos embedding space (d_model) as output — no projection down.
      - Fallback encoder is projected UP to the same output_dim.
      - Fixed output embedding dimension across both backends.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
        device: Optional[str] = None,
        strict: bool = False,
        n_fft_coeffs: int = 8,
        output_dim: Optional[int] = None,
    ):
        # Env vars override constructor defaults
        self.model_name = os.getenv("KRONOS_MODEL_ID", model_name or "NeoQuasar/Kronos-small")
        self.tokenizer_name = os.getenv(
            "KRONOS_TOKENIZER_ID", tokenizer_name or "NeoQuasar/Kronos-Tokenizer-base"
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.n_fft_coeffs = int(n_fft_coeffs)
        self._fallback_raw_dim = 10 + self.n_fft_coeffs  # stats + FFT

        # output_dim = d_model of chosen model (or explicit override)
        self.output_dim: int = output_dim or _KRONOS_D_MODEL.get(self.model_name, 512)

        self.strict = strict
        self.backend = "fallback"
        self.model = None
        self.tokenizer = None

        # Seeded fixed projection: fallback_raw_dim -> output_dim
        self._fallback_proj: Optional[nn.Linear] = None
        self._init_fallback_proj()

        self._load_model()

    # -------------------------------------------------------------------------
    # Fallback projection (raw stats -> output_dim)
    # -------------------------------------------------------------------------
    def _init_fallback_proj(self) -> None:
        g = torch.Generator(device="cpu")
        g.manual_seed(42)
        proj = nn.Linear(self._fallback_raw_dim, self.output_dim, bias=False)
        with torch.no_grad():
            proj.weight.copy_(
                torch.randn(self.output_dim, self._fallback_raw_dim, generator=g)
                / float(self._fallback_raw_dim) ** 0.5
            )
        proj.requires_grad_(False)
        proj.eval()
        self._fallback_proj = proj

    # -------------------------------------------------------------------------
    # Repo discovery / loading
    # -------------------------------------------------------------------------
    @staticmethod
    def _candidate_repo_paths() -> List[Path]:
        env_path = os.getenv("KRONOS_REPO_PATH", "").strip()
        candidates: List[Path] = []
        if env_path:
            candidates.append(Path(env_path))

        here = Path(__file__).resolve()
        parents = list(here.parents)

        candidates.append(Path.cwd() / "Kronos")
        for idx in range(min(3, len(parents))):
            candidates.append(parents[idx] / "Kronos")

        unique: List[Path] = []
        seen = set()
        for c in candidates:
            try:
                key = str(c.resolve())
            except FileNotFoundError:
                key = str(c)
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique

    def _load_model(self) -> None:
        repo_root: Optional[Path] = None
        try:
            for candidate in self._candidate_repo_paths():
                if (candidate / "model" / "__init__.py").exists():
                    repo_root = candidate
                    break

            if repo_root is None:
                msg = (
                    "Kronos repo not found. "
                    "Set KRONOS_REPO_PATH to your cloned Kronos directory."
                )
                if self.strict:
                    raise FileNotFoundError(msg)
                logger.warning("%s Falling back to deterministic encoder.", msg)
                return

            repo_str = str(repo_root.resolve())
            if repo_str not in sys.path:
                sys.path.insert(0, repo_str)

            from model import Kronos, KronosTokenizer  # type: ignore

            self.tokenizer = KronosTokenizer.from_pretrained(self.tokenizer_name)
            self.model = Kronos.from_pretrained(self.model_name)
            self.tokenizer = self.tokenizer.to(self.device)
            self.model = self.model.to(self.device)
            self.model.eval()
            self.backend = "kronos-local-repo"
            logger.info(
                "Initialized Kronos backend '%s' with model '%s' on %s (output_dim=%d)",
                self.backend,
                self.model_name,
                self.device,
                self.output_dim,
            )
        except Exception as e:
            logger.exception("Failed to initialize Kronos backend; using fallback. Reason: %s", e)
            if self.strict:
                raise
            self.model = None
            self.tokenizer = None
            self.backend = "fallback"

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    @staticmethod
    def _to_tensor(series: ArrayLikeSeries) -> torch.Tensor:
        if isinstance(series, torch.Tensor):
            ts = series.detach().float()
        elif isinstance(series, np.ndarray):
            ts = torch.tensor(series, dtype=torch.float32)
        else:
            ts = torch.tensor(list(series), dtype=torch.float32)

        if ts.numel() == 0:
            raise ValueError("Each time series must contain at least one value.")
        if not torch.isfinite(ts).all():
            raise ValueError("Time series values must be finite numbers.")
        return ts

    @staticmethod
    def _z_norm(ts: torch.Tensor) -> torch.Tensor:
        """Z-score normalize per channel, then clamp to [-5, 5] (Kronos pipeline)."""
        if ts.ndim == 1:
            mean = ts.mean()
            std = ts.std(unbiased=False)
            out = (ts - mean) / std if std.item() != 0 else ts - mean
        else:
            mean = ts.mean(dim=0, keepdim=True)
            std = ts.std(dim=0, keepdim=True, unbiased=False)
            std = torch.where(std == 0, torch.ones_like(std), std)
            out = (ts - mean) / std
        return out.clamp(-5.0, 5.0)

    # -------------------------------------------------------------------------
    # Fallback encoder
    # -------------------------------------------------------------------------
    def _fallback_encode_one(self, ts: torch.Tensor) -> torch.Tensor:
        """
        Compact deterministic stats + FFT representation, projected to output_dim.
        ts must be 1D after flattening.
        """
        ts = ts.flatten()
        delta = torch.diff(ts)
        slope = delta.mean() if delta.numel() > 0 else torch.tensor(0.0)
        vol = delta.std(unbiased=False) if delta.numel() > 0 else torch.tensor(0.0)

        stats = torch.tensor(
            [
                ts.mean().item(),
                ts.std(unbiased=False).item(),
                ts.min().item(),
                ts.max().item(),
                torch.quantile(ts, 0.25).item(),
                torch.quantile(ts, 0.50).item(),
                torch.quantile(ts, 0.75).item(),
                slope.item(),
                vol.item(),
                float(ts.numel()),
            ],
            dtype=torch.float32,
        )

        spec = torch.abs(torch.fft.rfft(ts))[: self.n_fft_coeffs]
        if spec.numel() < self.n_fft_coeffs:
            spec = torch.nn.functional.pad(spec, (0, self.n_fft_coeffs - spec.numel()))

        raw = torch.cat([stats, spec.float()], dim=0)  # (fallback_raw_dim,)
        assert self._fallback_proj is not None
        with torch.no_grad():
            return self._fallback_proj(raw)  # (output_dim,)

    # -------------------------------------------------------------------------
    # Kronos-based encoders
    # -------------------------------------------------------------------------
    def _kronos_encode_one(self, ts: torch.Tensor) -> torch.Tensor:
        """Univariate (T,) → pseudo-OHLCVA → raw Kronos d_model embedding."""
        assert self.model is not None and self.tokenizer is not None
        zeros = torch.zeros_like(ts)
        x = torch.stack([ts, ts, ts, ts, zeros, zeros], dim=-1).unsqueeze(0).to(self.device)

        s1_ids, s2_ids = self.tokenizer.encode(x, half=True)
        s1_ids, s2_ids = s1_ids.to(self.device), s2_ids.to(self.device)

        _, context = self.model.decode_s1(s1_ids, s2_ids, stamp=None, padding_mask=None)
        return context.mean(dim=1).squeeze(0).detach().cpu()  # (d_model,)

    def _kronos_encode_ohlcv(self, x_ohlcv: torch.Tensor) -> torch.Tensor:
        """
        Multivariate (T, 5) or (T, 6) → OHLCVA → raw Kronos d_model embedding.
        C=5: treated as OHLCV, amount=0 appended.
        C=6: treated as OHLCVA directly.
        """
        assert self.model is not None and self.tokenizer is not None
        T, C = x_ohlcv.shape
        if C not in (5, 6):
            raise ValueError(f"Expected C in (5, 6) for OHLCV(A), got C={C}.")

        if C == 5:
            O, H, L, Cl, V = x_ohlcv.unbind(dim=-1)
            feats = torch.stack([O, H, L, Cl, V, torch.zeros_like(V)], dim=-1)
        else:
            feats = x_ohlcv

        x = feats.unsqueeze(0).to(self.device)  # (1, T, 6)
        s1_ids, s2_ids = self.tokenizer.encode(x, half=True)
        s1_ids, s2_ids = s1_ids.to(self.device), s2_ids.to(self.device)

        _, context = self.model.decode_s1(s1_ids, s2_ids, stamp=None, padding_mask=None)
        return context.mean(dim=1).squeeze(0).detach().cpu()  # (d_model,)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    @torch.inference_mode()
    def encode(
        self,
        series_batch: Union[torch.Tensor, Sequence[ArrayLikeSeries]],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Encode one or more time series into dense embedding vectors.

        Args:
            series_batch:
                - Tensor (B, T)       → B univariate series
                - Tensor (B, T, C)    → B multivariate series, C must be 5 or 6
                - Iterable of 1D/2D arraylikes
            normalize: z-norm + clamp per series/channel before encoding.

        Returns:
            Tensor of shape (B, output_dim).
        """
        if isinstance(series_batch, torch.Tensor):
            if series_batch.ndim == 2:
                series_list = [series_batch[i] for i in range(series_batch.shape[0])]
            elif series_batch.ndim == 3:
                series_list = [series_batch[i] for i in range(series_batch.shape[0])]
            else:
                raise ValueError(
                    "Expected tensor of shape (B, T) or (B, T, C), "
                    f"got {tuple(series_batch.shape)}."
                )
        else:
            series_list = list(series_batch)

        kronos_ok = (
            self.backend == "kronos-local-repo"
            and self.model is not None
            and self.tokenizer is not None
        )

        reps: List[torch.Tensor] = []
        for series in series_list:
            ts = self._to_tensor(series)
            if ts.ndim not in (1, 2):
                raise ValueError(
                    f"Each series must be 1D (T,) or 2D (T, C), got {tuple(ts.shape)}."
                )
            if normalize:
                ts = self._z_norm(ts)

            try:
                if kronos_ok:
                    rep = self._kronos_encode_one(ts) if ts.ndim == 1 else self._kronos_encode_ohlcv(ts)
                else:
                    rep = self._fallback_encode_one(ts)
            except Exception as e:
                logger.exception("Encoding failed for one series; falling back. Reason: %s", e)
                rep = self._fallback_encode_one(ts)

            reps.append(rep)

        return torch.stack(reps, dim=0)

    @torch.inference_mode()
    def encode_timeseries_batch(
        self,
        series_batch: Union[torch.Tensor, Sequence[ArrayLikeSeries]],
        normalize: bool = True,
    ) -> torch.Tensor:
        """Alias for encode(). Returns Tensor of shape (B, output_dim)."""
        return self.encode(series_batch, normalize=normalize)


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------
def test_kronos_service() -> None:
    logging.basicConfig(level=logging.INFO)

    service = KronosService()
    print(f"Backend      : {service.backend}")
    print(f"Output dim   : {service.output_dim}")

    # 5-channel OHLCV series, batch of 3, 100 timesteps
    batch = torch.randn(3, 100, 5)
    reps = service.encode(batch)
    print(f"Output shape : {reps.shape}")   # (3, 512) with Kronos-small


if __name__ == "__main__":
    test_kronos_service()