"""
Kronos time-series encoder service.

Quick setup (because Kronos is source-code based, not a pip package):
1) Clone Kronos locally:
    git clone https://github.com/shiyu-coder/Kronos
2) Install Kronos dependencies:
    pip install -r <PATH_TO_KRONOS>/requirements.txt
3) Point this handler to the cloned repo:
    set KRONOS_REPO_PATH=C:\\path\\to\\Kronos
4) Optional model ids (defaults shown):
    set KRONOS_TOKENIZER_ID=NeoQuasar/Kronos-Tokenizer-base
    set KRONOS_MODEL_ID=NeoQuasar/Kronos-small

Main API:
- KronosService.encode_timeseries_batch(list_of_series) -> Tensor[B, D]
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import torch

ArrayLikeSeries = Union[Sequence[float], np.ndarray, torch.Tensor]


class KronosService:
    """
    Thin service wrapper around Kronos foundational model.

    Features:
      - optional Kronos SDK backend if available
      - batch encoding for time-series representations
      - deterministic fallback encoder for local/dev usage
    """

    def __init__(
        self,
        model_name: str = "NeoQuasar/Kronos-small",
        tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base",
        device: Optional[str] = None,
    ):
        self.model_name = model_name or os.getenv("KRONOS_MODEL_ID", "NeoQuasar/Kronos-small")
        self.tokenizer_name = tokenizer_name or os.getenv(
            "KRONOS_TOKENIZER_ID", "NeoQuasar/Kronos-Tokenizer-base"
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.backend = "fallback"
        self.model = None
        self.tokenizer = None
        self._load_model()

    @staticmethod
    def _candidate_repo_paths() -> List[Path]:
        env_path = os.getenv("KRONOS_REPO_PATH", "").strip()
        candidates: List[Path] = []
        if env_path:
            candidates.append(Path(env_path))

        # Common local defaults relative to this file/workspace.
        here = Path(__file__).resolve()
        candidates.extend(
            [
                Path.cwd() / "Kronos",
                here.parents[3] / "Kronos",
                here.parents[2] / "Kronos",
                here.parents[1] / "Kronos",
            ]
        )

        unique: List[Path] = []
        seen = set()
        for c in candidates:
            key = str(c.resolve()) if c.exists() else str(c)
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique

    def _load_model(self):
        """Load Kronos from local repo source path; otherwise keep fallback mode."""
        try:
            repo_root = None
            for candidate in self._candidate_repo_paths():
                if (candidate / "model" / "__init__.py").exists():
                    repo_root = candidate
                    break

            if repo_root is None:
                raise FileNotFoundError(
                    "Kronos repo not found. Set KRONOS_REPO_PATH to your cloned Kronos directory."
                )

            repo_str = str(repo_root.resolve())
            if repo_str not in sys.path:
                sys.path.insert(0, repo_str)

            # Kronos official import path from source repo.
            from model import Kronos, KronosTokenizer  # type: ignore

            self.tokenizer = KronosTokenizer.from_pretrained(self.tokenizer_name)
            self.model = Kronos.from_pretrained(self.model_name)
            self.tokenizer = self.tokenizer.to(self.device)
            self.model = self.model.to(self.device)
            self.model.eval()
            self.backend = "kronos-local-repo"
        except Exception:
            self.model = None
            self.tokenizer = None
            self.backend = "fallback"

    @staticmethod
    def _to_1d_tensor(series: ArrayLikeSeries) -> torch.Tensor:
        if isinstance(series, torch.Tensor):
            ts = series.detach().float().flatten()
        elif isinstance(series, np.ndarray):
            ts = torch.tensor(series, dtype=torch.float32).flatten()
        else:
            ts = torch.tensor(list(series), dtype=torch.float32).flatten()

        if ts.numel() == 0:
            raise ValueError("Each time series must contain at least one value.")
        if not torch.isfinite(ts).all():
            raise ValueError("Time series values must be finite numbers.")
        return ts

    @staticmethod
    def _z_norm(ts: torch.Tensor) -> torch.Tensor:
        std = ts.std(unbiased=False)
        if std.item() == 0:
            return ts - ts.mean()
        return (ts - ts.mean()) / std

    @staticmethod
    def _fallback_encode_one(ts: torch.Tensor, n_fft_coeffs: int = 8) -> torch.Tensor:
        """Compact deterministic representation for one series."""
        delta = torch.diff(ts)
        slope = delta.mean() if delta.numel() > 0 else torch.tensor(0.0)
        vol = delta.std(unbiased=False) if delta.numel() > 0 else torch.tensor(0.0)

        q25 = torch.quantile(ts, 0.25)
        q50 = torch.quantile(ts, 0.50)
        q75 = torch.quantile(ts, 0.75)

        stats = torch.tensor(
            [
                ts.mean().item(),
                ts.std(unbiased=False).item(),
                ts.min().item(),
                ts.max().item(),
                q25.item(),
                q50.item(),
                q75.item(),
                slope.item(),
                vol.item(),
                float(ts.numel()),
            ],
            dtype=torch.float32,
        )

        spec = torch.abs(torch.fft.rfft(ts))[:n_fft_coeffs]
        if spec.numel() < n_fft_coeffs:
            spec = torch.nn.functional.pad(spec, (0, n_fft_coeffs - spec.numel()))

        return torch.cat([stats, spec.float()], dim=0)

    def _kronos_encode_one(self, ts: torch.Tensor) -> torch.Tensor:
        """Encode one univariate series via Kronos context representation."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Kronos model/tokenizer not loaded.")

        # Kronos expects K-line features. For univariate inputs, map into
        # pseudo OHLC + zero volume/amount: [open, high, low, close, volume, amount].
        zeros = torch.zeros_like(ts)
        x = torch.stack([ts, ts, ts, ts, zeros, zeros], dim=-1).unsqueeze(0).to(self.device)

        tokens = self.tokenizer.encode(x, half=True)
        s1_ids, s2_ids = tokens

        # decode_s1 returns (logits, context). We use mean pooled context.
        _, context = self.model.decode_s1(s1_ids, s2_ids, stamp=None, padding_mask=None)
        rep = context.mean(dim=1).squeeze(0)
        return rep.detach().cpu()

    @torch.inference_mode()
    def encode(
        self,
        series_batch: Sequence[ArrayLikeSeries],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Encode one or more time series into dense representations.

        Args:
            series_batch: list of univariate time series.
            normalize: apply z-normalization per time series.

        Returns:
            Tensor of shape (batch_size, representation_dim).
        """
        if not isinstance(series_batch, Sequence) or len(series_batch) == 0:
            raise ValueError("series_batch must be a non-empty list.")

        prepared: List[torch.Tensor] = []
        for series in series_batch:
            ts = self._to_1d_tensor(series)
            if normalize:
                ts = self._z_norm(ts)
            prepared.append(ts)

        if self.backend == "kronos-local-repo" and self.model is not None and self.tokenizer is not None:
            reps: List[torch.Tensor] = []
            for ts in prepared:
                try:
                    reps.append(self._kronos_encode_one(ts))
                except Exception:
                    reps.append(self._fallback_encode_one(ts))
            return torch.stack(reps, dim=0)

        return torch.stack([self._fallback_encode_one(ts) for ts in prepared], dim=0)

    @torch.inference_mode()
    def encode_timeseries_batch(
        self,
        series_batch: Sequence[ArrayLikeSeries],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Alias method for clarity in callers.

        Returns:
            Tensor of shape (batch_size, representation_dim).
        """
        return self.encode(series_batch, normalize=normalize)


def test_kronos_service():
    service = KronosService()

    batch = [
        [100.0, 101.2, 102.1, 101.7, 103.0, 104.1],
        [50.0, 49.8, 50.2, 50.5, 50.4, 50.8, 51.1],
    ]

    reps = service.encode_timeseries_batch(batch)
    print("Backend:", service.backend)
    print("Representations shape:", reps.shape)
    print("Representations:", reps)


if __name__ == "__main__":
    test_kronos_service()