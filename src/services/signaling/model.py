import os
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Optional
from .config import SignalingConfig
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR
from .s3_risk_management import *
import logging
import math


logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 256):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


# ─────────────────────────────────────────────────────────────────────────────
#  Hierarchical Document Encoder
# ─────────────────────────────────────────────────────────────────────────────

class HierarchicalDocEncoder(nn.Module):
    def __init__(
        self,
        d_txt:           int   = 768,
        d_model:         int   = 512,
        n_chunks:        int   = 8,
        n_heads:         int   = 8,
        d_ff:            int   = 1024,
        dropout:         float = 0.1,
        n_local_layers:  int   = 2,
        n_global_layers: int   = 2,
    ):
        super().__init__()
        self.n_chunks = n_chunks
        self.d_model  = d_model

        self.input_proj   = nn.Linear(d_txt, d_model)
        self.chunk_tokens = nn.Parameter(torch.randn(n_chunks, d_model) * 0.02)
        self.chunk_mask_token = nn.Parameter(torch.randn(d_model) * 0.02)

        local_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.local_encoder = nn.TransformerEncoder(local_layer, num_layers=n_local_layers)

        global_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.global_encoder = nn.TransformerEncoder(global_layer, num_layers=n_global_layers)
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.normal_(self.chunk_tokens,     mean=0.0, std=0.02)
        nn.init.normal_(self.chunk_mask_token, mean=0.0, std=0.02)

    def _pad_input(self, x, doc_mask):
        D_doc   = x.shape[1]
        pad_len = (self.n_chunks - D_doc % self.n_chunks) % self.n_chunks
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
            if doc_mask is not None:
                doc_mask = F.pad(doc_mask, (0, pad_len), value=False)
        return x, doc_mask, x.shape[1]

    def _local_encode(self, x, doc_mask, B, D_padded):
        chunk_len = D_padded // self.n_chunks
        x = x.view(B, self.n_chunks, chunk_len, self.d_model)

        if doc_mask is not None:
            chunk_mask = ~doc_mask.view(B, self.n_chunks, chunk_len)
        else:
            chunk_mask = None

        tok = self.chunk_tokens.unsqueeze(0).unsqueeze(2).expand(B, -1, -1, -1)
        x_with_tok = torch.cat([tok, x], dim=2)  # (B, n_chunks, 1+chunk_len, d_model)

        if chunk_mask is not None:
            tok_mask = torch.zeros(B, self.n_chunks, 1, dtype=torch.bool, device=x.device)
            chunk_mask_with_tok = torch.cat([tok_mask, chunk_mask], dim=2)
        else:
            chunk_mask_with_tok = None

        x_flat    = x_with_tok.view(B * self.n_chunks, 1 + chunk_len, self.d_model)
        mask_flat = (
            chunk_mask_with_tok.view(B * self.n_chunks, 1 + chunk_len)
            if chunk_mask_with_tok is not None else None
        )

        if mask_flat is not None:
            all_masked = mask_flat.all(dim=-1)
            if all_masked.any():
                mask_flat[all_masked, 0] = False

        local_out  = self.local_encoder(x_flat, src_key_padding_mask=mask_flat)
        chunk_repr = local_out[:, 0, :].view(B, self.n_chunks, self.d_model)
        return chunk_repr  # (B, n_chunks, d_model)

    def forward(
        self,
        doc_emb:    torch.Tensor,
        doc_mask:   Optional[torch.Tensor] = None,
        apply_mask: bool  = False,
        mask_ratio: float = 0.125,
    ):
        """
        Returns
        -------
        chunk_repr : (B, n_chunks, d_model)
        masked_idx : (B,) indices of masked chunk, or None
        """
        B = doc_emb.shape[0]
        x = self.input_proj(doc_emb)
        x, doc_mask, D_padded = self._pad_input(x, doc_mask)

        chunk_repr = self._local_encode(x, doc_mask, B, D_padded)  # (B, n_chunks, d_model)

        masked_idx = None
        if apply_mask:
            masked_idx = torch.randint(0, self.n_chunks, (B,), device=doc_emb.device)
            mask_tok   = self.chunk_mask_token.unsqueeze(0).expand(B, -1)
            chunk_repr = chunk_repr.clone()
            batch_idx  = torch.arange(B, device=doc_emb.device)
            chunk_repr[batch_idx, masked_idx] = mask_tok

        chunk_repr = self.global_encoder(chunk_repr)
        chunk_repr = self.norm(chunk_repr)

        return chunk_repr, masked_idx  # (B, n_chunks, d_model), (B,) or None


# ─────────────────────────────────────────────────────────────────────────────
#  TSEncoder (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class TSEncoder(nn.Module):
    def __init__(
        self,
        d_model:    int,
        n_heads:    int,
        d_ff:       int,
        dropout:    float,
        patch_size: int   = 32,
        mask_ratio: float = 0.15,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        self.patch_embed   = nn.Linear(patch_size, d_model)
        self.patch_norm    = nn.LayerNorm(d_model)
        self.patch_pos_enc = PositionalEncoding(d_model, dropout, max_len=256)
        self.mask_token    = nn.Parameter(torch.randn(d_model))

        intra_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.intra_encoder = nn.TransformerEncoder(intra_layer, num_layers=1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer  = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.norm         = nn.LayerNorm(d_model)
        self.dropout      = nn.Dropout(dropout)
        self.regul_vector = nn.Parameter(torch.randn(d_model))

    def _make_patch_mask(self, BC, n_patches, device):
        n_mask = max(1, int(self.mask_ratio * n_patches))
        noise  = torch.rand(BC, n_patches, device=device)
        return noise.argsort(dim=1) < n_mask

    def _apply_patch_mask(self, x_p, mask):
        mask_tok = self.mask_token.view(1, 1, -1).expand_as(x_p)
        return torch.where(mask.unsqueeze(-1), mask_tok, x_p)

    def forward(self, x, target_channels, apply_mpm_mask=False):
        B, C, L = x.shape
        P = self.patch_size

        x_p = x.reshape(B * C, 1, L).unfold(-1, P, P).squeeze(1)
        x_p = self.patch_norm(self.patch_embed(x_p))  # (B*C, n_patches, d_model)

        n_patches  = x_p.shape[1]
        patch_mask = None

        if apply_mpm_mask:
            target_rows = torch.zeros(B * C, dtype=torch.bool, device=x.device)
            for b in range(B):
                for tc in target_channels:
                    target_rows[b * C + tc] = True

            full_mask = torch.zeros(B * C, n_patches, dtype=torch.bool, device=x.device)
            n_target  = target_rows.sum().item()
            if n_target > 0:
                sub_mask = self._make_patch_mask(int(n_target), n_patches, x.device)
                full_mask[target_rows] = sub_mask

            x_p        = self._apply_patch_mask(x_p, full_mask)
            patch_mask = full_mask

        x_p = self.patch_pos_enc(x_p)
        x_p = self.intra_encoder(x_p)

        x_pooled = x_p.mean(dim=1).reshape(B, C, -1)
        x_pooled = self.dropout(x_pooled)

        channel_is_zero = (x.abs().sum(dim=-1) == 0)
        all_zero = channel_is_zero.all(dim=-1)
        if all_zero.any():
            channel_is_zero[all_zero, 0] = False

        x_tf = self.transformer(x_pooled, src_key_padding_mask=channel_is_zero)
        x_tf = x_tf * (~channel_is_zero).float().unsqueeze(-1)
        x_tf = self.norm(x_tf)

        out = x_tf[:, target_channels, :]
        return out, patch_mask, x_p


# ─────────────────────────────────────────────────────────────────────────────
#  Fusion Layer
#  doc_encoder removed — HierarchicalDocEncoder runs upstream in _encode.
#  Receives (B, n_chunks, d_model) memory directly; cross-attends TS token.
# ─────────────────────────────────────────────────────────────────────────────

class TSDocFusionLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, **_):
        super().__init__()
        self.cross_decoder = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )

    def forward(self, z_ts, z_doc_chunks):
        """
        z_ts         : (B, d_model)
        z_doc_chunks : (B, n_chunks, d_model)  ← chunk representations from HierarchicalDocEncoder
        Returns      : (B, d_model)
        """
        tgt     = z_ts.unsqueeze(1)                      # (B, 1, d_model)
        z_fused = self.cross_decoder(tgt, z_doc_chunks)  # (B, 1, d_model)
        return z_fused.squeeze(1)                        # (B, d_model)


# ─────────────────────────────────────────────────────────────────────────────
#  Pre-training Heads (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class MPMHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

    def forward(self, patch_emb, mask):
        return self.proj(patch_emb[mask])


class MFMHead(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_ff, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, z_fused):
        return self.norm(self.proj(z_fused))

class MCHMHead(nn.Module):
    """
    Masked Chunk Head Modeling — reconstructs a masked chunk's
    global representation via MSE regression.
    Input : global chunk repr at masked position  (B, d_model)
    Output: reconstructed chunk repr              (B, d_model)
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_ff, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))  # (B, d_model)
# ─────────────────────────────────────────────────────────────────────────────
#  SignalingModelV4
# ─────────────────────────────────────────────────────────────────────────────
class HeadModel(nn.Module):
    """
    Head-only ablation with frozen TS+Doc encoders.

    Architecture:
        Same as TSDocFusionModel forward, but with all encoder outputs detached
        from the computational graph to prevent gradient flow.

    Forward returns:
        pred    : (B, 1)
        z_fused : (B, d_model)  fused representation (detached)
        0.0     : placeholder to match other models' 3-tuple return
    """

    def __init__(self, d_input: int, d_output: int ):
        super().__init__()
        self.layer_1 = nn.Linear(d_input, d_input*2)
        self.layer_3 = nn.Linear(d_input*2, d_input)
        self.activation = nn.GELU()
        self.out=nn.Linear(d_input, d_output)
        
    def forward(self, z_fused):
        out1 = self.layer_1(z_fused)
        out2 = self.activation(out1)
        out3 = (self.layer_3(out2)+z_fused)/2
        out4 = self.out(out3)
        return out4

class SignalingModelV4(nn.Module):

    def __init__(self, kwargs):
        super().__init__()

        d_ts            = kwargs.get("d_ts",            512)
        d_txt           = kwargs.get("d_txt",           768)
        d_model         = kwargs.get("d_model",         256)
        n_heads         = kwargs.get("n_heads",         4)
        d_ff            = kwargs.get("d_ff",            1024)
        dropout         = kwargs.get("dropout",         0.1)
        n_layers        = kwargs.get("n_layers",        1)
        patch_size      = kwargs.get("patch_size",      32)
        mask_ratio      = kwargs.get("mask_ratio",      0.15)
        mfm_mask_prob   = kwargs.get("mfm_mask_prob",   0.30)
        n_chunks        = kwargs.get("n_chunks",        8)
        n_local_layers  = kwargs.get("n_local_layers",  2)
        n_global_layers = kwargs.get("n_global_layers", 2)

        self.lr            = kwargs.get("lr",         1e-4)
        self.epoch         = kwargs.get("epoch",      7)
        self.lambda_sim    = kwargs.get("lambda_sim", 0.0)
        self.grad_clip     = kwargs.get("grad_clip",  5.0)
        self.mfm_mask_prob = mfm_mask_prob

        # ── TS encoder + projection ──────────────────────────────────────────
        self.ts_encoder = TSEncoder(
            d_model=d_ts, n_heads=n_heads, d_ff=d_ff, dropout=dropout,
            patch_size=patch_size, mask_ratio=mask_ratio,
        )
        self.ts_proj = nn.Linear(d_ts, d_model)

        # ── Hierarchical document encoder ───────────────────────────────────
        # Replaces the flat txt_proj + TSDocFusionLayer's internal doc_encoder.
        # Outputs (B, n_chunks, d_model) chunk representations.
        self.doc_encoder = HierarchicalDocEncoder(
            d_txt=d_txt, d_model=d_model,
            n_chunks=n_chunks, n_heads=n_heads, d_ff=d_ff, dropout=dropout,
            n_local_layers=n_local_layers, n_global_layers=n_global_layers,
        )

        # ── Fusion layers ────────────────────────────────────────────────────
        self.fusion = nn.ModuleList([
            TSDocFusionLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # ── Downstream head ──────────────────────────────────────────────────
        n_classes = kwargs.get("n_classes", 3)
        self.head  = HeadModel(d_model, n_classes)

        # ── Pre-training heads ───────────────────────────────────────────────
        self.mpm_head         = MPMHead(d_ts)
        self.mfm_head         = MFMHead(d_model, d_ff)
        self.mchm_head        = MCHMHead(d_model, d_ff)   # ← add this
        self.fused_mask_token = nn.Parameter(torch.randn(d_model))

        self._init_weights()

    # ── Weight init ──────────────────────────────────────────────────────────

    def _init_weights(self):
        for module in [self.ts_proj, self.head]:
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.zeros_(p)

    # ── Shared encode step ───────────────────────────────────────────────────

    def _encode(self, ts, doc_emb, doc_mask, target_channels, apply_mpm_mask=False):
        """
        Returns
        -------
        z_ts         : (B, d_model)
        z_doc_chunks : (B, n_chunks, d_model)
        patch_mask   : (B*C, n_patches) bool or None
        patch_emb    : (B*C, n_patches, d_ts)
        """
        z_ts_raw, patch_mask, patch_emb = self.ts_encoder(
            ts, target_channels, apply_mpm_mask=apply_mpm_mask
        )
        z_ts = self.ts_proj(z_ts_raw.squeeze(1))                 # (B, d_model)

        z_doc_chunks, _ = self.doc_encoder(doc_emb, doc_mask)   # (B, n_chunks, d_model)

        return z_ts, z_doc_chunks, patch_mask, patch_emb

    def _fuse(self, z_ts, z_doc_chunks):
        z_fused = self.fusion[0](z_ts, z_doc_chunks)
        for layer in self.fusion[1:]:
            z_fused = layer(z_fused, z_doc_chunks)
        return z_fused                                           # (B, d_model)

    # ── Pre-training Task 1: Masked Patch Modeling ───────────────────────────


    def pretrain_mchm(self, doc_emb, doc_mask=None):
        """
        Masked Chunk Head Modeling (MCHM).

        1. Run doc encoder (no mask) → clean chunk representations as targets.
        2. Run doc encoder (with mask) → one chunk replaced by chunk_mask_token.
        3. Extract masked chunk output → MCHMHead → MSE vs clean chunk repr.

        Returns
        -------
        mchm_loss : scalar
        """
        # Clean forward — reconstruction targets
        with torch.no_grad():
            clean_repr, _ = self.doc_encoder(doc_emb, doc_mask, apply_mask=False)
            # clean_repr: (B, n_chunks, d_model)

        # Masked forward
        masked_repr, masked_idx = self.doc_encoder(
            doc_emb, doc_mask, apply_mask=True
        )
        # masked_repr: (B, n_chunks, d_model)
        # masked_idx:  (B,) — which chunk was masked per sample

        B         = doc_emb.shape[0]
        batch_idx = torch.arange(B, device=doc_emb.device)

        pred   = self.mchm_head(masked_repr[batch_idx, masked_idx])  # (B, d_model)
        target = clean_repr[batch_idx, masked_idx]                   # (B, d_model)

        return F.mse_loss(pred, target)
    def pretrain_mpm(self, ts, doc_emb, target_channels, doc_mask=None):
        B, C, L = ts.shape

        with torch.no_grad():
            _, _, _, patch_emb_clean = self._encode(
                ts, doc_emb, doc_mask, target_channels, apply_mpm_mask=False
            )

        z_ts_raw, patch_mask, patch_emb_masked = self.ts_encoder(
            ts, target_channels, apply_mpm_mask=True
        )

        if patch_mask is None or patch_mask.sum() == 0:
            return torch.tensor(0.0, device=ts.device)

        target_rows = torch.zeros(B * C, dtype=torch.bool, device=ts.device)
        for b in range(B):
            for tc in target_channels:
                target_rows[b * C + tc] = True

        tc_mask_flat  = patch_mask[target_rows]
        tc_emb_masked = patch_emb_masked[target_rows]
        tc_emb_clean  = patch_emb_clean[target_rows]

        recon    = self.mpm_head(tc_emb_masked, tc_mask_flat)
        targets  = tc_emb_clean[tc_mask_flat]
        mpm_loss = F.mse_loss(recon, targets)

        return mpm_loss

    # ── Pre-training Task 2: Masked Fusion Modeling ──────────────────────────

    def pretrain_mfm(self, ts, doc_emb, target_channels, doc_mask=None):
        z_ts, z_doc_chunks, _, _ = self._encode(
            ts, doc_emb, doc_mask, target_channels, apply_mpm_mask=False
        )

        with torch.no_grad():
            z_fused_clean = self._fuse(z_ts, z_doc_chunks)

        B    = z_ts.shape[0]
        mask = torch.rand(B, device=ts.device) < self.mfm_mask_prob
        mask_tok    = self.fused_mask_token.unsqueeze(0).expand(B, -1)
        z_ts_masked = torch.where(mask.unsqueeze(-1), mask_tok, z_ts)

        z_fused_masked = self._fuse(z_ts_masked, z_doc_chunks)
        z_recon        = self.mfm_head(z_fused_masked)
        mfm_loss       = F.mse_loss(z_recon[mask], z_fused_clean[mask])

        return mfm_loss

    # ── Fine-tuning / inference ───────────────────────────────────────────────

    def forward(self, ts, doc_emb, doc_mask=None, target_channels=None, freeze_encoder=False):
        if target_channels is None:
            target_channels = [0]

        z_ts, z_doc_chunks, _, _ = self._encode(
            ts, doc_emb, doc_mask, target_channels, apply_mpm_mask=False
        )
        z_fused = self._fuse(z_ts, z_doc_chunks)

        cpi_pred = self.head(z_fused)

        return cpi_pred, z_fused

    # ── Pre-training loop ────────────────────────────────────────────────────

    def pretrain(self, dataloader, label_percentages=None):
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        model  = self
        model.train()

        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=self.epoch
        )

        print("\n" + "═" * 60)
        print("  PRE-TRAINING  — MPM + MFM + MCHM  (joint)")
        print("═" * 60)

        for epoch in range(self.epoch):
            totals    = {"mpm": 0.0, "mfm": 0.0, "mchm": 0.0, "total": 0.0}
            n_batches = 0

            for embd, ctx, mask, _ in dataloader:
                embd      = embd.to(DEVICE)
                ctx, mask = ctx.to(DEVICE), mask.to(DEVICE)

                optimizer.zero_grad()

                # ── MPM ──────────────────────────────────────────────────────
                try:
                    mpm_loss = model.pretrain_mpm(
                        ts=embd, doc_emb=ctx,
                        target_channels=[0], doc_mask=mask,
                    )
                except ValueError as e:
                    logging.warning("MPM skipped — %s", e)
                    continue

                # ── MFM ──────────────────────────────────────────────────────
                try:
                    mfm_loss = model.pretrain_mfm(
                        ts=embd, doc_emb=ctx,
                        target_channels=[0], doc_mask=mask,
                    )
                except ValueError as e:
                    logging.warning("MFM skipped — %s", e)
                    continue

                # ── MCHM ─────────────────────────────────────────────────────
                try:
                    mchm_loss = model.pretrain_mchm(doc_emb=ctx, doc_mask=mask)
                except ValueError as e:
                    logging.warning("MCHM skipped — %s", e)
                    continue

                loss = mpm_loss + mfm_loss + mchm_loss

                if torch.isnan(loss) or torch.isinf(loss):
                    logging.warning(
                        "NaN/Inf joint loss at epoch=%d — skipping batch", epoch + 1
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue

                loss.backward()
                logging.info(
                    f"  epoch={epoch+1}/{self.epoch}  batch={n_batches}"
                    f"  loss={loss.item():.6f}"
                )
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                optimizer.step()

                totals["mpm"]   += mpm_loss.item()
                totals["mfm"]   += mfm_loss.item()
                totals["mchm"]  += mchm_loss.item()
                totals["total"] += loss.item()
                n_batches       += 1

            n = max(n_batches, 1)
            print(
                f"── epoch={epoch+1:>3}  "
                f"total={totals['total']/n:.4f}  "
                f"mpm={totals['mpm']/n:.4f}  "
                f"mfm={totals['mfm']/n:.4f}  "
                f"mchm={totals['mchm']/n:.4f}"
            )
            scheduler.step()
    # ── Fit ──────────────────────────────────────────────────────────────────

    def _reset_nan_params(self):
        with torch.no_grad():
            for name, p in self.named_parameters():
                bad = torch.isnan(p) | torch.isinf(p)
                if bad.any():
                    logging.warning("Resetting %d NaN/Inf values in %s", bad.sum().item(), name)
                    p[bad] = 0.0

    def fit(self, dataloader, label_percentages=None):
        DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
        model     = self
        self.pretrain(dataloader)
        self._reset_nan_params()
        model.train()

        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=2)
        criterion = (
            torch.nn.CrossEntropyLoss(weight=label_percentages)
            if label_percentages is not None else torch.nn.CrossEntropyLoss()
        )

        for epoch in range(self.epoch):
            epoch_loss, n_batches = 0.0, 0
            for embd, ctx, mask, target in dataloader:
                embd      = embd.to(DEVICE)
                ctx, mask = ctx.to(DEVICE), mask.to(DEVICE)
                target    = target.to(DEVICE)
                optimizer.zero_grad()

                pred, _ = model(embd, ctx, mask)
                loss    = criterion(pred, target.squeeze(-1).long())

                if torch.isnan(loss) or torch.isinf(loss):
                    logging.warning("NaN/Inf loss at epoch=%d batch=%d — skipping", epoch+1, n_batches+1)
                    optimizer.zero_grad()
                    continue

                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches  += 1
                logging.info(
                    f"  epoch={epoch+1}/{self.epoch}  batch={n_batches}"
                    f"  loss={loss.item():.6f}  grad={grad_norm:.3f}"
                )

            scheduler.step()
            logging.info(
                f"── epoch={epoch+1}  avg_loss={epoch_loss/max(n_batches,1):.6f}"
                f"  lr={scheduler.get_last_lr()[0]:.2e}"
            )

        model.eval()
        return model


    def inference(
        self,
        input_data,
        confidence_level: float,
        risk_method:   str   = "fixed_fractional",
        account_value: float = None,
        entry_price:   float = None,
        risk_kwargs:   dict  = None,
    ) -> dict:
        self.eval()
        risk_kwargs = risk_kwargs or {}

        with torch.no_grad():
            pred, _    = self(*input_data)
            probs      = F.softmax(pred, dim=-1)   # (B, n_classes)

            # Take first sample — inference is single-sample by contract
            probs_1    = probs[0]                  # (n_classes,)
            confidence = probs_1.max().item()
            pred_idx   = probs_1.argmax().item()   # scalar ✅

        signal = pred_idx - 1  # {0,1,2} → {-1, 0, 1}

        if confidence < confidence_level or signal == 0:
            return {
                "estimated_action": "hold",
                "signal":           0,
                "probability":      round(confidence, 4),
                "volume":           0.0,
                "notional":         0.0,
                "stop_loss_price":  None,
                "risk_amount":      0.0,
                "sizing_method":    None,
                "warnings":         [],
            }

        direction     = "BUY" if signal == 1 else "SELL"
        risk_result   = None
        risk_warnings = []

        if account_value is not None and entry_price is not None:
            if risk_method == "kelly" and "confidence" not in risk_kwargs:
                risk_kwargs["confidence"] = confidence
            try:
                risk_result   = compute_position_size(
                    method=risk_method, account_value=account_value,
                    entry_price=entry_price, signal_direction=direction,
                    **risk_kwargs,
                )
                risk_warnings = risk_result.warnings
            except Exception as e:
                logger.warning(f"S3 sizing failed: {e}")

        return {
            "estimated_action": direction,
            "signal":           signal,
            "probability":      round(confidence, 4),
            "probabilities": {
                "sell": round(probs_1[0].item(), 4),
                "hold": round(probs_1[1].item(), 4),
                "buy":  round(probs_1[2].item(), 4),
            },
            "volume":          round(risk_result.position_size, 4) if risk_result else None,
            "notional":        round(risk_result.notional, 2)       if risk_result else None,
            "stop_loss_price": round(risk_result.stop_loss_price, 4) if risk_result else None,
            "risk_amount":     round(risk_result.risk_amount, 2)     if risk_result else None,
            "sizing_method":   risk_result.sizing_method_used        if risk_result else None,
            "warnings":        risk_warnings,
        }
    

if __name__ == "__main__":
    import torch

    print("=" * 50)
    print("  SignalingModelV4 — Smoke Test")
    print("=" * 50)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    CFG = {
        "d_ts": 64, "d_txt": 32, "d_model": 32,
        "n_heads": 4, "d_ff": 64, "dropout": 0.0,
        "n_layers": 1, "patch_size": 8, "mask_ratio": 0.5,
        "mfm_mask_prob": 0.5, "n_chunks": 4,
        "n_local_layers": 1, "n_global_layers": 1,
        "n_classes": 3, "lr": 1e-3, "epoch": 1,
        "lambda_sim": 0.0, "grad_clip": 5.0,
    }

    B, C, L, D_DOC = 2, 3, 32, 8

    def make_batch():
        ts       = torch.randn(B, C, L)
        doc_emb  = torch.randn(B, D_DOC, CFG["d_txt"])
        doc_mask = torch.ones(B, D_DOC, dtype=torch.bool)
        target   = torch.randint(0, CFG["n_classes"], (B, 1))
        return ts, doc_emb, doc_mask, target

    model = SignalingModelV4(CFG).to(DEVICE)
    print(f"  device : {DEVICE}")
    print(f"  params : {sum(p.numel() for p in model.parameters()):,}\n")

    ts, doc_emb, doc_mask, target = make_batch()

    # 1 — forward
    pred, z_fused = model(ts, doc_emb, doc_mask)
    assert pred.shape    == (B, CFG["n_classes"])
    assert z_fused.shape == (B, CFG["d_model"])
    assert not torch.isnan(pred).any()
    print("  ✅  forward pass")

    # 2 — pretrain MPM
    model.train()
    mpm_loss = model.pretrain_mpm(ts, doc_emb, target_channels=[0], doc_mask=doc_mask)
    assert not torch.isnan(mpm_loss)
    print(f"  ✅  pretrain_mpm   loss={mpm_loss.item():.4f}")

    # 3 — pretrain MFM
    mfm_loss = model.pretrain_mfm(ts, doc_emb, target_channels=[0], doc_mask=doc_mask)
    assert not torch.isnan(mfm_loss)
    print(f"  ✅  pretrain_mfm   loss={mfm_loss.item():.4f}")

    # 4 — fit (1 epoch, 2 mini-batches)
    dataloader = [make_batch() for _ in range(2)]
    model.fit(dataloader)
    print("  ✅  fit loop")

    # 5 — inference
    model.eval()
    result = model.inference(
        input_data=(ts, doc_emb, doc_mask),
        confidence_level=0.0,
    )
    assert result["estimated_action"] in ("BUY", "SELL", "hold")
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-4
    print(f"  ✅  inference      action={result['estimated_action']}  "
          f"p={result['probability']:.3f}")

    print("\n  all checks passed ✅")
    print("=" * 50)