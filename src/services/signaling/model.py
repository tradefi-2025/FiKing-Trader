import os
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from .config import SignalingConfig
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR

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
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


# ─────────────────────────────────────────────────────────────────────────────
#  Vector Quantizer (EMA-updated codebook)
# ─────────────────────────────────────────────────────────────────────────────

class VectorQuantizer(nn.Module):
    """
    Straight-through estimator VQ with optional EMA codebook updates.

    Forward returns:
        z_q      : quantized tensor, same shape as z_e      (B, T, d_model)
        indices  : codebook indices                          (B, T)
        vq_loss  : commitment + codebook loss (scalar)
    """

    def __init__(
        self,
        codebook_size: int,
        d_model:       int,
        commitment_cost: float = 0.25,
        ema_decay:     float   = 0.99,
        use_ema:       bool    = True,
    ):
        super().__init__()
        self.codebook_size   = codebook_size
        self.d_model         = d_model
        self.commitment_cost = commitment_cost
        self.use_ema         = use_ema

        # Codebook
        self.embedding = nn.Embedding(codebook_size, d_model)
        nn.init.uniform_(self.embedding.weight, -1 / codebook_size, 1 / codebook_size)

        if use_ema:
            self.register_buffer('ema_cluster_size', torch.zeros(codebook_size))
            self.register_buffer('ema_w', self.embedding.weight.data.clone())
            self.ema_decay = ema_decay

    def forward(self, z_e: torch.Tensor):
        # z_e: (B, T, d_model) or (N, d_model) — flatten/unflatten internally
        shape = z_e.shape
        flat  = z_e.reshape(-1, self.d_model)               # (N, d_model)

        # L2 distances to each codebook vector
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1)
        )                                                    # (N, codebook_size)
        indices_flat = dist.argmin(dim=1)                    # (N,)
        z_q_flat     = self.embedding(indices_flat)          # (N, d_model)

        # EMA codebook update (training only)
        if self.use_ema and self.training:
            with torch.no_grad():
                one_hot = F.one_hot(indices_flat, self.codebook_size).float()
                n_i     = one_hot.sum(0)                     # (codebook_size,)
                self.ema_cluster_size = (
                    self.ema_decay * self.ema_cluster_size + (1 - self.ema_decay) * n_i
                )
                dw = one_hot.t() @ flat                      # (codebook_size, d_model)
                self.ema_w = (
                    self.ema_decay * self.ema_w + (1 - self.ema_decay) * dw
                )
                # Laplace smoothing
                n = self.ema_cluster_size.sum()
                smoothed = (self.ema_cluster_size + 1e-5) / (n + self.codebook_size * 1e-5) * n
                self.embedding.weight.data.copy_(self.ema_w / smoothed.unsqueeze(1))

        # VQ loss: commitment (push encoder → codebook), codebook (EMA handles if ema=True)
        if self.use_ema:
            vq_loss = self.commitment_cost * F.mse_loss(z_q_flat.detach(), flat)
        else:
            vq_loss = (
                F.mse_loss(z_q_flat.detach(), flat)          # codebook loss
                + self.commitment_cost * F.mse_loss(z_q_flat, flat.detach())  # commitment
            )

        # Straight-through estimator: gradients flow through z_e unchanged
        z_q_flat = flat + (z_q_flat - flat).detach()

        z_q     = z_q_flat.reshape(shape)                   # restore original shape
        indices = indices_flat.reshape(shape[:-1])           # (B, T) or (N,)
        return z_q, indices, vq_loss


# ─────────────────────────────────────────────────────────────────────────────
#  TSEncoder with VQ codebook
# ─────────────────────────────────────────────────────────────────────────────

class TSEncoder(nn.Module):
    """
    Architecture:
      1. Patchify each channel (B, C, L) → (B*C, n_patches, d_model)
      2. Positional encoding over patches
      3. Intra-channel Transformer over patches
      4. Vector Quantization of patch embeddings (codebook)
      5. Pool patches → (B, C, d_model)
      6. Cross-channel Transformer → (B, C, d_model)

    Pre-training hooks:
      - mask_patches()  : mask a subset of patches (for MPM)
      - mask_channels() : mask output channel tokens (for MFM from outside)
    """

    def __init__(
        self,
        d_model:       int,
        n_heads:       int,
        d_ff:          int,
        dropout:       float,
        patch_size:    int  = 32,
        codebook_size: int  = 512,
        commitment_cost: float = 0.25,
        mask_ratio:    float = 0.15,   # fraction of patches to mask in MPM
    ):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        # Stage 1: Patch embedding
        self.patch_embed   = nn.Linear(patch_size, d_model)
        self.patch_norm    = nn.LayerNorm(d_model)
        self.patch_pos_enc = PositionalEncoding(d_model, dropout, max_len=256)

        # Learnable MASK token (used during MPM)
        self.mask_token = nn.Parameter(torch.randn(d_model))

        # Stage 2: Intra-channel Transformer
        intra_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.intra_encoder = nn.TransformerEncoder(intra_layer, num_layers=1)

        # VQ codebook (applied per patch AFTER intra_encoder)
        self.vq = VectorQuantizer(codebook_size, d_model, commitment_cost)

        # Stage 3: Cross-channel Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer   = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.norm          = nn.LayerNorm(d_model)
        self.dropout       = nn.Dropout(dropout)
        self.regul_vector  = nn.Parameter(torch.randn(d_model))

    # ── Masking helpers ──────────────────────────────────────────────────────

    def _make_patch_mask(self, BC: int, n_patches: int, device) -> torch.BoolTensor:
        """Returns a (BC, n_patches) bool mask — True = masked position."""
        n_mask = max(1, int(self.mask_ratio * n_patches))
        noise  = torch.rand(BC, n_patches, device=device)
        mask   = noise.argsort(dim=1) < n_mask                # (BC, n_patches)
        return mask

    def _apply_patch_mask(self, x_p: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """Replace masked patch positions with the learnable mask token."""
        mask_tok = self.mask_token.view(1, 1, -1).expand_as(x_p)
        return torch.where(mask.unsqueeze(-1), mask_tok, x_p)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        x:               torch.Tensor,      # (B, C, L)
        target_channels: list[int],
        apply_mpm_mask:  bool = False,       # True during MPM pre-training
    ):
        B, C, L = x.shape
        P = self.patch_size

        # ── Stage 1: Patchify ────────────────────────────────────────────
        x_p = x.reshape(B * C, 1, L).unfold(-1, P, P).squeeze(1)   # (B*C, n_patches, P)
        x_p = self.patch_norm(self.patch_embed(x_p))                 # (B*C, n_patches, d_model)

        n_patches = x_p.shape[1]

        # MPM masking: only on target channels
        patch_mask = None
        if apply_mpm_mask:
            # Identify which rows in (B*C) correspond to target channels
            target_rows = torch.zeros(B * C, dtype=torch.bool, device=x.device)
            for b in range(B):
                for tc in target_channels:
                    target_rows[b * C + tc] = True

            # Build patch mask only for target channel rows
            full_mask = torch.zeros(B * C, n_patches, dtype=torch.bool, device=x.device)
            n_target  = target_rows.sum().item()
            if n_target > 0:
                sub_mask = self._make_patch_mask(int(n_target), n_patches, x.device)
                full_mask[target_rows] = sub_mask

            x_p       = self._apply_patch_mask(x_p, full_mask)
            patch_mask = full_mask                                    # (B*C, n_patches)

        # ── Stage 2: PE + Intra-channel Transformer ──────────────────────
        x_p = self.patch_pos_enc(x_p)
        x_p = self.intra_encoder(x_p)                               # (B*C, n_patches, d_model)

        # ── VQ Quantization ──────────────────────────────────────────────
        x_q, vq_indices, vq_loss = self.vq(x_p)                    # (B*C, n_patches, d_model)

        if torch.isnan(x_q).any():
            raise ValueError("NaN after VQ")

        # Pool patches → single vector per channel
        x_pooled = x_q.mean(dim=1).reshape(B, C, -1)               # (B, C, d_model)
        x_pooled = self.dropout(x_pooled)

        # ── Stage 3: Cross-channel Transformer ───────────────────────────
        channel_is_zero = (x.abs().sum(dim=-1) == 0)               # (B, C)
        all_zero = channel_is_zero.all(dim=-1)
        if all_zero.any():
            channel_is_zero[all_zero, 0] = False

        x_tf = self.transformer(x_pooled, src_key_padding_mask=channel_is_zero)
        x_tf = x_tf * (~channel_is_zero).float().unsqueeze(-1)
        x_tf = self.norm(x_tf)                                      # (B, C, d_model)

        out = x_tf[:, target_channels, :]                           # (B, len(tc), d_model)

        
        return out, vq_loss, vq_indices, x_p

    


# ─────────────────────────────────────────────────────────────────────────────
#  Fusion Layer (unchanged architecture, Transformer decoder)
# ─────────────────────────────────────────────────────────────────────────────

class TSDocFusionLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, **_):
        super().__init__()
        doc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.doc_encoder = nn.TransformerEncoder(doc_layer, num_layers=1)
        self.cross_decoder = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )

    def forward(self, z_ts, z_doc, doc_mask=None):
        pad_mask = ~doc_mask if doc_mask is not None else None
        if pad_mask is not None:
            all_masked = pad_mask.all(dim=-1)
            if all_masked.any():
                pad_mask[all_masked, 0] = False

        z_doc = self.doc_encoder(z_doc, src_key_padding_mask=pad_mask)
        if doc_mask is not None:
            z_doc = z_doc * doc_mask.float().unsqueeze(-1)

        tgt     = z_ts.unsqueeze(1)                                  # (B, 1, d_model)
        z_fused = self.cross_decoder(tgt, z_doc, memory_key_padding_mask=pad_mask)
        return z_fused.squeeze(1)                                    # (B, d_model)


# ─────────────────────────────────────────────────────────────────────────────
#  Pre-training Heads
# ─────────────────────────────────────────────────────────────────────────────

class MPMHead(nn.Module):
    """
    Task 1 — Masked Patch Modeling inside TS modality (target channels only).
    Predicts the VQ codebook index of each masked patch.
    Input : patch embeddings BEFORE quantization (B*C, n_patches, d_model)
    Output: logits over codebook                 (n_masked, codebook_size)
    """
    def __init__(self, d_model: int, codebook_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, codebook_size),
        )

    def forward(self, patch_emb: torch.Tensor, mask: torch.BoolTensor):
        # patch_emb: (B*C, n_patches, d_model), mask: (B*C, n_patches)
        masked_emb = patch_emb[mask]          # (n_masked, d_model)
        return self.proj(masked_emb)          # (n_masked, codebook_size)


class MFMHead(nn.Module):
    """
    Task 2 — Masked Fusion Modeling after cross-attention.
    Masks fused tokens (B, d_model) and reconstructs them via regression.
    Input : fused z tokens  (B, d_model)
    Output: reconstructed   (B, d_model)
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

    def forward(self, z_fused: torch.Tensor):
        return self.norm(self.proj(z_fused))  # (B, d_model)
    



class SignalingModelV4(nn.Module):

    def __init__(self, kwargs):
        super().__init__()   # ← must be FIRST

        # extract all hyperparams cleanly from kwargs
        d_ts            = kwargs.get("d_ts",            512)
        d_txt           = kwargs.get("d_txt",           768)
        d_model         = kwargs.get("d_model",         256)
        n_heads         = kwargs.get("n_heads",         4)
        d_ff            = kwargs.get("d_ff",            1024)
        dropout         = kwargs.get("dropout",         0.1)
        n_layers        = kwargs.get("n_layers",        1)
        patch_size      = kwargs.get("patch_size",      32)
        codebook_size   = kwargs.get("codebook_size",   512)
        commitment_cost = kwargs.get("commitment_cost", 0.25)
        mask_ratio      = kwargs.get("mask_ratio",      0.15)
        mfm_mask_prob   = kwargs.get("mfm_mask_prob",   0.30)

        # training hyperparams stored as attributes
        self.lr           = kwargs.get("lr",         1e-4)
        self.epoch        = kwargs.get("epoch",     7)
        self.lambda_vq    = kwargs.get("lambda_vq",  1.0)
        self.lambda_sim   = kwargs.get("lambda_sim", 0.0)
        self.grad_clip    = kwargs.get("grad_clip",  5.0)
        self.mfm_mask_prob = mfm_mask_prob

        # Encoder
        self.ts_encoder = TSEncoder(
            d_model=d_ts, n_heads=n_heads, d_ff=d_ff, dropout=dropout,
            patch_size=patch_size, codebook_size=codebook_size,
            commitment_cost=commitment_cost, mask_ratio=mask_ratio,
        )
        self.ts_proj  = nn.Linear(d_ts,   d_model)
        self.txt_proj = nn.Linear(d_txt,  d_model)

        # Fusion layers
        self.fusion = nn.ModuleList([
            TSDocFusionLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Downstream head — outputs n_classes logits (3: sell/hold/buy)
        n_classes   = kwargs.get("n_classes", 3)
        self.head   = nn.Linear(d_model, n_classes)

        # Pre-training heads
        self.mpm_head        = MPMHead(d_ts, codebook_size)
        self.mfm_head        = MFMHead(d_model, d_ff)
        self.fused_mask_token = nn.Parameter(torch.randn(d_model))

        self._init_weights()

    # ── Weight init ──────────────────────────────────────────────────────────

    def _init_weights(self):
        for module in [self.ts_proj, self.txt_proj, self.head]:
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.zeros_(p)

    # ── Shared encode step ───────────────────────────────────────────────────

    def _encode(self, ts, doc_emb, target_channels, apply_mpm_mask=False):
        z_ts_raw, vq_loss, vq_indices, patch_emb = \
            self.ts_encoder(ts, target_channels, apply_mpm_mask=apply_mpm_mask)
        z_ts  = self.ts_proj(z_ts_raw.squeeze(1))           # (B, d_model)
        z_doc = self.txt_proj(doc_emb)                      # (B, D, d_model)
        return z_ts, z_doc, vq_loss, vq_indices, patch_emb

    def _fuse(self, z_ts, z_doc, doc_mask):
        z_fused = self.fusion[0](z_ts, z_doc, doc_mask)
        for layer in self.fusion[1:]:
            z_fused = layer(z_fused, z_doc, doc_mask)
        return z_fused                                       # (B, d_model)

    # ── Pre-training Task 1: Masked Patch Modeling ───────────────────────────

    def pretrain_mpm(self, ts, doc_emb, target_channels, doc_mask=None):
        """
        Mask patches of target channels, encode through TSEncoder,
        then predict masked VQ codebook indices.

        Returns:
            mpm_loss   : cross-entropy over masked patch indices
            vq_loss    : VQ commitment/codebook loss
            similarity : cosine similarity between estimated and actual targets
        """
        B, C, L = ts.shape

        z_ts_raw, vq_loss, vq_indices, patch_emb = \
            self.ts_encoder(ts, target_channels, apply_mpm_mask=True)

        # Recover the patch mask built inside TSEncoder
        # We rebuild it deterministically from the same target_rows logic
        # NOTE: in practice store the mask as a return value from ts_encoder.
        # Here we use the indices returned from the VQ as targets.

        # vq_indices shape: (B*C, n_patches) — use target channel rows as supervision
        n_patches  = vq_indices.shape[1]
        target_rows = torch.zeros(B * C, dtype=torch.bool, device=ts.device)
        for b in range(B):
            for tc in target_channels:
                target_rows[b * C + tc] = True

        # patch_emb: (B*C, n_patches, d_ts) — post intra_encoder, pre VQ pool
        # mpm_head predicts codebook indices from the masked positions
        # We reuse the mask stored in the encoder (refactored below to expose it)
        # For simplicity: use all target-channel patches as supervision signal
        tc_patch_emb = patch_emb[target_rows]               # (B*n_tc, n_patches, d_ts)
        tc_vq_idx    = vq_indices[target_rows]              # (B*n_tc, n_patches)

        # Flatten patches and predict
        flat_emb = tc_patch_emb.reshape(-1, tc_patch_emb.shape[-1])   # (N, d_ts)
        flat_idx = tc_vq_idx.reshape(-1)                               # (N,)
        logits   = self.mpm_head.proj(flat_emb)                        # (N, codebook_size)
        mpm_loss = F.cross_entropy(logits, flat_idx)

        z_ts  = self.ts_proj(z_ts_raw.squeeze(1))

        return mpm_loss, vq_loss, 0

    # ── Pre-training Task 2: Masked Fusion Modeling ──────────────────────────

    def pretrain_mfm(self, ts, doc_emb, target_channels, doc_mask=None):
        """
        Encode TS (no patch masking), get fused token z_fused.
        Randomly replace z_fused with mask token, then reconstruct via MFMHead.

        Returns:
            mfm_loss   : MSE reconstruction of fused token
            vq_loss    : VQ commitment/codebook loss
            similarity : cosine similarity between estimated and actual targets
        """
        z_ts, z_doc, vq_loss, _, _ = self._encode(
            ts, doc_emb, target_channels, apply_mpm_mask=False
        )

        # Get clean fused token as regression target
        with torch.no_grad():
            z_fused_clean = self._fuse(z_ts, z_doc, doc_mask)   # (B, d_model)

        # Stochastically mask the fused token
        B = z_ts.shape[0]
        mask = torch.rand(B, device=ts.device) < self.mfm_mask_prob  # (B,)
        mask_tok = self.fused_mask_token.unsqueeze(0).expand(B, -1)
        z_ts_masked = torch.where(mask.unsqueeze(-1), mask_tok, z_ts)

        # Fuse with masked TS token
        z_fused_masked = self._fuse(z_ts_masked, z_doc, doc_mask)    # (B, d_model)

        # Reconstruct original fused token
        z_recon  = self.mfm_head(z_fused_masked)                     # (B, d_model)
        mfm_loss = F.mse_loss(z_recon[mask], z_fused_clean[mask])


        return mfm_loss, vq_loss, 0

    # ── Fine-tuning / inference ───────────────────────────────────────────────

    def forward(self, ts, doc_emb, doc_mask=None, target_channels=None,freeze_encoder=False):
        if target_channels is None:
            target_channels = [0]

        
        z_ts, z_doc, vq_loss, _, _ = self._encode(
            ts, doc_emb, target_channels, apply_mpm_mask=False
        )

        z_fused  = self._fuse(z_ts, z_doc, doc_mask)
        
        if freeze_encoder:
            cpi_pred=self.head(z_fused.detach())                                # (B, 1)
        else:
            cpi_pred = self.head(z_fused)                                 # (B, 1)

        return cpi_pred, z_fused,0, vq_loss
    
    def pretrain(self, dataloader,label_percentages=None):
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = self
        model.train()
        epochs_mpm=self.epoch
        epochs_mfm=self.epoch
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-5)

        # ── Phase 1: MPM ─────────────────────────────────────────────────────────
        print("\n" + "═" * 50)
        print("  PRE-TRAINING  Phase 1 — Masked Patch Modeling (MPM)")
        print("═" * 50)

        scheduler_mpm = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=epochs_mpm,
        )

        for epoch in range(epochs_mpm):
            epoch_loss, epoch_mpm, epoch_vq, epoch_sim = 0.0, 0.0, 0.0, 0.0
            n_batches = 0

            for embd,ctx,mask, _ in dataloader:
                embd = embd.to(DEVICE)
                ctx, mask = ctx.to(DEVICE), mask.to(DEVICE)

                optimizer.zero_grad()

                try:
                    mpm_loss, vq_loss, similarity = model.pretrain_mpm(
                        ts=embd,
                        doc_emb=ctx,
                        target_channels=[0],
                        doc_mask=mask,
                    )
                except ValueError as e:
                    logging.warning("MPM step skipped — %s", e)
                    continue

                loss = mpm_loss + self.lambda_vq * vq_loss + self.lambda_sim * (1 - similarity)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.warning("NaN/Inf MPM loss at epoch=%d — skipping batch", epoch + 1)
                    optimizer.zero_grad(set_to_none=True)
                    continue

                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)

                optimizer.step()

                epoch_loss += loss.item()
                epoch_mpm  += mpm_loss.item()
                epoch_vq   += vq_loss.item()
                epoch_sim  += 0
                n_batches  += 1

                # print(
                #     f"  [MPM] epoch={epoch+1}/{epochs_mpm}  batch={n_batches}"
                #     f"  loss={loss.item():.5f}"
                #     f"  mpm={mpm_loss.item():.5f}"
                #     f"  vq={vq_loss.item():.5f}"
                #     f"  grad={grad_norm:.4f}"
                # )

            n = max(n_batches, 1)
            print(
                f"── [MPM] epoch={epoch+1}  avg_loss={epoch_loss/n:.5f}"
                f"  avg_mpm={epoch_mpm/n:.5f}"
                f"  avg_vq={epoch_vq/n:.5f}"
                f"  avg_sim={epoch_sim/n:.4f}"
            )
            scheduler_mpm.step()

        # ── Phase 2: MFM ─────────────────────────────────────────────────────────
        print("\n" + "═" * 50)
        print("  PRE-TRAINING  Phase 2 — Masked Fusion Modeling (MFM)")
        print("═" * 50)

        optimizer = optim.Adam(model.parameters(), lr=self.lr * 0.5, weight_decay=1e-5)
        scheduler_mfm = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=epochs_mfm,
        )

        for epoch in range(epochs_mfm):
            epoch_loss, epoch_mfm, epoch_vq, epoch_sim = 0.0, 0.0, 0.0, 0.0
            n_batches = 0

            for embd, ctx, mask, _ in dataloader:
                embd = embd.to(DEVICE)
                ctx, mask = ctx.to(DEVICE), mask.to(DEVICE)

                optimizer.zero_grad()

                try:
                    mfm_loss, vq_loss, similarity = model.pretrain_mfm(
                        ts=embd,
                        doc_emb=ctx,
                        target_channels=[0],
                        doc_mask=mask,
                    )
                except ValueError as e:
                    logging.warning("MFM step skipped — %s", e)
                    continue

                loss = mfm_loss + self.lambda_vq * vq_loss + self.lambda_sim * (1 - similarity)

                if torch.isnan(loss) or torch.isinf(loss):
                    logging.warning("NaN/Inf MFM loss at epoch=%d — skipping batch", epoch + 1)
                    optimizer.zero_grad(set_to_none=True)

                    continue

                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)

                optimizer.step()

                epoch_loss += loss.item()
                epoch_mfm  += mfm_loss.item()
                epoch_vq   += vq_loss.item()
                epoch_sim  += 0
                n_batches  += 1

                # print(
                #     f"  [MFM] epoch={epoch+1}/{epochs_mfm}  batch={n_batches}"
                #     f"  loss={loss.item():.5f}"
                #     f"  mfm={mfm_loss.item():.5f}"
                #     f"  vq={vq_loss.item():.5f}"
                #     f"  grad={grad_norm:.4f}"
                # )

            n = max(n_batches, 1)
            print(
                f"── [MFM] epoch={epoch+1}  avg_loss={epoch_loss/n:.5f}"
                f"  avg_mfm={epoch_mfm/n:.5f}"
                f"  avg_vq={epoch_vq/n:.5f}"
                f"  avg_sim={epoch_sim/n:.4f}"
            )
            scheduler_mfm.step()

    def _reset_nan_params(self):
        """Replace any NaN/Inf weights left by a bad pretrain step."""
        with torch.no_grad():
            for name, p in self.named_parameters():
                bad = torch.isnan(p) | torch.isinf(p)
                if bad.any():
                    logging.warning("Resetting %d NaN/Inf values in %s", bad.sum().item(), name)
                    p[bad] = 0.0
    def fit(self, dataloader,label_percentages=None):
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        lr = self.lr
        epochs = self.epoch
        lambda_vq = self.lambda_vq
        grad_clip = self.grad_clip
        self.pretrain(dataloader)
        self._reset_nan_params()          # ← flush any corruption from pretrain

        model = self
        model.train()

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=2)
        criterion = torch.nn.CrossEntropyLoss(weight=label_percentages) if label_percentages is not None else torch.nn.CrossEntropyLoss()
        for epoch in range(self.epoch):
            epoch_loss, n_batches = 0.0, 0

            for embd, ctx, mask, target in dataloader:
                embd        = embd.to(DEVICE)
                ctx, mask   = ctx.to(DEVICE), mask.to(DEVICE)
                target      = target.to(DEVICE)

                optimizer.zero_grad()
                pred, _,_, vq_loss = model(embd, ctx, mask)
                loss = criterion(pred,  target.squeeze(-1).long()) + lambda_vq * vq_loss.clamp(max=1.0)

                if torch.isnan(loss) or torch.isinf(loss):
                    logging.warning("NaN/Inf loss at epoch=%d batch=%d — skipping", epoch+1, n_batches+1)
                    optimizer.zero_grad()
                    continue

                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches  += 1
                logging.info(f"  epoch={epoch+1}/{epochs}  batch={n_batches}"
                    f"  loss={loss.item():.6f}  vq={vq_loss.item():.4f}"
                    f"  grad={grad_norm:.3f}")

            scheduler.step()
            logging.info(f"── epoch={epoch+1}  avg_loss={epoch_loss/max(n_batches,1):.6f}"
                f"  lr={scheduler.get_last_lr()[0]:.2e}")

        model.eval()
        return model


    def test(self,test_dataset):
        ts,ctx,mask, target = test_dataset
        batch_size = self.config.batch_size
        self.eval()
        with torch.no_grad():
            ts,ctx,mask,target = ts.chunk(batch_size), ctx.chunk(batch_size), mask.chunk(batch_size), target.chunk(batch_size)
            for i in range(len(ts)):
                pred, _ ,_,_= self(ts[i], ctx[i], mask[i])
                logger.info(f"Predictions: {pred}")

                probs=F.softmax(pred, dim=-1)
                pred_label=probs.argmax(dim=-1)

                logger.info(f"Batch {i} predictions: {pred_label}, targets: {target[i]}")
                del _
                if i==0:
                    all_pred=pred.cpu()
                    all_target=target[i].cpu()
                else:
                    all_pred=torch.cat([all_pred,pred.cpu()],dim=0)
                    all_target=torch.cat([all_target,target[i].cpu()],dim=0)
            
    def inference(self, input_data, confidence_level):
        self.eval()
        with torch.no_grad():
            pred, _ = self(*input_data)
            probs=F.softmax(pred, dim=-1)
            pred_label=probs.argmax(dim=-1)
            res = {
                "estimated action" : pred_label.squeeze().item()-1,
                "probability"     : probs.max().item()
            }
        return res






# class TSEncoder(nn.Module):
#     """
#     Architecture:
#       1. Patchify each channel independently          (B, C, L) → (B*C, n_patches, d_model)
#       2. Per-channel self-attention (intra-channel)   (B*C, n_patches, d_model) → (B*C, d_model)
#       3. Cross-channel Transformer (inter-channel)    (B, C, d_model) → (B, C, d_model)
#     """

#     def __init__(
#         self,
#         d_model:    int,
#         n_heads:    int,
#         d_ff:       int,
#         dropout:    float,
#         patch_size: int = 32,      # L=6 → 3 patches; tune based on your sequence length
#     ):
#         super().__init__()
#         self.patch_size = patch_size

#         # ── Stage 1: Patch embedding ─────────────────────────────────────
#         # Projects each patch (patch_size raw values) to d_model
#         # Applied identically to every channel (weights shared)
#         self.patch_embed = nn.Linear(patch_size, d_model)
#         self.patch_norm  = nn.LayerNorm(d_model)

#         # ── Stage 2: Per-channel self-attention ──────────────────────────
#         # Operates on patches WITHIN a single channel
#         # Input:  (B*C, n_patches, d_model)
#         # Output: (B*C, n_patches, d_model)  → pooled to (B*C, d_model)
#         # ── Stage 2: Per-channel Transformer over patches ───────────────────
#         intra_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=n_heads,
#             dim_feedforward=d_ff,
#             dropout=dropout,
#             batch_first=True,
#             norm_first=True,
#         )
#         self.intra_encoder = nn.TransformerEncoder(intra_layer, num_layers=1)

#         # ── Stage 3: Cross-channel Transformer ───────────────────────────
#         # Each token = one channel's pooled embedding
#         # Input:  (B, C, d_model)
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=n_heads,
#             dim_feedforward=d_ff,
#             dropout=dropout,
#             batch_first=True,
#             norm_first=True,
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
#         self.norm        = nn.LayerNorm(d_model)
#         self.dropout     = nn.Dropout(dropout)

#     def forward(
#         self,
#         x: torch.Tensor,            # (B, C, L)
#         target_channels: list[int],
#     ) -> torch.Tensor:              # (B, len(target_channels), d_model)

#         B, C, L = x.shape
#         P = self.patch_size
#         n_patches = L // P

#         # ── Stage 1: Patchify ────────────────────────────────────────────
#         # Fold each channel into non-overlapping patches
#         x_p = x.reshape(B * C, 1, L)                    # (B*C, 1, L)
#         x_p = x_p.unfold(-1, P, P)                      # (B*C, 1, n_patches, P)
#         x_p = x_p.squeeze(1)                            # (B*C, n_patches, P)
#         x_p = self.patch_embed(x_p)
#         if torch.isnan(x_p).any():
#             print(f"[TSEncoder] NaN after patch embed  max={x_p.abs().nan_to_num().max():.2f}")
#             raise ValueError("NaN after patch_embed")
#                      # (B*C, n_patches, d_model)
#         x_p = self.patch_norm(x_p)

#         if torch.isnan(x_p).any():
#             print(f"[TSEncoder] NaN after norm  max={x_p.abs().nan_to_num().max():.2f}")
#             raise ValueError("NaN after patch_embed")

#         # ── Stage 2: Per-channel self-attention ──────────────────────────
#         # Patches within each channel attend to each other
#         x_p = self.intra_encoder(x_p)   # (B*C, n_patches, d_model)
#         # x_p = self.intra_norm(x_p + attn_out)           # residual + norm

#         if torch.isnan(x_p).any():
#             print(f"[TSEncoder] NaN after intra-channel attention")
#             raise ValueError("NaN after intra_attn")

#         # Pool patches → single vector per channel
#         x_pooled = x_p.mean(dim=1)                      # (B*C, d_model)
#         x_pooled = x_pooled.reshape(B, C, -1)           # (B, C, d_model)
#         x_pooled = self.dropout(x_pooled)

#         # ── Stage 3: Cross-channel Transformer ───────────────────────────
#         # Detect and mask all-zero channels (missing data)
#         channel_is_zero = (x.abs().sum(dim=-1) == 0)    # (B, C)
#         all_zero = channel_is_zero.all(dim=-1)
#         if all_zero.any():
#             channel_is_zero[all_zero, 0] = False         # keep at least one

#         x_tf = self.transformer(
#             x_pooled,
#             src_key_padding_mask=channel_is_zero,        # True = ignore
#         )                                                # (B, C, d_model)

#         if torch.isnan(x_tf).any():
#             print(f"[TSEncoder] NaN after cross-channel Transformer")
#             raise ValueError("NaN after transformer")

#         # Zero out padding channel outputs
#         x_tf = x_tf * (~channel_is_zero).float().unsqueeze(-1)
#         x_tf = self.norm(x_tf)                          # (B, C, d_model)

#         # ── Extract target channels ──────────────────────────────────────
#         out = x_tf[:, target_channels, :]               # (B, len(target_channels), d_model)
#         return out

# class TSDocFusionLayer(nn.Module):
#     """
#     Two-stage attention fusion:
#       1. Self-attention over document embeddings
#       2. Cross-attention: TS embedding queries the fused document pool
#       3. Linear head → scalar CPI prediction

#     Inputs
#     ------
#     ts_emb  : (B, d_ts)      — Bistro/Moirai CLS token
#     doc_emb : (B, D, d_txt)  — D document CLS tokens (padded)
#     doc_mask : (B, D)        — bool, True = real doc, False = padding

#     Output
#     ------
#     cpi_pred : (B, 1)        — next-month CPI forecast
#     """

#     def __init__(
#         self,
#         d_ts:    int   = 768,   # Bistro CLS dim
#         d_txt:   int   = 768,   # FLANG/FinBERT CLS dim
#         d_model: int   = 512,   # internal dim
#         n_heads: int   = 4,
#         d_ff:    int   = 1024,
#         dropout: float = 0.1,
#     ):
#         super().__init__()

        
        
#         # 1. Self-attention over documents
#         doc_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=n_heads,
#             dim_feedforward=d_ff,
#             dropout=dropout,
#             batch_first=True,
#             norm_first=True,
#         )
#         self.doc_encoder = nn.TransformerEncoder(doc_layer, num_layers=1)

#         # 2. Cross-attention: TS queries documents
#         self.cross_attn = nn.MultiheadAttention(
#             d_model, n_heads, dropout=dropout, batch_first=True
#         )
#         self.norm2 = nn.LayerNorm(d_model)

#         # 3. FFN on fused TS embedding
#         self.ffn = nn.Sequential(
#             nn.Linear(d_model, d_ff),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_ff, d_model),
#         )
#         self.norm3 = nn.LayerNorm(d_model)




#     def forward(self, z_ts, z_doc, doc_mask=None):
#         B = z_ts.shape[0]
#         pad_mask = ~doc_mask if doc_mask is not None else None  # (B, D) True=ignore

#         # 1. Self-attention over documents
#         z_doc = self.doc_encoder(
#             z_doc,
#             src_key_padding_mask=pad_mask,    # (B, D) True = ignore [web:623]
#         )        

        

#         # NaN guard: ensure no sample is fully masked
#         if pad_mask is not None:
#             all_masked = pad_mask.all(dim=-1)
#             if all_masked.any():
#                 pad_mask[all_masked, 0] = False

#         # 1. Self-attention over documents
#         z_doc = self.doc_encoder(
#             z_doc,
#             src_key_padding_mask=pad_mask,    # (B, D) True = ignore [web:623]
#         )

        
#         # Zero out padding positions for safety
#         if doc_mask is not None:
#             z_doc = z_doc * doc_mask.float().unsqueeze(-1)

#         # 2. Cross-attention: documents query TS
#         q = z_ts.unsqueeze(1)                            # (B, 1, d_model)
#         cross_out, attn_weights = self.cross_attn(
#             q, z_doc, z_doc,
#             key_padding_mask=pad_mask,    # (B, D) True = ignore
#         )                                                
        
#         # # Pool over real documents only
#         # if doc_mask is not None:
#         #     mask_f  = doc_mask.float().unsqueeze(-1)     # (B, D, 1)
#         #     z_fused = (cross_out * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
#         # else:
#         #     z_fused = cross_out.mean(dim=1)

        
#         z_fused = self.norm2(cross_out+q).squeeze(1)            # (B, d_model)
        
#         # 3. FFN
#         z_fused = self.norm3(z_fused + self.ffn(z_fused))
                
#         return  z_fused


    


class SignalingModelV3(nn.Module):
    def __init__(self,kwargs ):
        super().__init__()
        self.config = kwargs
        self.fusion = nn.ModuleList([TSDocFusionLayer(kwargs) for _ in range(kwargs.get("n_layers", 4))])
        self.head = nn.Linear(kwargs.get("d_model", 512), 3)
        #time series encoder
        self.ts_encoder = TSEncoder(kwargs.get("d_ts", 768), kwargs.get("n_heads", 4), kwargs.get("d_ff", 1024), kwargs.get("dropout", 0.1))
        # --- project both modalities to shared d_model ---
        self.ts_proj  = nn.Linear(kwargs.get("d_ts", 768), kwargs.get("d_model", 512))
        self.txt_proj = nn.Linear(kwargs.get("d_txt", 768), kwargs.get("d_model", 512))


    def forward(self, ts, doc_emb, doc_mask=None):
        
        z_ts = self.ts_encoder(ts, [0]).squeeze(1)     # (B, d_ts)
        z_ts = self.ts_proj(z_ts)                      # (B, d_model)
        z_doc = self.txt_proj(doc_emb)
        
        z_fused=self.fusion[0](z_ts, z_doc, doc_mask)                # (B, D, d_model)
        for layer in self.fusion[1:]:
            z_fused = layer(z_fused, z_doc, doc_mask)

        cpi_pred = self.head(z_fused)                  # (B, 3)
        return cpi_pred, z_fused
    
    def fit(self, dataloader,label_percentages=None):
        optimizer=torch.optim.Adam(self.parameters(),lr=self.config.lr, weight_decay=1e-4) 
        self.train()
        criterion = torch.nn.CrossEntropyLoss(weight=label_percentages) if label_percentages is not None else torch.nn.CrossEntropyLoss()
        for epoch in range(self.config.epoch):
            l=0
            for ts,ctx,mask, target in dataloader:
                optimizer.zero_grad(set_to_none=True)
                pred, _ = self(ts, ctx, mask)
                del _
                loss=criterion(pred, target.squeeze(-1).long())
                l+=loss.item()
                loss.backward()
                optimizer.step()
                grad_norm = torch.stack([p.grad.norm() for p in self.parameters() if p.grad is not None]).norm()
                logger.info(f"Batch loss: {loss.item()}")
                logger.info(f"Gradient norm: {grad_norm}")

            logger.info(f"loss epoch{epoch} / {self.config.epoch} : {l/len(dataloader)}")
        self.eval()


    def test(self,test_dataset):
        ts,ctx,mask, target = test_dataset
        batch_size = self.config.batch_size
        self.eval()
        with torch.no_grad():
            ts,ctx,mask,target = ts.chunk(batch_size), ctx.chunk(batch_size), mask.chunk(batch_size), target.chunk(batch_size)
            for i in range(len(ts)):
                pred, _ = self(ts[i], ctx[i], mask[i])
                logger.info(f"Predictions: {pred}")

                probs=F.softmax(pred, dim=-1)
                pred_label=probs.argmax(dim=-1)

                logger.info(f"Batch {i} predictions: {pred_label}, targets: {target[i]}")
                del _
                if i==0:
                    all_pred=pred.cpu()
                    all_target=target[i].cpu()
                else:
                    all_pred=torch.cat([all_pred,pred.cpu()],dim=0)
                    all_target=torch.cat([all_target,target[i].cpu()],dim=0)
            
    def inference(self, input_data, confidence_level):
        self.eval()
        with torch.no_grad():
            pred, _ = self(*input_data)
            probs=F.softmax(pred, dim=-1)
            pred_label=probs.argmax(dim=-1)
            res = {
                "estimated action" : pred_label.squeeze().item()-1,
                "probability"     : probs.max().item()
            }
        return res

    
# class SignalingModelV3(nn.Module):
#     """
#     Two-stage attention fusion:
#       1. Self-attention over document embeddings
#       2. Cross-attention: TS embedding queries the fused document pool
#       3. Linear head → scalar CPI prediction

#     Inputs
#     ------
#     ts_emb  : (B, d_ts)      — Bistro/Moirai CLS token
#     doc_emb : (B, D, d_txt)  — D document CLS tokens (padded)
#     doc_mask : (B, D)        — bool, True = real doc, False = padding

#     Output
#     ------
#     cpi_pred : (B, 1)        — next-month CPI forecast
#     """

#     def __init__(
#         self,
#         config
#     ):
#         super().__init__()
#         self.config = config
#         d_ts:    int   = self.config.d_model[0]   # Bistro CLS dim
#         d_txt:   int   = self.config.d_model[1]   # FLANG/FinBERT CLS dim
#         d_model: int   = 256   # internal dim
#         n_heads: int   = self.config.n_heads
#         d_ff:    int   = self.config.d_ff
#         dropout: float = self.config.dropout
        

#         #time series encoder
#         self.ts_encoder = TSEncoder(d_ts, n_heads, d_ff, dropout)
#         # --- project both modalities to shared d_model ---
#         self.ts_proj  = nn.Linear(d_ts,  d_model)
#         self.txt_proj = nn.Linear(d_txt, d_model)

#         # 1. Self-attention over documents
#         doc_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=n_heads,
#             dim_feedforward=d_ff,
#             dropout=dropout,
#             batch_first=True,
#             norm_first=True,
#         )
#         self.doc_encoder = nn.TransformerEncoder(doc_layer, num_layers=1)

#         # 2. Cross-attention: TS queries documents
#         self.cross_attn = nn.MultiheadAttention(
#             d_model, n_heads, dropout=dropout, batch_first=True
#         )
#         self.norm2 = nn.LayerNorm(d_model)

#         # 3. FFN on fused TS embedding
#         self.ffn = nn.Sequential(
#             nn.Linear(d_model, d_ff),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_ff, d_model),
#         )
#         self.norm3 = nn.LayerNorm(d_model)

#         # 4. Prediction head
#         self.head = nn.Linear(d_model, 3)

#         self._init_weights()

    

#     def forward(
#         self,
#         ts:   Tensor,              # (B, channels, d_ts)
#         doc_emb:  Tensor,              # (B, D, d_txt)
#         doc_mask: Tensor | None = None,  # (B, D) bool, True=real
#     ) -> tuple[Tensor, Tensor]:
#         """
#         Returns
#         -------
#         cpi_pred   : (B, 1)       — scalar CPI forecast
#         fused_emb  : (B, d_model) — fused TS embedding (reusable)
#         """
#         B = ts.shape[0]

#         ts_emb = self.ts_encoder(ts, [0]).squeeze(1)     # (B, d_ts)
#         if torch.isnan(ts_emb).any():
#             print("NaN in TS embedding!")
#             print(f"  ts_emb={ts_emb.detach().cpu().numpy()}")
#             raise ValueError("Debug stop")  # Remove this after debugging


#         z_ts   = self.ts_proj(ts_emb)                    # (B, d_model)
#         if torch.isnan(z_ts).any():
#             print("NaN in projected TS embedding!")
#             print(f"  z_ts={z_ts.detach().cpu().numpy()}")
#             raise ValueError("Debug stop")  # Remove this after debugging


#         z_doc  = self.txt_proj(doc_emb)                  # (B, D, d_model)
#         if torch.isnan(z_doc).any():
#             print("NaN in projected doc embeddings!")
#             print(f"  z_doc={z_doc.detach().cpu().numpy()}")
#             raise ValueError("Debug stop")  # Remove this after debugging


#         pad_mask = ~doc_mask if doc_mask is not None else None  # (B, D) True=ignore

#         # NaN guard: ensure no sample is fully masked
#         if pad_mask is not None:
#             all_masked = pad_mask.all(dim=-1)
#             if all_masked.any():
#                 pad_mask[all_masked, 0] = False

#         # 1. Self-attention over documents
#         z_doc = self.doc_encoder(
#             z_doc,
#             src_key_padding_mask=pad_mask,    # (B, D) True = ignore [web:623]
#         )

#         if torch.isnan(z_doc).any():
#             print("NaN after doc TransformerEncoder!")
#             raise ValueError("NaN after doc_encoder")

#         # Zero out padding positions for safety
#         if doc_mask is not None:
#             z_doc = z_doc * doc_mask.float().unsqueeze(-1)

#         # 2. Cross-attention: documents query TS
#         q = z_ts.unsqueeze(1)                            # (B, 1, d_model)
#         cross_out, attn_weights = self.cross_attn(
#             z_doc, q, q                                  # Q=docs, K=TS, V=TS
#         )                                                # (B, D, d_model)
#         if torch.isnan(cross_out).any():
#             print("NaN in cross-attention output!")
#             print(f"  cross_out={cross_out.detach().cpu().numpy()}")
#             raise ValueError("Debug stop")  # Remove this after debugging


#         # Pool over real documents only
#         if doc_mask is not None:
#             mask_f  = doc_mask.float().unsqueeze(-1)     # (B, D, 1)
#             z_fused = (cross_out * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
#         else:
#             z_fused = cross_out.mean(dim=1)

#         if torch.isnan(z_fused).any():
#             print("NaN in fused embedding after cross-attention!")
#             print(f"  z_fused={z_fused.detach().cpu().numpy()}")
#             raise ValueError("Debug stop")  # Remove this after debugging

#         z_fused = self.norm2(z_fused+z_ts)              # (B, d_model)
#         if torch.isnan(z_fused).any():
#             print("NaN after adding TS embedding and norm!")
#             print(f"  z_fused={z_fused.detach().cpu().numpy()}")
#             raise ValueError("Debug stop")  # Remove this after debugging


#         # 3. FFN
#         z_fused = self.norm3(z_fused + self.ffn(z_fused))
#         if torch.isnan(z_fused).any():
#             print("NaN in fused embedding after FFN!")
#             print(f"  z_fused={z_fused.detach().cpu().numpy()}")
#             raise ValueError("Debug stop")  # Remove this after debugging


#         # 4. Predict
#         out = self.head(z_fused)                    # (B, 1)
#         if torch.isnan(out).any():
#             print("NaN in output!")
#             print(f"  out={out.detach().cpu().numpy()}")
#             raise ValueError("Debug stop")  # Remove this after debugging
        

#         return out, z_fused

    
#     def fit(self, dataloader,label_percentages=None):
#         optimizer=torch.optim.Adam(self.parameters(),lr=self.config.lr, weight_decay=1e-4) 
#         self.train()
#         criterion = torch.nn.CrossEntropyLoss(weight=label_percentages) if label_percentages is not None else torch.nn.CrossEntropyLoss()
#         for epoch in range(self.config.epoch):
#             l=0
#             for ts,ctx,mask, target in dataloader:
#                 optimizer.zero_grad(set_to_none=True)
#                 pred, _ = self(ts, ctx, mask)
#                 del _
#                 loss=criterion(pred, target.squeeze(-1).long())
#                 l+=loss.item()
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)  # gradient clipping
#                 optimizer.step()
#                 logger.debug(f"Batch loss: {loss.item()}")

#             logger.info(f"loss epoch{epoch} / {self.config.epoch} : {l/len(dataloader)}")
#         self.eval()

#     def _init_weights(self):
#         for module in [self.ts_proj, self.txt_proj, self.ffn, self.head]:          # ← add this
#             for p in module.parameters():
#                 if p.dim() > 1:
#                     nn.init.xavier_uniform_(p)
#                 else:
#                     nn.init.zeros_(p)

#     def test(self,test_dataset):
#         ts,ctx,mask, target = test_dataset
#         batch_size = self.config.batch_size
#         self.eval()
#         with torch.no_grad():
#             ts,ctx,mask,target = ts.chunk(batch_size), ctx.chunk(batch_size), mask.chunk(batch_size), target.chunk(batch_size)
#             for i in range(len(ts)):
#                 pred, _ = self(ts[i], ctx[i], mask[i])
#                 logger.info(f"Predictions: {pred}")

#                 probs=F.softmax(pred, dim=-1)
#                 pred_label=probs.argmax(dim=-1)

#                 logger.info(f"Batch {i} predictions: {pred_label}, targets: {target[i]}")
#                 del _
#                 if i==0:
#                     all_pred=pred.cpu()
#                     all_target=target[i].cpu()
#                 else:
#                     all_pred=torch.cat([all_pred,pred.cpu()],dim=0)
#                     all_target=torch.cat([all_target,target[i].cpu()],dim=0)
            
            
#     def _finalize_plot(self, name):
#         if threading.current_thread() is threading.main_thread():
#             plt.show()
#             return

#         plot_dir = os.getenv("SIGNALING_PLOT_DIR", "logs/signaling_plots")
#         os.makedirs(plot_dir, exist_ok=True)
#         filename = os.path.join(plot_dir, f"{name}_{os.getpid()}.png")
#         plt.savefig(filename, dpi=150, bbox_inches="tight")
#         plt.close()


#     def inference(self, input_data, confidence_level):
#         self.eval()
#         with torch.no_grad():
#             pred, _ = self(*input_data)
#             probs=F.softmax(pred, dim=-1)
#             pred_label=probs.argmax(dim=-1)
#             res = {
#                 "estimated action" : pred_label.squeeze().item()-1,
#                 "probability"     : probs.max().item()
#             }
#         return res

class SignalingModelV1(nn.Module):

    def __init__(self, config):
        super(SignalingModelV1,self).__init__()
        self.config = config
        d_model=self.config.d_model
        self.layer_norm1 = nn.LayerNorm(d_model//2)  # For o1+o3 
        self.layer_norm2 = nn.LayerNorm(d_model//4)  # For o5+o2
        self.layer1=nn.Linear(d_model,d_model//2)
        self.layer2=nn.Linear(d_model//2,d_model//4)
        self.layer3=nn.Linear(d_model//4,d_model//2)
        self.layer4=nn.Linear(d_model//2,d_model//8)
        self.layer5=nn.Linear(d_model//8,d_model//4)
        self.out = nn.Linear(d_model//4,2)


        
    def forward(self,x):
        B,D=x.shape
        activation = self.config.activation
        o1 = activation(self.layer1(x))
        o2 = activation(self.layer2(o1))
        o3 = self.layer3(o2)
        o3 = self.layer_norm1(o1+o3)  # Use layer_norm1 for d_model//2
        o4 = activation(self.layer4(o3))
        o5 = self.layer5(o4)
        o5 = self.layer_norm2(o5+o2)  # Use layer_norm2 for d_model//4
        out = self.out(o5)

        mu= out[:,0].unsqueeze(-1)
        logvar=out[:,1].unsqueeze(-1)
        
        pred= logvar.exp()*torch.randn((B,1),requires_grad=False) 
        return pred, mu, logvar

    def reconstruction_loss(self,pred,Y):
        return F.mse_loss(pred,Y)
    
    def fit(self, dataloader):
        optimizer=torch.optim.Adam(self.parameters(),lr=self.config.lr, weight_decay=1e-4) 
        
        for epoch in range(self.config.epoch):
            l=0
            for X, Y in dataloader:
                optimizer.zero_grad()
                pred, _,_ = self(X)
                loss=self.reconstruction_loss(pred,Y)
                l+=loss.item()
                loss.backward()
                optimizer.step()

            logger.info(f"loss epoch{epoch} / {self.config.epoch} : {l/len(dataloader)}")

    def test(self,test_dataset):
        self.eval()
        with torch.no_grad():
            X, Y = test_dataset.tensors
            pred, _,_ = self(X)
            loss=self.reconstruction_loss(pred,Y)
        logger.info(f"Test loss: {loss.item()}")
        plt.plot(pred.cpu().numpy(), label="Predicted")
        plt.plot(Y.cpu().numpy(), label="Actual")
        plt.legend()
        plt.title("Predicted vs Actual")
        self._finalize_plot("pred_vs_actual")
        

    def inference(self, input_data, confidence_level):
        with torch.no_grad():
            pred, mu, logvar = self(input_data)
            dist = torch.distributions.Normal(mu, torch.exp(0.5 * logvar))
            res = {
                "estimated price"  : round(pred.item(),2),
                "prob"             : round(dist.log_prob(pred).exp().item(),2),  # PDF at pred
                "mu"               : round(mu.item(),2),
                "logvar"           : round(logvar.item(),2),
                "estimated action" : t_test(mu, logvar, confidence_level).item()
            }
        return res
    

        
def t_test(mu: torch.Tensor, logvar: torch.Tensor, confidence_level: float) -> torch.Tensor:
    """
    Performs t test of a gaussian distribution and the value 0.
    If the distribution is confidently positive, returns 1.
    If it is confidently negative, returns -1. Otherwise, zero.
    """
    std = torch.exp(0.5 * logvar)                          # σ from log-variance

    alpha = 1.0 - confidence_level
    # z critical value via torch — equivalent to scipy.stats.norm.ppf(1 - alpha/2)
    standard_normal = Normal(
        torch.zeros_like(mu),
        torch.ones_like(mu)
    )
    z = standard_normal.icdf(torch.tensor(1 - alpha / 2, device=mu.device, dtype=mu.dtype))

    lower = mu - z * std
    upper = mu + z * std

    result = torch.where(lower > 0,  torch.ones_like(mu),
             torch.where(upper < 0, -torch.ones_like(mu),
                                     torch.zeros_like(mu)))
    return result


def test_signaling_model():

    config = SignalingConfig()
    config.epoch = 1
    model = SignalingModelV1(config)

    # Dummy data
    X = torch.randn(100, config.d_model)
    Y = torch.randn(100, 1)

    dataset = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    model.fit(dataloader)

    input_data = torch.randn(1, config.d_model)
    confidence_level = 0.95
    result = model.inference(input_data, confidence_level)
    logger.info("Inference result: %s", result)



class SignalingModelV2(nn.Module):

    def __init__(self, config):
        super(SignalingModelV2, self).__init__()
        self.config = config
        self.sequential = nn.Sequential(
            nn.Linear(config.d_model, 1)
        )
    
    
    def reconstruction_loss(self,pred,Y):
        return F.mse_loss(pred,Y)
    

    def fit(self, dataloader):
        optimizer=torch.optim.Adam(self.parameters(),lr=self.config.lr) 
        
        for epoch in range(self.config.epoch):
            l=0
            for X, Y in dataloader:
                optimizer.zero_grad()
                pred= self(X)
                loss=self.reconstruction_loss(pred,Y)
                l+=loss.item()
                loss.backward()
                optimizer.step()

            logger.info(f"loss epoch{epoch} / {self.config.epoch} : {l/len(dataloader)}")

    def forward(self, input_data):
        return self.sequential(input_data)
    
    def test(self,test_dataset):
        self.eval()
        with torch.no_grad():
            X, Y = test_dataset.tensors
            pred= self(X)
            loss=self.reconstruction_loss(pred,Y)

        logger.info(f"Test loss: {loss.item()}")
        # still on CPU here (use .cpu() earlier if needed)
        pred_cpu = pred.detach().cpu()
        y_cpu = Y.detach().cpu()

        idx = torch.arange(pred_cpu.shape[0])

        plt.scatter(idx.numpy(), pred_cpu.numpy(), label="Predicted")
        plt.scatter(idx.numpy(), y_cpu.numpy(), label="Actual")
        plt.legend()
        plt.title("Predicted vs Actual over index")
        self._finalize_plot("pred_vs_actual_index")

    def _finalize_plot(self, name):
        if threading.current_thread() is threading.main_thread():
            plt.show()
            return

        plot_dir = os.getenv("SIGNALING_PLOT_DIR", "logs/signaling_plots")
        os.makedirs(plot_dir, exist_ok=True)
        filename = os.path.join(plot_dir, f"{name}_{os.getpid()}.png")
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()

    def inference(self, input_data, confidence_level):
        with torch.no_grad():
            pred = self(input_data)
            
            res = {
                "estimated price"  : round(pred.item(),2),
                "estimated action" : t_test(pred, torch.ones_like(pred), confidence_level).item()
            }
        return res


if __name__ == "__main__":

    from .config import SignalingConfig

    
    X_ts = torch.randn(10, 5, 1000)       # (B, C, L)
    X_doc = torch.randn(10, 10, 768)      # (B
    doc_mask = torch.zeros(10, 10, dtype=torch.bool)  # (B, D) all real
    target = torch.randint(0, 3, (10, 1))  # (B, 1) random class labels
    train_loader= torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_ts, X_doc, doc_mask, target), batch_size=2)


    kwargs = SignalingConfig()

    
    model = SignalingModelV4(kwargs)

    model.fit(train_loader)