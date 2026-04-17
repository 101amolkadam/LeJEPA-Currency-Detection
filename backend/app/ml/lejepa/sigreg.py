"""SIGReg — Sketched Isotropic Gaussian Regularization.

The core innovation of LeJEPA: enforces isotropic Gaussian distribution
on latent embeddings to prevent representation collapse without heuristics.

Reference: "LeJEPA: Provable and Scalable Self-Supervised Learning Without
the Heuristics" — Balestriero & LeCun, 2025 (arXiv:2511.08544)
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def sigreg_loss(
    embeddings: torch.Tensor,
    num_projections: int = 128,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute the SIGReg loss on a batch of embeddings.

    Uses random projections to test if the embedding distribution
    matches an isotropic standard Gaussian via sorted-quantile matching.

    Args:
        embeddings: (B, D) batch of embeddings
        num_projections: number of random 1D projections
        eps: numerical stability constant

    Returns:
        Scalar loss encouraging isotropic Gaussian embedding distribution.
    """
    B, D = embeddings.shape
    if B < 4:
        return torch.tensor(0.0, device=embeddings.device)

    # ── 1. Center and standardize per-dimension ──────────────
    mean = embeddings.mean(dim=0, keepdim=True)
    std = embeddings.std(dim=0, keepdim=True).clamp(min=eps)
    z = (embeddings - mean) / std                               # (B, D)

    # ── 2. Random projection directions ──────────────────────
    proj = torch.randn(D, num_projections, device=embeddings.device)
    proj = F.normalize(proj, dim=0)                             # unit vectors

    # ── 3. Project embeddings to 1-D slices ──────────────────
    projected = z @ proj                                        # (B, num_proj)

    # ── 4. Standardize each projection ───────────────────────
    p_mean = projected.mean(dim=0, keepdim=True)
    p_std = projected.std(dim=0, keepdim=True).clamp(min=eps)
    projected = (projected - p_mean) / p_std

    # ── 5. Sort for QQ-comparison ────────────────────────────
    projected_sorted, _ = projected.sort(dim=0)                 # (B, num_proj)

    # ── 6. Expected standard Gaussian quantiles ──────────────
    # Uniform quantiles in (0, 1), mapped through inverse error function
    quantiles = torch.linspace(
        0.5 / B, 1.0 - 0.5 / B, B,
        device=embeddings.device, dtype=embeddings.dtype,
    )
    expected = torch.erfinv(2.0 * quantiles - 1.0) * math.sqrt(2.0)  # (B,)

    # ── 7. L2 distance between sorted projections and quantiles
    loss = ((projected_sorted - expected.unsqueeze(1)) ** 2).mean()

    return loss


def variance_loss(embeddings: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Auxiliary variance regularization — ensures per-dimension variance stays
    close to 1 (complementary to SIGReg for extra stability on small batches).
    """
    var = embeddings.var(dim=0)
    return F.relu(1.0 - var.clamp(min=eps).sqrt()).mean()


def covariance_loss(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Auxiliary covariance regularization — drives off-diagonal covariance
    entries toward zero (decorrelation).
    """
    B, D = embeddings.shape
    z = embeddings - embeddings.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / (B - 1)                                  # (D, D)
    off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    return off_diag / D
