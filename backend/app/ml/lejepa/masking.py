"""Multi-block masking strategy for LeJEPA pretraining."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch


@dataclass
class MaskResult:
    """Result of the masking procedure."""
    context_mask: torch.Tensor           # (N,) boolean — True = context patch
    target_indices: list[torch.Tensor]   # list of (Nt_i,) index tensors
    context_indices: torch.Tensor        # (Nc,) index tensor


def sample_block_mask(
    num_patches_h: int,
    num_patches_w: int,
    min_scale: float = 0.15,
    max_scale: float = 0.2,
    min_aspect: float = 0.75,
    max_aspect: float = 1.5,
) -> torch.Tensor:
    """
    Sample a single rectangular block mask on the patch grid.

    Returns:
        (N,) boolean tensor where True indicates a selected (target) patch.
    """
    N = num_patches_h * num_patches_w

    # Random block area
    area = random.uniform(min_scale, max_scale) * N
    aspect = math.exp(random.uniform(math.log(min_aspect), math.log(max_aspect)))

    h = int(round(math.sqrt(area * aspect)))
    w = int(round(math.sqrt(area / aspect)))
    h = min(h, num_patches_h)
    w = min(w, num_patches_w)

    # Random top-left corner
    top = random.randint(0, num_patches_h - h)
    left = random.randint(0, num_patches_w - w)

    mask = torch.zeros(num_patches_h, num_patches_w, dtype=torch.bool)
    mask[top: top + h, left: left + w] = True
    return mask.flatten()


def generate_masks(
    num_patches_h: int = 14,
    num_patches_w: int = 14,
    num_targets: int = 4,
    min_context_ratio: float = 0.5,
    target_min_scale: float = 0.15,
    target_max_scale: float = 0.2,
) -> MaskResult:
    """
    Generate context + target masks for LeJEPA pretraining.

    Strategy:
        1. Sample `num_targets` non-overlapping target blocks
        2. Context = all patches NOT in any target block
        3. Ensure context has at least `min_context_ratio` of all patches

    Args:
        num_patches_h: grid height
        num_patches_w: grid width
        num_targets: number of target blocks
        min_context_ratio: minimum fraction of patches to keep as context
        target_min_scale: min fraction of total patches per target block
        target_max_scale: max fraction of total patches per target block

    Returns:
        MaskResult with context/target index tensors
    """
    N = num_patches_h * num_patches_w
    all_targets_mask = torch.zeros(N, dtype=torch.bool)
    target_masks: list[torch.Tensor] = []

    for _ in range(num_targets):
        # Try to find a non-overlapping block
        for _attempt in range(50):
            block = sample_block_mask(
                num_patches_h, num_patches_w,
                min_scale=target_min_scale,
                max_scale=target_max_scale,
            )
            # Check overlap
            overlap = (block & all_targets_mask).sum().item()
            if overlap == 0:
                break
        else:
            # Fallback: accept some overlap
            block = sample_block_mask(
                num_patches_h, num_patches_w,
                min_scale=target_min_scale,
                max_scale=target_max_scale,
            )

        target_masks.append(block)
        all_targets_mask = all_targets_mask | block

    # Ensure minimum context ratio
    if all_targets_mask.float().mean() > (1.0 - min_context_ratio):
        # Too many targets — trim the last ones
        while all_targets_mask.float().mean() > (1.0 - min_context_ratio) and target_masks:
            removed = target_masks.pop()
            all_targets_mask = all_targets_mask & ~removed

    # Context = everything NOT in targets
    context_mask = ~all_targets_mask                             # (N,) bool

    # Convert to index tensors
    context_indices = context_mask.nonzero(as_tuple=True)[0]     # (Nc,)
    target_index_tensors = [
        m.nonzero(as_tuple=True)[0] for m in target_masks
    ]

    return MaskResult(
        context_mask=context_mask,
        target_indices=target_index_tensors,
        context_indices=context_indices,
    )
