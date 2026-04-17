"""Vision Transformer (ViT-Tiny) encoder for LeJEPA."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """Convert image to sequence of patch embeddings."""

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, embed_dim: int = 192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, N, D)
        x = self.proj(x)                       # (B, D, H', W')
        x = x.flatten(2).transpose(1, 2)       # (B, N, D)
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, dim: int, num_heads: int = 3, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)      # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    """Feed-forward network with GELU."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, dim: int, num_heads: int = 3, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """
    Vision Transformer encoder (ViT-Tiny configuration).

    - Patch size: 16×16
    - Embed dim: 192
    - Depth: 12 blocks
    - Heads: 3
    - No CLS token (JEPA uses per-patch embeddings)
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        # Learnable position embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        # Initialize position embeddings with truncated normal
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_module)

    @staticmethod
    def _init_module(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def interpolate_pos_embed(self, num_patches: int) -> torch.Tensor:
        """Interpolate position embeddings if input resolution differs."""
        if num_patches == self.num_patches:
            return self.pos_embed

        N = self.pos_embed.shape[1]
        dim = self.pos_embed.shape[2]
        sqrt_N = int(math.sqrt(N))
        sqrt_new = int(math.sqrt(num_patches))

        pos = self.pos_embed.reshape(1, sqrt_N, sqrt_N, dim).permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=(sqrt_new, sqrt_new), mode="bilinear", align_corners=False)
        pos = pos.permute(0, 2, 3, 1).reshape(1, num_patches, dim)
        return pos

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor
            mask: optional boolean mask (B, N) — True = keep, False = drop
        Returns:
            (B, N', D) patch embeddings (N' = kept patches if masked)
        """
        x = self.patch_embed(x)                 # (B, N, D)
        x = x + self.interpolate_pos_embed(x.shape[1])

        if mask is not None:
            # Apply mask: keep only True positions
            B, N, D = x.shape
            # mask shape: (B, N) boolean
            x = x[mask].reshape(B, -1, D)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward without masking — for classification."""
        return self.forward(x, mask=None)
