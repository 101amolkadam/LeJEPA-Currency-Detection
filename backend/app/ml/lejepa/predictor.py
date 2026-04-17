"""Narrow ViT predictor for LeJEPA latent prediction."""

from __future__ import annotations

import torch
import torch.nn as nn


class PredictorBlock(nn.Module):
    """Lightweight transformer block for the predictor."""

    def __init__(self, dim: int, num_heads: int = 3, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        head_dim = dim // num_heads
        self.num_heads = num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        B, N, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, D // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        h = (attn @ v).transpose(1, 2).reshape(B, N, D)
        h = self.proj(h)
        x = x + h

        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class ViTPredictor(nn.Module):
    """
    Narrow ViT predictor for LeJEPA.

    Takes context embeddings + positional tokens for target positions,
    and predicts target embeddings in the encoder's representation space.

    Config (ViT-Tiny predictor):
        embed_dim   = 96  (half of encoder)
        depth       = 6
        num_heads   = 3
    """

    def __init__(
        self,
        encoder_embed_dim: int = 192,
        predictor_embed_dim: int = 96,
        depth: int = 6,
        num_heads: int = 3,
        num_patches: int = 196,         # (224/16)^2 = 196
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.encoder_embed_dim = encoder_embed_dim
        self.predictor_embed_dim = predictor_embed_dim

        # Project encoder embeddings → predictor space
        self.input_proj = nn.Linear(encoder_embed_dim, predictor_embed_dim)

        # Learnable position embeddings for all patch positions
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, predictor_embed_dim)
        )

        # Learnable mask token for target positions
        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, predictor_embed_dim)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            PredictorBlock(predictor_embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(predictor_embed_dim)

        # Project back to encoder space for loss computation
        self.output_proj = nn.Linear(predictor_embed_dim, encoder_embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
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

    def forward(
        self,
        context_embeddings: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            context_embeddings: (B, Nc, D_enc) — encoder output for context patches
            context_indices:    (B, Nc) — patch indices of context
            target_indices:     (B, Nt) — patch indices of targets to predict

        Returns:
            predicted: (B, Nt, D_enc) — predicted target embeddings
        """
        B = context_embeddings.shape[0]
        Nc = context_indices.shape[1]
        Nt = target_indices.shape[1]

        # Project context to predictor space
        ctx = self.input_proj(context_embeddings)                   # (B, Nc, D_pred)

        # Add positional embeddings for context positions
        ctx_pos = self.pos_embed[:, :, :].expand(B, -1, -1)        # (B, N_total, D_pred)
        ctx_pos_selected = torch.gather(
            ctx_pos, 1,
            context_indices.unsqueeze(-1).expand(-1, -1, self.predictor_embed_dim)
        )
        ctx = ctx + ctx_pos_selected

        # Create mask tokens for target positions
        mask_tokens = self.mask_token.expand(B, Nt, -1)             # (B, Nt, D_pred)
        tgt_pos_selected = torch.gather(
            ctx_pos, 1,
            target_indices.unsqueeze(-1).expand(-1, -1, self.predictor_embed_dim)
        )
        mask_tokens = mask_tokens + tgt_pos_selected

        # Concatenate context + target tokens
        x = torch.cat([ctx, mask_tokens], dim=1)                    # (B, Nc+Nt, D_pred)

        # Transformer
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # Extract only the target predictions
        predicted = x[:, Nc:, :]                                    # (B, Nt, D_pred)

        # Project back to encoder embedding space
        predicted = self.output_proj(predicted)                     # (B, Nt, D_enc)
        return predicted
