"""Full LeJEPA model assembly — pretraining + classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.ml.lejepa.encoder import ViTEncoder
from app.ml.lejepa.predictor import ViTPredictor
from app.ml.lejepa.sigreg import sigreg_loss, variance_loss, covariance_loss
from app.ml.lejepa.masking import generate_masks, MaskResult


class LeJEPAPretrainModel(nn.Module):
    """
    LeJEPA pretraining model.

    Components:
        - Context encoder (ViT-Tiny): encodes visible context patches
        - Target encoder  (ViT-Tiny): encodes target patches (shared weights)
        - Predictor (narrow ViT):     predicts target embeddings from context

    Loss = L2 prediction loss + λ * SIGReg
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 192,
        encoder_depth: int = 12,
        encoder_heads: int = 3,
        predictor_embed_dim: int = 96,
        predictor_depth: int = 6,
        predictor_heads: int = 3,
        sigreg_lambda: float = 1.0,
    ):
        super().__init__()
        self.sigreg_lambda = sigreg_lambda
        num_patches = (img_size // patch_size) ** 2
        self.num_patches_h = img_size // patch_size
        self.num_patches_w = img_size // patch_size

        # Context encoder (trained)
        self.encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
        )

        # Target encoder (shared weights with context encoder — no stop-gradient in LeJEPA)
        # In LeJEPA, both encoders share the SAME weights; SIGReg prevents collapse
        self.target_encoder = self.encoder  # shared reference

        # Predictor
        self.predictor = ViTPredictor(
            encoder_embed_dim=embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            depth=predictor_depth,
            num_heads=predictor_heads,
            num_patches=num_patches,
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass for pretraining.

        Args:
            images: (B, 3, H, W) input images

        Returns:
            dict with 'loss', 'pred_loss', 'sigreg_loss'
        """
        B = images.shape[0]
        device = images.device

        # Generate masks
        mask_result = generate_masks(
            self.num_patches_h, self.num_patches_w,
            num_targets=4,
        )

        # ── Target encoder: encode ALL patches, then select targets ──
        with torch.no_grad():
            # We compute gradients through shared weights via the context path
            all_target_embeddings = self.target_encoder.forward_features(images)  # (B, N, D)

        # ── Context encoder: encode only context patches ──
        all_context_embeddings = self.encoder.forward_features(images)            # (B, N, D)

        # Select context embeddings
        ctx_idx = mask_result.context_indices.to(device)                          # (Nc,)
        ctx_idx_expanded = ctx_idx.unsqueeze(0).unsqueeze(-1).expand(
            B, -1, self.encoder.embed_dim
        )
        context_emb = torch.gather(all_context_embeddings, 1, ctx_idx_expanded)   # (B, Nc, D)

        # ── Predict targets and compute loss ──
        total_pred_loss = torch.tensor(0.0, device=device)
        num_target_blocks = 0

        for tgt_indices in mask_result.target_indices:
            if len(tgt_indices) == 0:
                continue
            tgt_idx = tgt_indices.to(device)                                      # (Nt,)

            # Batch-expand indices
            ctx_batch = ctx_idx.unsqueeze(0).expand(B, -1)                        # (B, Nc)
            tgt_batch = tgt_idx.unsqueeze(0).expand(B, -1)                        # (B, Nt)

            # Predict target embeddings
            predicted = self.predictor(context_emb, ctx_batch, tgt_batch)          # (B, Nt, D)

            # Ground-truth target embeddings
            tgt_idx_expanded = tgt_idx.unsqueeze(0).unsqueeze(-1).expand(
                B, -1, self.encoder.embed_dim
            )
            target_emb = torch.gather(all_target_embeddings, 1, tgt_idx_expanded)  # (B, Nt, D)

            # L2 prediction loss (normalized)
            pred_norm = F.normalize(predicted, dim=-1)
            tgt_norm = F.normalize(target_emb.detach(), dim=-1)
            total_pred_loss = total_pred_loss + ((pred_norm - tgt_norm) ** 2).mean()
            num_target_blocks += 1

        if num_target_blocks > 0:
            total_pred_loss = total_pred_loss / num_target_blocks

        # ── SIGReg on encoder output ──
        all_emb_flat = all_context_embeddings.mean(dim=1)                          # (B, D)
        sig_loss = sigreg_loss(all_emb_flat)
        var_loss = variance_loss(all_emb_flat)
        cov_loss = covariance_loss(all_emb_flat)

        reg_loss = sig_loss + 0.5 * var_loss + 0.1 * cov_loss

        total_loss = total_pred_loss + self.sigreg_lambda * reg_loss

        return {
            "loss": total_loss,
            "pred_loss": total_pred_loss.detach(),
            "sigreg_loss": reg_loss.detach(),
        }


class LeJEPAClassifier(nn.Module):
    """
    Classification model using a pre-trained LeJEPA encoder.

    Architecture: Frozen/unfrozen ViT encoder → Global Average Pool → MLP head
    """

    def __init__(
        self,
        encoder: ViTEncoder,
        num_classes: int = 2,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        embed_dim = encoder.embed_dim
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) image tensor
        Returns:
            (B, num_classes) logits
        """
        if self.freeze_encoder:
            with torch.no_grad():
                features = self.encoder.forward_features(x)    # (B, N, D)
        else:
            features = self.encoder.forward_features(x)        # (B, N, D)

        # Global average pool
        pooled = features.mean(dim=1)                          # (B, D)
        logits = self.head(pooled)                             # (B, C)
        return logits


def save_checkpoint(model: nn.Module, path: str, metadata: dict | None = None):
    """Save model checkpoint."""
    state = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(state, path)


def load_checkpoint(model: nn.Module, path: str) -> dict:
    """Load model checkpoint. Returns metadata dict."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"], strict=False)
    return state.get("metadata", {})
