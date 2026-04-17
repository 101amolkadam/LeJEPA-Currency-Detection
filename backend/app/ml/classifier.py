"""Classification head wrapper and denomination classifier."""

from __future__ import annotations

import torch
import torch.nn as nn

from app.ml.lejepa.encoder import ViTEncoder
from app.ml.lejepa.model import LeJEPAClassifier


def build_authenticity_classifier(
    encoder: ViTEncoder,
    freeze_encoder: bool = False,
) -> LeJEPAClassifier:
    """Build a REAL/FAKE binary classifier on top of the encoder."""
    return LeJEPAClassifier(
        encoder=encoder,
        num_classes=2,
        freeze_encoder=freeze_encoder,
    )


def build_denomination_classifier(
    encoder: ViTEncoder,
    freeze_encoder: bool = True,
) -> LeJEPAClassifier:
    """Build a 7-class denomination classifier (₹10–₹2000)."""
    return LeJEPAClassifier(
        encoder=encoder,
        num_classes=7,
        freeze_encoder=freeze_encoder,
    )


DENOMINATION_LABELS = ["10", "20", "50", "100", "200", "500", "2000"]


def decode_denomination(class_idx: int) -> str:
    """Map class index to denomination string."""
    if 0 <= class_idx < len(DENOMINATION_LABELS):
        return DENOMINATION_LABELS[class_idx]
    return "unknown"
