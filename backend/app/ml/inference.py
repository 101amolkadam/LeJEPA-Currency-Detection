"""Production inference engine — singleton model loader + prediction."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

from app.config import get_settings
from app.ml.device import detect_device
from app.ml.lejepa.encoder import ViTEncoder
from app.ml.lejepa.model import LeJEPAClassifier, load_checkpoint
from app.ml.classifier import (
    build_authenticity_classifier,
    build_denomination_classifier,
    decode_denomination,
    DENOMINATION_LABELS,
)

logger = logging.getLogger(__name__)

_inference_engine = None


class InferenceEngine:
    """Singleton inference engine that caches loaded models."""

    def __init__(self):
        self.settings = get_settings()
        self.device = detect_device()
        self.auth_model: LeJEPAClassifier | None = None
        self.denom_model: LeJEPAClassifier | None = None
        self.model_loaded = False
        self.model_version = "none"

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self._try_load_model()

    def _try_load_model(self):
        """Attempt to load the latest trained model."""
        model_dir = self.settings.model_dir_abs

        # Find latest classifier checkpoint
        classifier_files = sorted(
            model_dir.glob("lejepa_classifier_*.pth"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not classifier_files:
            logger.warning("No trained classifier found in %s — using fallback mode.", model_dir)
            return

        classifier_path = classifier_files[0]
        logger.info("Loading classifier from %s", classifier_path)

        try:
            # Build encoder
            encoder = ViTEncoder(
                img_size=self.settings.IMAGE_SIZE,
                patch_size=self.settings.PATCH_SIZE,
                embed_dim=self.settings.EMBED_DIM,
                depth=self.settings.ENCODER_DEPTH,
                num_heads=self.settings.ENCODER_HEADS,
            )

            # Build classifier and load weights
            self.auth_model = build_authenticity_classifier(encoder, freeze_encoder=False)
            metadata = load_checkpoint(self.auth_model, str(classifier_path))
            self.auth_model.to(self.device)
            self.auth_model.eval()

            # Find matching denomination model
            denom_files = sorted(
                model_dir.glob("lejepa_denom_*.pth"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if denom_files:
                denom_encoder = ViTEncoder(
                    img_size=self.settings.IMAGE_SIZE,
                    patch_size=self.settings.PATCH_SIZE,
                    embed_dim=self.settings.EMBED_DIM,
                    depth=self.settings.ENCODER_DEPTH,
                    num_heads=self.settings.ENCODER_HEADS,
                )
                self.denom_model = build_denomination_classifier(denom_encoder)
                load_checkpoint(self.denom_model, str(denom_files[0]))
                self.denom_model.to(self.device)
                self.denom_model.eval()

            self.model_loaded = True
            self.model_version = classifier_path.stem
            logger.info("Model loaded: %s (accuracy: %.2f%%)",
                        self.model_version,
                        metadata.get("accuracy", 0) * 100)
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            self.model_loaded = False

    def reload_model(self):
        """Force reload the latest model (after training)."""
        self.auth_model = None
        self.denom_model = None
        self.model_loaded = False
        self._try_load_model()

    def predict_authenticity(self, pil_image: Image.Image) -> tuple[str, float, float]:
        """
        Predict if currency is REAL or FAKE.

        Returns:
            (result: "REAL"|"FAKE", confidence: float, processing_time_ms: float)
        """
        start = time.perf_counter()

        if not self.model_loaded or self.auth_model is None:
            # Fallback: return low-confidence neutral result
            elapsed = (time.perf_counter() - start) * 1000
            return "REAL", 0.50, elapsed

        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.auth_model(tensor)
            probs = F.softmax(logits, dim=-1)

        # Class 0 = fake, Class 1 = real
        fake_prob = probs[0, 0].item()
        real_prob = probs[0, 1].item()

        if real_prob >= fake_prob:
            result = "REAL"
            confidence = real_prob
        else:
            result = "FAKE"
            confidence = fake_prob

        elapsed = (time.perf_counter() - start) * 1000
        return result, confidence, elapsed

    def predict_denomination(self, pil_image: Image.Image) -> tuple[str, float]:
        """
        Predict currency denomination.

        Returns:
            (denomination: str, confidence: float)
        """
        if not self.model_loaded or self.denom_model is None:
            return "unknown", 0.0

        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.denom_model(tensor)
            probs = F.softmax(logits, dim=-1)

        class_idx = probs.argmax(dim=-1).item()
        confidence = probs[0, class_idx].item()
        denomination = decode_denomination(class_idx)

        return denomination, confidence


def get_inference_engine() -> InferenceEngine:
    """Get or create the singleton inference engine."""
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = InferenceEngine()
    return _inference_engine
