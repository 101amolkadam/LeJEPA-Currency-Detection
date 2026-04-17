"""Denomination detection service."""

from __future__ import annotations

from PIL import Image

from app.ml.inference import get_inference_engine


def detect_denomination(pil_image: Image.Image) -> tuple[str, float]:
    """
    Detect currency denomination from image.

    Returns:
        (denomination: str, confidence: float)
    """
    engine = get_inference_engine()
    return engine.predict_denomination(pil_image)
