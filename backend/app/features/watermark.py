"""Watermark detection using template matching and SSIM."""

from __future__ import annotations

import logging
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

logger = logging.getLogger(__name__)

# Expected watermark region (relative coordinates) per denomination
# Indian notes: watermark is typically on the right side with Gandhi portrait
WATERMARK_REGIONS = {
    "default": {"x_rel": 0.65, "y_rel": 0.15, "w_rel": 0.25, "h_rel": 0.55},
}


def analyze_watermark(image: np.ndarray, denomination: str | None = None) -> dict:
    """
    Analyze watermark presence in a currency image.

    Args:
        image: BGR OpenCV image
        denomination: detected denomination string

    Returns:
        dict with status, confidence, location, ssim_score
    """
    try:
        h, w = image.shape[:2]
        region = WATERMARK_REGIONS.get(denomination, WATERMARK_REGIONS["default"])

        x = int(w * region["x_rel"])
        y = int(h * region["y_rel"])
        rw = int(w * region["w_rel"])
        rh = int(h * region["h_rel"])

        # Ensure bounds
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        rw = min(rw, w - x)
        rh = min(rh, h - y)

        # Extract ROI
        roi = image[y:y + rh, x:x + rw]
        if roi.size == 0:
            return _unknown_result()

        # Convert to grayscale
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Watermark detection heuristics:
        # 1. Check if the region has lower contrast (watermarks are subtle)
        std_dev = float(np.std(gray_roi))
        mean_val = float(np.mean(gray_roi))

        # 2. Edge density in watermark region
        edges = cv2.Canny(gray_roi, 30, 100)
        edge_density = float(np.sum(edges > 0)) / max(edges.size, 1)

        # 3. Apply adaptive threshold to reveal watermark pattern
        thresh = cv2.adaptiveThreshold(
            gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 5,
        )
        pattern_score = float(np.std(thresh)) / 128.0  # Normalized

        # 4. SSIM between original and heavily blurred (watermarks survive blur)
        blurred = cv2.GaussianBlur(gray_roi, (15, 15), 0)
        if gray_roi.shape == blurred.shape and gray_roi.size > 0:
            win_size = min(7, gray_roi.shape[0], gray_roi.shape[1])
            if win_size >= 3 and win_size % 2 == 1:
                ssim_score = float(ssim(gray_roi, blurred, win_size=win_size))
            else:
                ssim_score = 0.5
        else:
            ssim_score = 0.5

        # Scoring: watermarks have moderate edge density and moderate contrast
        score = 0.0
        if 15 < std_dev < 60:
            score += 0.3
        if 0.02 < edge_density < 0.15:
            score += 0.3
        if pattern_score > 0.3:
            score += 0.2
        if ssim_score > 0.6:
            score += 0.2

        confidence = min(score, 1.0)
        status = "present" if confidence >= 0.4 else "absent"

        return {
            "status": status,
            "confidence": round(confidence, 4),
            "location": {"x": x, "y": y, "width": rw, "height": rh},
            "ssim_score": round(ssim_score, 4),
        }

    except Exception as e:
        logger.warning("Watermark analysis error: %s", e)
        return _unknown_result()


def _unknown_result() -> dict:
    return {
        "status": "unknown",
        "confidence": 0.0,
        "location": None,
        "ssim_score": None,
    }
