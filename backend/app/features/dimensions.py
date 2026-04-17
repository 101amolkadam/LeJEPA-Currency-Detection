"""Note dimensions and aspect ratio analysis."""

from __future__ import annotations

import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Indian currency note dimensions (mm) and expected aspect ratios
DENOMINATION_DIMENSIONS = {
    "10":   {"width": 123, "height": 63, "aspect": 1.952},
    "20":   {"width": 129, "height": 63, "aspect": 2.048},
    "50":   {"width": 135, "height": 66, "aspect": 2.045},
    "100":  {"width": 142, "height": 66, "aspect": 2.152},
    "200":  {"width": 146, "height": 66, "aspect": 2.212},
    "500":  {"width": 150, "height": 66, "aspect": 2.273},
    "2000": {"width": 166, "height": 66, "aspect": 2.515},
}

DEFAULT_ASPECT = 2.18  # Average across denominations


def analyze_dimensions(image: np.ndarray, denomination: str | None = None) -> dict:
    """
    Analyze note dimensions and aspect ratio.

    Args:
        image: BGR OpenCV image
        denomination: detected denomination string

    Returns:
        dict with status, confidence, aspect_ratio, expected_aspect_ratio, deviation_percent
    """
    try:
        h, w = image.shape[:2]

        # 1. Try to find note contour for precise dimensions
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)

        # Dilate to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest rectangular contour
        note_width = w
        note_height = h

        if contours:
            # Sort by area, get largest
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            # Only use if contour covers enough of the image
            if area > 0.3 * w * h:
                rect = cv2.minAreaRect(largest)
                box_w, box_h = rect[1]
                if box_w > 0 and box_h > 0:
                    note_width = max(box_w, box_h)
                    note_height = min(box_w, box_h)

        # 2. Compute aspect ratio
        if note_height > 0:
            aspect_ratio = float(note_width) / float(note_height)
        else:
            aspect_ratio = float(w) / max(float(h), 1)

        # 3. Get expected aspect ratio
        if denomination and denomination in DENOMINATION_DIMENSIONS:
            expected = DENOMINATION_DIMENSIONS[denomination]["aspect"]
        else:
            expected = DEFAULT_ASPECT

        # 4. Deviation
        deviation = abs(aspect_ratio - expected) / expected * 100

        # 5. Scoring
        score = 0.0

        if deviation < 5:
            score = 0.9
        elif deviation < 10:
            score = 0.7
        elif deviation < 15:
            score = 0.5
        elif deviation < 25:
            score = 0.3
        else:
            score = 0.1

        # Bonus for reasonable absolute ratio (currency notes are 1.8-2.6)
        if 1.7 < aspect_ratio < 2.8:
            score = min(score + 0.1, 1.0)

        confidence = score
        status = "correct" if confidence >= 0.5 else "incorrect"

        return {
            "status": status,
            "confidence": round(confidence, 4),
            "aspect_ratio": round(aspect_ratio, 4),
            "expected_aspect_ratio": round(expected, 4),
            "deviation_percent": round(deviation, 2),
        }

    except Exception as e:
        logger.warning("Dimensions analysis error: %s", e)
        return _unknown_result()


def _unknown_result() -> dict:
    return {
        "status": "unknown",
        "confidence": 0.0,
        "aspect_ratio": None,
        "expected_aspect_ratio": None,
        "deviation_percent": None,
    }
