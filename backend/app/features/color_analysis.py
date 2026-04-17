"""Color histogram analysis for currency authenticity."""

from __future__ import annotations

import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Reference dominant hue ranges for denominations (HSV hue values 0-179)
DENOMINATION_HUE_RANGES = {
    "10":   {"primary": (10, 25),  "name": "brown/orange"},
    "20":   {"primary": (5, 20),   "name": "greenish-yellow"},
    "50":   {"primary": (130, 170),"name": "blue/violet"},
    "100":  {"primary": (80, 130), "name": "blue/lavender"},
    "200":  {"primary": (10, 30),  "name": "orange/yellow"},
    "500":  {"primary": (0, 15),   "name": "stone grey/beige"},
    "2000": {"primary": (160, 179),"name": "magenta/pink"},
}


def analyze_color(image: np.ndarray, denomination: str | None = None) -> dict:
    """
    Analyze color authenticity of currency image.

    Args:
        image: BGR OpenCV image
        denomination: detected denomination string

    Returns:
        dict with status, confidence, bhattacharyya_distance, dominant_colors
    """
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = image.shape[:2]

        # 1. Compute HSV histograms
        hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])

        # Normalize
        cv2.normalize(hist_h, hist_h, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_s, hist_s, 0, 1, cv2.NORM_MINMAX)

        # 2. Dominant hue
        dominant_hue = int(np.argmax(hist_h))
        saturation_mean = float(np.mean(hsv[:, :, 1]))

        # 3. Extract dominant colors using K-means
        pixels = image.reshape(-1, 3).astype(np.float32)
        # Subsample for speed
        if len(pixels) > 10000:
            indices = np.random.choice(len(pixels), 10000, replace=False)
            pixels = pixels[indices]

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        k = 5
        _, labels, centers = cv2.kmeans(
            pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )

        # Convert centers to hex
        dominant_colors = []
        for center in centers:
            b, g, r = int(center[0]), int(center[1]), int(center[2])
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            dominant_colors.append(hex_color)

        # 4. Bhattacharyya distance against expected distribution
        bhattacharyya_dist = 0.0
        if denomination and denomination in DENOMINATION_HUE_RANGES:
            ref = DENOMINATION_HUE_RANGES[denomination]
            hue_low, hue_high = ref["primary"]

            # Create a reference histogram (Gaussian centered on expected hue range)
            ref_hist = np.zeros((180, 1), dtype=np.float32)
            center_hue = (hue_low + hue_high) // 2
            sigma = max((hue_high - hue_low) // 2, 10)
            for i in range(180):
                ref_hist[i] = np.exp(-0.5 * ((i - center_hue) / sigma) ** 2)
            cv2.normalize(ref_hist, ref_hist, 0, 1, cv2.NORM_MINMAX)

            bhattacharyya_dist = float(cv2.compareHist(hist_h, ref_hist, cv2.HISTCMP_BHATTACHARYYA))

        # 5. Score
        score = 0.0

        # Color richness (real notes have good saturation)
        if saturation_mean > 30:
            score += 0.25
        if saturation_mean > 60:
            score += 0.15

        # Histogram shape (real notes have characteristic distributions)
        hist_entropy = -float(np.sum(hist_h[hist_h > 0] * np.log2(hist_h[hist_h > 0] + 1e-10)))
        if hist_entropy > 2.0:  # Some spread indicates proper printing
            score += 0.2

        # Bhattacharyya match (lower = closer match)
        if bhattacharyya_dist < 0.3:
            score += 0.3
        elif bhattacharyya_dist < 0.5:
            score += 0.15

        # Color diversity (real notes have multiple distinct colors)
        color_spread = float(np.std([np.mean(c) for c in centers]))
        if color_spread > 20:
            score += 0.1

        confidence = min(score, 1.0)
        status = "match" if confidence >= 0.45 else "mismatch"

        return {
            "status": status,
            "confidence": round(confidence, 4),
            "bhattacharyya_distance": round(bhattacharyya_dist, 4),
            "dominant_colors": dominant_colors[:5],
        }

    except Exception as e:
        logger.warning("Color analysis error: %s", e)
        return _unknown_result()


def _unknown_result() -> dict:
    return {
        "status": "unknown",
        "confidence": 0.0,
        "bhattacharyya_distance": None,
        "dominant_colors": None,
    }
