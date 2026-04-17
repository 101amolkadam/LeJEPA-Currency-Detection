"""Security thread detection using edge and line analysis."""

from __future__ import annotations

import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Security thread is typically a vertical metallic strip
THREAD_REGION = {
    "x_rel_start": 0.10,
    "x_rel_end": 0.30,
}


def analyze_security_thread(image: np.ndarray, denomination: str | None = None) -> dict:
    """
    Detect the security thread in a currency image.

    Args:
        image: BGR OpenCV image
        denomination: detected denomination string

    Returns:
        dict with status, confidence, position, coordinates
    """
    try:
        h, w = image.shape[:2]
        x_start = int(w * THREAD_REGION["x_rel_start"])
        x_end = int(w * THREAD_REGION["x_rel_end"])

        # Extract the strip region
        roi = image[:, x_start:x_end]
        if roi.size == 0:
            return _unknown_result()

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 1. Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # 2. Hough Line Transform to find vertical lines
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180,
            threshold=int(h * 0.3),
            minLineLength=int(h * 0.4),
            maxLineGap=int(h * 0.1),
        )

        vertical_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if line is mostly vertical (angle < 15 degrees from vertical)
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                if dy > 0 and dx / dy < 0.27:  # tan(15°) ≈ 0.27
                    vertical_lines.append(line[0])

        # 3. Column-wise intensity analysis (thread appears as bright/dark strip)
        col_means = np.mean(gray, axis=0)
        col_std = np.std(col_means)

        # Find peaks/valleys in column means (indicates metallic thread)
        col_diff = np.abs(np.diff(col_means))
        sharp_transitions = np.sum(col_diff > col_std * 1.5)

        # Scoring
        score = 0.0
        thread_x = None

        if len(vertical_lines) >= 1:
            score += 0.4
            # Get average x position of vertical lines
            avg_x = int(np.mean([l[0] for l in vertical_lines]))
            thread_x = x_start + avg_x

        if sharp_transitions >= 2:
            score += 0.3

        if col_std > 10:
            score += 0.2

        # Bonus if there's a clear narrow bright/dark strip
        if col_std > 5:
            peak_idx = np.argmax(np.abs(col_means - np.mean(col_means)))
            if thread_x is None:
                thread_x = x_start + peak_idx
            score += 0.1

        confidence = min(score, 1.0)
        status = "present" if confidence >= 0.35 else "absent"

        coordinates = None
        position = None
        if thread_x is not None:
            coordinates = {
                "x_start": int(thread_x - 10),
                "x_end": int(thread_x + 10),
            }
            position = "left-third"

        return {
            "status": status,
            "confidence": round(confidence, 4),
            "position": position,
            "coordinates": coordinates,
        }

    except Exception as e:
        logger.warning("Security thread analysis error: %s", e)
        return _unknown_result()


def _unknown_result() -> dict:
    return {
        "status": "unknown",
        "confidence": 0.0,
        "position": None,
        "coordinates": None,
    }
