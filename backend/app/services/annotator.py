"""Image annotation — draws analysis results on the currency image."""

from __future__ import annotations

import cv2
import numpy as np


# Colors (BGR)
COLOR_PASS = (0, 200, 80)      # Green
COLOR_FAIL = (0, 0, 220)       # Red
COLOR_WARN = (0, 200, 220)     # Yellow
COLOR_INFO = (220, 160, 0)     # Blue


def _status_color(status: str) -> tuple[int, int, int]:
    """Map feature status to a color."""
    pass_statuses = {"present", "match", "normal", "valid", "correct"}
    fail_statuses = {"absent", "mismatch", "abnormal", "invalid", "incorrect"}

    if status in pass_statuses:
        return COLOR_PASS
    elif status in fail_statuses:
        return COLOR_FAIL
    else:
        return COLOR_WARN


def annotate_image(
    image: np.ndarray,
    analysis_results: dict,
) -> np.ndarray:
    """
    Draw analysis results on the image.

    Args:
        image: BGR OpenCV image (will be copied)
        analysis_results: dict with all feature analysis results

    Returns:
        Annotated image (BGR)
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.35, min(w, h) / 800)
    thickness = max(1, int(font_scale * 2))

    y_text = 25

    # ── Watermark ──
    wm = analysis_results.get("watermark", {})
    if wm.get("location"):
        loc = wm["location"]
        color = _status_color(wm.get("status", "unknown"))
        cv2.rectangle(
            annotated,
            (int(loc["x"]), int(loc["y"])),
            (int(loc["x"] + loc["width"]), int(loc["y"] + loc["height"])),
            color, 2,
        )
        label = f"Watermark: {wm['status']} ({wm['confidence']:.0%})"
        cv2.putText(annotated, label,
                    (int(loc["x"]), int(loc["y"]) - 5),
                    font, font_scale, color, thickness)

    # ── Security Thread ──
    st = analysis_results.get("security_thread", {})
    if st.get("coordinates"):
        coords = st["coordinates"]
        color = _status_color(st.get("status", "unknown"))
        x_s = int(coords["x_start"])
        x_e = int(coords["x_end"])
        cv2.rectangle(annotated, (x_s, 0), (x_e, h), color, 2)
        label = f"Thread: {st['status']} ({st['confidence']:.0%})"
        cv2.putText(annotated, label, (x_s, 20),
                    font, font_scale, color, thickness)

    # ── Text overlay for other features ──
    features_text = [
        ("Color", analysis_results.get("color_analysis", {})),
        ("Texture", analysis_results.get("texture_analysis", {})),
        ("Serial", analysis_results.get("serial_number", {})),
        ("Dimensions", analysis_results.get("dimensions", {})),
    ]

    for name, feat in features_text:
        status = feat.get("status", "unknown")
        conf = feat.get("confidence", 0)
        color = _status_color(status)
        label = f"{name}: {status} ({conf:.0%})"
        cv2.putText(annotated, label, (10, y_text),
                    font, font_scale * 0.9, color, thickness)
        y_text += int(25 * font_scale / 0.35)

    # ── Overall result banner ──
    cnn = analysis_results.get("cnn_classification", {})
    result = cnn.get("result", "?")
    result_conf = cnn.get("confidence", 0)
    banner_color = COLOR_PASS if result == "REAL" else COLOR_FAIL

    # Draw banner at bottom
    banner_h = int(40 * font_scale / 0.35)
    cv2.rectangle(annotated, (0, h - banner_h), (w, h), banner_color, -1)
    result_text = f"{result} ({result_conf:.0%})"
    text_size = cv2.getTextSize(result_text, font, font_scale * 1.2, thickness + 1)[0]
    text_x = (w - text_size[0]) // 2
    text_y = h - (banner_h - text_size[1]) // 2
    cv2.putText(annotated, result_text, (text_x, text_y),
                font, font_scale * 1.2, (255, 255, 255), thickness + 1)

    return annotated
