"""Serial number extraction and validation using EasyOCR."""

from __future__ import annotations

import logging
import re
import cv2
import numpy as np

logger = logging.getLogger(__name__)

_reader = None

# Indian currency serial number patterns
# Format: {prefix}{2-letter series}{6-digit number} e.g., 0AB123456
SERIAL_PATTERNS = [
    re.compile(r"[0-9][A-Z]{2}\s*[0-9]{6}"),      # Standard: 0AB 123456
    re.compile(r"[A-Z]{2,3}\s*[0-9]{5,7}"),         # Alternative
    re.compile(r"[0-9]{2}[A-Z]\s*[0-9]{6}"),         # Older format
]


def _get_reader():
    global _reader
    if _reader is None:
        try:
            import easyocr
            _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        except Exception as e:
            logger.error("Failed to initialize EasyOCR: %s", e)
            return None
    return _reader


def analyze_serial_number(image: np.ndarray, denomination: str | None = None) -> dict:
    """
    Extract and validate serial number from currency image.

    Args:
        image: BGR OpenCV image

    Returns:
        dict with status, confidence, extracted_text, format_valid
    """
    try:
        h, w = image.shape[:2]

        # Serial number regions: bottom-left and top-right on Indian notes
        rois = [
            image[int(h * 0.80):h, 0:int(w * 0.50)],       # Bottom-left
            image[0:int(h * 0.20), int(w * 0.50):w],        # Top-right
            image[int(h * 0.85):h, int(w * 0.50):w],        # Bottom-right
        ]

        reader = _get_reader()
        if reader is None:
            return _unknown_result()

        best_text = ""
        best_confidence = 0.0
        format_valid = False

        for roi in rois:
            if roi.size == 0:
                continue

            # Preprocess ROI
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            enhanced = clahe.apply(gray)
            # Threshold
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # OCR
            try:
                results = reader.readtext(binary, detail=1)
            except Exception:
                results = []

            for bbox, text, conf in results:
                text_clean = text.strip().upper().replace(" ", "")

                # Check against patterns
                for pattern in SERIAL_PATTERNS:
                    match = pattern.search(text_clean)
                    if match and conf > best_confidence:
                        best_text = match.group(0)
                        best_confidence = float(conf)
                        format_valid = True

                # Even without pattern match, keep best OCR result
                if not format_valid and len(text_clean) >= 6 and conf > best_confidence:
                    best_text = text_clean
                    best_confidence = float(conf) * 0.5  # Lower confidence for unmatched

        if not best_text:
            return _unknown_result()

        # Final scoring
        confidence = best_confidence
        if format_valid:
            confidence = min(confidence + 0.2, 1.0)

        status = "valid" if format_valid and confidence > 0.3 else "invalid"
        if not best_text:
            status = "unknown"

        return {
            "status": status,
            "confidence": round(confidence, 4),
            "extracted_text": best_text if best_text else None,
            "format_valid": format_valid,
        }

    except Exception as e:
        logger.warning("Serial number analysis error: %s", e)
        return _unknown_result()


def _unknown_result() -> dict:
    return {
        "status": "unknown",
        "confidence": 0.0,
        "extracted_text": None,
        "format_valid": None,
    }
