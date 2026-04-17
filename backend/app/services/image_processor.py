"""Image processing utilities: base64 decode/encode, resize, convert."""

from __future__ import annotations

import base64
import io
import logging
import re

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def decode_base64_image(data_uri: str) -> tuple[np.ndarray, Image.Image]:
    """
    Decode a base64 data-URI or raw base64 string to OpenCV + PIL images.

    Returns:
        (cv2_bgr_image, pil_rgb_image)
    """
    # Strip data URI prefix if present
    if "," in data_uri:
        header, encoded = data_uri.split(",", 1)
    else:
        encoded = data_uri

    img_bytes = base64.b64decode(encoded)

    # PIL image (RGB)
    pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # OpenCV image (BGR)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    cv2_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if cv2_image is None:
        # Fallback: convert from PIL
        cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return cv2_image, pil_image


def encode_image_base64(cv2_image: np.ndarray, fmt: str = ".jpg", quality: int = 90) -> str:
    """Encode an OpenCV image to base64 data-URI string."""
    params = []
    if fmt == ".jpg":
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif fmt == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 6]

    success, buffer = cv2.imencode(fmt, cv2_image, params)
    if not success:
        raise ValueError("Failed to encode image")

    b64 = base64.b64encode(buffer).decode("utf-8")
    mime = "image/jpeg" if fmt == ".jpg" else "image/png"
    return f"data:{mime};base64,{b64}"


def create_thumbnail(cv2_image: np.ndarray, size: int = 96) -> str:
    """Create a small thumbnail as base64 data-URI."""
    h, w = cv2_image.shape[:2]
    scale = size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(cv2_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad to square
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y_off = (size - new_h) // 2
    x_off = (size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    return encode_image_base64(canvas, fmt=".jpg", quality=75)
