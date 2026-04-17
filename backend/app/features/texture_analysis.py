"""Texture analysis using GLCM features and sharpness scoring."""

from __future__ import annotations

import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _compute_glcm(gray: np.ndarray, distance: int = 1, angle: float = 0.0) -> np.ndarray:
    """Compute Gray-Level Co-occurrence Matrix."""
    levels = 64
    quantized = (gray / 256.0 * levels).astype(np.int32)
    quantized = np.clip(quantized, 0, levels - 1)

    glcm = np.zeros((levels, levels), dtype=np.float64)

    dx = int(round(distance * np.cos(angle)))
    dy = int(round(distance * np.sin(angle)))

    h, w = quantized.shape
    for y in range(max(0, -dy), min(h, h - dy)):
        for x in range(max(0, -dx), min(w, w - dx)):
            i = quantized[y, x]
            j = quantized[y + dy, x + dx]
            glcm[i, j] += 1

    # Normalize
    total = glcm.sum()
    if total > 0:
        glcm /= total

    return glcm


def _glcm_features(glcm: np.ndarray) -> dict:
    """Extract contrast, energy, homogeneity, correlation from GLCM."""
    levels = glcm.shape[0]
    i_indices, j_indices = np.meshgrid(range(levels), range(levels), indexing="ij")
    i_indices = i_indices.astype(np.float64)
    j_indices = j_indices.astype(np.float64)

    # Contrast
    contrast = float(np.sum(glcm * (i_indices - j_indices) ** 2))

    # Energy (Angular Second Moment)
    energy = float(np.sum(glcm ** 2))

    # Homogeneity
    homogeneity = float(np.sum(glcm / (1.0 + np.abs(i_indices - j_indices))))

    # Correlation
    mu_i = np.sum(i_indices * glcm)
    mu_j = np.sum(j_indices * glcm)
    sigma_i = np.sqrt(np.sum(glcm * (i_indices - mu_i) ** 2))
    sigma_j = np.sqrt(np.sum(glcm * (j_indices - mu_j) ** 2))
    if sigma_i > 0 and sigma_j > 0:
        correlation = float(np.sum(glcm * (i_indices - mu_i) * (j_indices - mu_j)) / (sigma_i * sigma_j))
    else:
        correlation = 0.0

    return {
        "contrast": contrast,
        "energy": energy,
        "homogeneity": homogeneity,
        "correlation": correlation,
    }


def analyze_texture(image: np.ndarray, denomination: str | None = None) -> dict:
    """
    Analyze texture quality of currency image.

    Real notes have characteristic micro-printing texture patterns
    that show up in GLCM features and sharpness scores.

    Args:
        image: BGR OpenCV image

    Returns:
        dict with status, confidence, glcm_contrast, glcm_energy, sharpness_score
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Resize for consistent analysis
        analysis_size = 256
        gray_resized = cv2.resize(gray, (analysis_size, analysis_size))

        # 1. GLCM features (average across 4 angles)
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        all_features = []
        for angle in angles:
            glcm = _compute_glcm(gray_resized, distance=1, angle=angle)
            features = _glcm_features(glcm)
            all_features.append(features)

        avg_contrast = float(np.mean([f["contrast"] for f in all_features]))
        avg_energy = float(np.mean([f["energy"] for f in all_features]))
        avg_homogeneity = float(np.mean([f["homogeneity"] for f in all_features]))

        # 2. Sharpness via Laplacian variance
        laplacian = cv2.Laplacian(gray_resized, cv2.CV_64F)
        sharpness = float(np.var(laplacian))

        # Normalize sharpness to 0-1 range (empirical)
        sharpness_norm = min(sharpness / 1500.0, 1.0)

        # 3. Local Binary Pattern-like texture richness
        # Use Sobel gradients as proxy
        sobelx = cv2.Sobel(gray_resized, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_resized, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
        texture_richness = float(np.mean(gradient_mag))

        # 4. Scoring
        score = 0.0

        # Real notes have moderate contrast (not too smooth, not too noisy)
        if 5 < avg_contrast < 200:
            score += 0.25

        # Real notes have low energy (diverse texture, not uniform)
        if avg_energy < 0.1:
            score += 0.2

        # Good sharpness indicates proper printing
        if sharpness_norm > 0.1:
            score += 0.25
        if sharpness_norm > 0.3:
            score += 0.1

        # Texture richness
        if texture_richness > 10:
            score += 0.2

        confidence = min(score, 1.0)
        status = "normal" if confidence >= 0.4 else "abnormal"

        return {
            "status": status,
            "confidence": round(confidence, 4),
            "glcm_contrast": round(avg_contrast, 4),
            "glcm_energy": round(avg_energy, 6),
            "sharpness_score": round(sharpness_norm, 4),
        }

    except Exception as e:
        logger.warning("Texture analysis error: %s", e)
        return _unknown_result()


def _unknown_result() -> dict:
    return {
        "status": "unknown",
        "confidence": 0.0,
        "glcm_contrast": None,
        "glcm_energy": None,
        "sharpness_score": None,
    }
