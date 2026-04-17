"""Main analysis orchestrator — coordinates all analysis components."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime

import numpy as np
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml.inference import get_inference_engine
from app.services.image_processor import decode_base64_image, encode_image_base64, create_thumbnail
from app.services.annotator import annotate_image
from app.services.denomination import detect_denomination
from app.features.watermark import analyze_watermark
from app.features.security_thread import analyze_security_thread
from app.features.color_analysis import analyze_color
from app.features.texture_analysis import analyze_texture
from app.features.serial_number import analyze_serial_number
from app.features.dimensions import analyze_dimensions
from app.models.analysis import AnalysisRecord
from app.schemas.analysis import (
    AnalysisResult, FullAnalysis,
    CNNClassification, WatermarkAnalysis, WatermarkLocation,
    SecurityThreadAnalysis, ThreadCoordinates,
    ColorAnalysis, TextureAnalysis, SerialNumberAnalysis, DimensionsAnalysis,
)

logger = logging.getLogger(__name__)

# Ensemble weights
ENSEMBLE_WEIGHTS = {
    "cnn": 0.50,
    "watermark": 0.10,
    "security_thread": 0.08,
    "color": 0.10,
    "texture": 0.08,
    "serial": 0.07,
    "dimensions": 0.07,
}


def _confidence_to_real_score(status: str, confidence: float) -> float:
    """Convert feature status+confidence to a 'realness' score (0-1)."""
    pass_statuses = {"present", "match", "normal", "valid", "correct"}
    fail_statuses = {"absent", "mismatch", "abnormal", "invalid", "incorrect"}

    if status in pass_statuses:
        return confidence
    elif status in fail_statuses:
        return 1.0 - confidence
    return 0.5  # unknown


async def analyze_currency(
    image_b64: str,
    source: str,
    db: AsyncSession,
) -> AnalysisResult:
    """
    Full analysis pipeline:
    1. Decode base64 image
    2. Run CNN (LeJEPA) classification
    3. Run 6 traditional CV feature analyzers
    4. Detect denomination
    5. Compute ensemble score
    6. Annotate image
    7. Store in MySQL
    8. Return AnalysisResult

    Args:
        image_b64: base64-encoded image (data URI or raw)
        source: "upload" or "camera"
        db: async database session

    Returns:
        AnalysisResult matching frontend contract
    """
    total_start = time.perf_counter()

    # ── 1. Decode image ──
    cv2_image, pil_image = decode_base64_image(image_b64)

    # ── 2. CNN classification (LeJEPA) ──
    engine = get_inference_engine()
    cnn_result, cnn_confidence, cnn_time = engine.predict_authenticity(pil_image)

    # ── 3. Denomination detection ──
    denomination, denom_confidence = detect_denomination(pil_image)

    # ── 4. Traditional feature analysis ──
    watermark_result = analyze_watermark(cv2_image, denomination)
    thread_result = analyze_security_thread(cv2_image, denomination)
    color_result = analyze_color(cv2_image, denomination)
    texture_result = analyze_texture(cv2_image, denomination)
    serial_result = analyze_serial_number(cv2_image, denomination)
    dimension_result = analyze_dimensions(cv2_image, denomination)

    # ── 5. Ensemble score ──
    scores = {
        "cnn": cnn_confidence if cnn_result == "REAL" else (1.0 - cnn_confidence),
        "watermark": _confidence_to_real_score(watermark_result["status"], watermark_result["confidence"]),
        "security_thread": _confidence_to_real_score(thread_result["status"], thread_result["confidence"]),
        "color": _confidence_to_real_score(color_result["status"], color_result["confidence"]),
        "texture": _confidence_to_real_score(texture_result["status"], texture_result["confidence"]),
        "serial": _confidence_to_real_score(serial_result["status"], serial_result["confidence"]),
        "dimensions": _confidence_to_real_score(dimension_result["status"], dimension_result["confidence"]),
    }

    ensemble_score = sum(ENSEMBLE_WEIGHTS[k] * scores[k] for k in ENSEMBLE_WEIGHTS)
    overall_result = "REAL" if ensemble_score >= 0.5 else "FAKE"
    overall_confidence = ensemble_score if overall_result == "REAL" else (1.0 - ensemble_score)

    # ── 6. Annotate image ──
    analysis_dict = {
        "cnn_classification": {"result": cnn_result, "confidence": cnn_confidence},
        "watermark": watermark_result,
        "security_thread": thread_result,
        "color_analysis": color_result,
        "texture_analysis": texture_result,
        "serial_number": serial_result,
        "dimensions": dimension_result,
    }
    annotated_cv2 = annotate_image(cv2_image, analysis_dict)
    annotated_b64 = encode_image_base64(annotated_cv2)

    # ── 7. Thumbnail ──
    thumbnail_b64 = create_thumbnail(cv2_image)

    total_time_ms = int((time.perf_counter() - total_start) * 1000)
    timestamp = datetime.utcnow()

    def _clean_float(v):
        if v is None:
            return None
        v = float(v)
        import math
        return None if math.isnan(v) else v

    # ── 8. Store in database ──
    record = AnalysisRecord(
        result=overall_result,
        confidence=round(overall_confidence, 4),
        currency_denomination=denomination if denomination != "unknown" else None,
        denomination_confidence=round(denom_confidence, 4) if denomination != "unknown" else None,
        cnn_result=cnn_result,
        cnn_confidence=round(cnn_confidence, 4),
        cnn_model="LeJEPA-ViT-T",
        cnn_time_ms=round(cnn_time, 2),
        watermark_status=watermark_result["status"],
        watermark_confidence=round(watermark_result["confidence"], 4),
        watermark_location=json.dumps(watermark_result["location"]) if watermark_result["location"] else None,
        watermark_ssim=_clean_float(watermark_result.get("ssim_score")),
        thread_status=thread_result["status"],
        thread_confidence=round(thread_result["confidence"], 4),
        thread_position=thread_result.get("position"),
        thread_coordinates=json.dumps(thread_result["coordinates"]) if thread_result.get("coordinates") else None,
        color_status=color_result["status"],
        color_confidence=round(color_result["confidence"], 4),
        color_bhattacharyya=_clean_float(color_result.get("bhattacharyya_distance")),
        color_dominant=json.dumps(color_result.get("dominant_colors")) if color_result.get("dominant_colors") else None,
        texture_status=texture_result["status"],
        texture_confidence=round(texture_result["confidence"], 4),
        texture_contrast=_clean_float(texture_result.get("glcm_contrast")),
        texture_energy=_clean_float(texture_result.get("glcm_energy")),
        texture_sharpness=_clean_float(texture_result.get("sharpness_score")),
        serial_status=serial_result["status"],
        serial_confidence=round(serial_result["confidence"], 4),
        serial_text=serial_result.get("extracted_text"),
        serial_format_valid=1 if serial_result.get("format_valid") else 0 if serial_result.get("format_valid") is not None else None,
        dim_status=dimension_result["status"],
        dim_confidence=round(dimension_result["confidence"], 4),
        dim_aspect_ratio=_clean_float(dimension_result.get("aspect_ratio")),
        dim_expected_ratio=_clean_float(dimension_result.get("expected_aspect_ratio")),
        dim_deviation=_clean_float(dimension_result.get("deviation_percent")),
        ensemble_score=round(ensemble_score, 4),
        source=source,
        original_image=image_b64,
        annotated_image=annotated_b64,
        thumbnail=thumbnail_b64,
        processing_time_ms=total_time_ms,
        analyzed_at=timestamp,
    )
    db.add(record)
    await db.flush()
    await db.refresh(record)

    # ── 9. Build response ──
    response = AnalysisResult(
        id=record.id,
        result=overall_result,
        confidence=round(overall_confidence, 4),
        currency_denomination=denomination if denomination != "unknown" else None,
        denomination_confidence=round(denom_confidence, 4) if denomination != "unknown" else None,
        analysis=FullAnalysis(
            cnn_classification=CNNClassification(
                result=cnn_result,
                confidence=round(cnn_confidence, 4),
                model="LeJEPA-ViT-T",
                processing_time_ms=round(cnn_time, 2),
            ),
            watermark=WatermarkAnalysis(
                status=watermark_result["status"],
                confidence=round(watermark_result["confidence"], 4),
                location=WatermarkLocation(**watermark_result["location"]) if watermark_result["location"] else None,
                ssim_score=watermark_result.get("ssim_score"),
            ),
            security_thread=SecurityThreadAnalysis(
                status=thread_result["status"],
                confidence=round(thread_result["confidence"], 4),
                position=thread_result.get("position"),
                coordinates=ThreadCoordinates(**thread_result["coordinates"]) if thread_result.get("coordinates") else None,
            ),
            color_analysis=ColorAnalysis(
                status=color_result["status"],
                confidence=round(color_result["confidence"], 4),
                bhattacharyya_distance=color_result.get("bhattacharyya_distance"),
                dominant_colors=color_result.get("dominant_colors"),
            ),
            texture_analysis=TextureAnalysis(
                status=texture_result["status"],
                confidence=round(texture_result["confidence"], 4),
                glcm_contrast=texture_result.get("glcm_contrast"),
                glcm_energy=texture_result.get("glcm_energy"),
                sharpness_score=texture_result.get("sharpness_score"),
            ),
            serial_number=SerialNumberAnalysis(
                status=serial_result["status"],
                confidence=round(serial_result["confidence"], 4),
                extracted_text=serial_result.get("extracted_text"),
                format_valid=serial_result.get("format_valid"),
            ),
            dimensions=DimensionsAnalysis(
                status=dimension_result["status"],
                confidence=round(dimension_result["confidence"], 4),
                aspect_ratio=dimension_result.get("aspect_ratio"),
                expected_aspect_ratio=dimension_result.get("expected_aspect_ratio"),
                deviation_percent=dimension_result.get("deviation_percent"),
            ),
        ),
        ensemble_score=round(ensemble_score, 4),
        annotated_image=annotated_b64,
        processing_time_ms=total_time_ms,
        timestamp=timestamp.isoformat() + "Z",
    )

    logger.info(
        "Analysis #%d: %s (%.1f%%) — denomination: %s — %dms",
        record.id, overall_result, overall_confidence * 100,
        denomination, total_time_ms,
    )

    return response
