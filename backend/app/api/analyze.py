"""Analysis API endpoints."""

from __future__ import annotations

import json
import logging
import math

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.analysis import AnalysisRecord
from app.schemas.analysis import (
    AnalyzeRequest, AnalysisResult, HistoryResponse,
    HistoryItem, PaginationInfo,
    FullAnalysis, CNNClassification,
    WatermarkAnalysis, WatermarkLocation,
    SecurityThreadAnalysis, ThreadCoordinates,
    ColorAnalysis, TextureAnalysis, SerialNumberAnalysis, DimensionsAnalysis,
)
from app.services.analyzer import analyze_currency

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("", response_model=AnalysisResult)
async def analyze(request: AnalyzeRequest, db: AsyncSession = Depends(get_db)):
    """
    Analyze a currency image for authenticity.

    Accepts a base64-encoded image and returns comprehensive analysis
    including LeJEPA CNN classification and traditional CV feature analysis.
    """
    try:
        result = await analyze_currency(request.image, request.source, db)
        return result
    except Exception as exc:
        logger.error("Analysis failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(exc)}")


@router.get("/history", response_model=HistoryResponse)
async def get_history(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    filter: str = Query("all"),
    db: AsyncSession = Depends(get_db),
):
    """Get paginated analysis history with optional filtering."""

    # Base query
    query = select(AnalysisRecord)
    count_query = select(func.count(AnalysisRecord.id))

    # Filter
    if filter == "real":
        query = query.where(AnalysisRecord.result == "REAL")
        count_query = count_query.where(AnalysisRecord.result == "REAL")
    elif filter == "fake":
        query = query.where(AnalysisRecord.result == "FAKE")
        count_query = count_query.where(AnalysisRecord.result == "FAKE")

    # Total count
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0
    total_pages = max(1, math.ceil(total / limit))

    # Paginate
    offset = (page - 1) * limit
    query = query.order_by(desc(AnalysisRecord.analyzed_at)).offset(offset).limit(limit)
    result = await db.execute(query)
    records = result.scalars().all()

    items = [
        HistoryItem(
            id=r.id,
            result=r.result,
            confidence=r.confidence,
            denomination=r.currency_denomination,
            thumbnail=r.thumbnail,
            analyzed_at=r.analyzed_at.isoformat() + "Z",
        )
        for r in records
    ]

    return HistoryResponse(
        data=items,
        pagination=PaginationInfo(
            page=page,
            limit=limit,
            total=total,
            total_pages=total_pages,
        ),
    )


@router.get("/history/{analysis_id}", response_model=AnalysisResult)
async def get_analysis_by_id(
    analysis_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get a specific analysis result by ID."""
    result = await db.execute(
        select(AnalysisRecord).where(AnalysisRecord.id == analysis_id)
    )
    record = result.scalar_one_or_none()

    if record is None:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return _record_to_response(record)


@router.delete("/history/{analysis_id}", status_code=204)
async def delete_analysis(
    analysis_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Delete an analysis record."""
    result = await db.execute(
        select(AnalysisRecord).where(AnalysisRecord.id == analysis_id)
    )
    record = result.scalar_one_or_none()

    if record is None:
        raise HTTPException(status_code=404, detail="Analysis not found")

    await db.delete(record)
    return None


def _record_to_response(r: AnalysisRecord) -> AnalysisResult:
    """Convert a database record to API response schema."""
    # Parse JSON fields
    watermark_loc = json.loads(r.watermark_location) if r.watermark_location else None
    thread_coords = json.loads(r.thread_coordinates) if r.thread_coordinates else None
    dominant_colors = json.loads(r.color_dominant) if r.color_dominant else None

    return AnalysisResult(
        id=r.id,
        result=r.result,
        confidence=r.confidence,
        currency_denomination=r.currency_denomination,
        denomination_confidence=r.denomination_confidence,
        analysis=FullAnalysis(
            cnn_classification=CNNClassification(
                result=r.cnn_result,
                confidence=r.cnn_confidence,
                model=r.cnn_model,
                processing_time_ms=r.cnn_time_ms,
            ),
            watermark=WatermarkAnalysis(
                status=r.watermark_status,
                confidence=r.watermark_confidence,
                location=WatermarkLocation(**watermark_loc) if watermark_loc else None,
                ssim_score=r.watermark_ssim,
            ),
            security_thread=SecurityThreadAnalysis(
                status=r.thread_status,
                confidence=r.thread_confidence,
                position=r.thread_position,
                coordinates=ThreadCoordinates(**thread_coords) if thread_coords else None,
            ),
            color_analysis=ColorAnalysis(
                status=r.color_status,
                confidence=r.color_confidence,
                bhattacharyya_distance=r.color_bhattacharyya,
                dominant_colors=dominant_colors,
            ),
            texture_analysis=TextureAnalysis(
                status=r.texture_status,
                confidence=r.texture_confidence,
                glcm_contrast=r.texture_contrast,
                glcm_energy=r.texture_energy,
                sharpness_score=r.texture_sharpness,
            ),
            serial_number=SerialNumberAnalysis(
                status=r.serial_status,
                confidence=r.serial_confidence,
                extracted_text=r.serial_text,
                format_valid=bool(r.serial_format_valid) if r.serial_format_valid is not None else None,
            ),
            dimensions=DimensionsAnalysis(
                status=r.dim_status,
                confidence=r.dim_confidence,
                aspect_ratio=r.dim_aspect_ratio,
                expected_aspect_ratio=r.dim_expected_ratio,
                deviation_percent=r.dim_deviation,
            ),
        ),
        ensemble_score=r.ensemble_score,
        annotated_image=r.annotated_image,
        processing_time_ms=r.processing_time_ms,
        timestamp=r.analyzed_at.isoformat() + "Z",
    )
