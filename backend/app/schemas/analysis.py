"""Pydantic schemas matching the frontend TypeScript types exactly."""

from __future__ import annotations

from typing import Optional, List
from pydantic import BaseModel


# ── Request ───────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    image: str                         # base64 data-URI
    source: str = "upload"             # "upload" | "camera"


# ── Sub-analysis schemas ─────────────────────────────────────
class WatermarkLocation(BaseModel):
    x: float
    y: float
    width: float
    height: float


class WatermarkAnalysis(BaseModel):
    status: str
    confidence: float
    location: Optional[WatermarkLocation] = None
    ssim_score: Optional[float] = None


class ThreadCoordinates(BaseModel):
    x_start: float
    x_end: float


class SecurityThreadAnalysis(BaseModel):
    status: str
    confidence: float
    position: Optional[str] = None
    coordinates: Optional[ThreadCoordinates] = None


class ColorAnalysis(BaseModel):
    status: str
    confidence: float
    bhattacharyya_distance: Optional[float] = None
    dominant_colors: Optional[List[str]] = None


class TextureAnalysis(BaseModel):
    status: str
    confidence: float
    glcm_contrast: Optional[float] = None
    glcm_energy: Optional[float] = None
    sharpness_score: Optional[float] = None


class SerialNumberAnalysis(BaseModel):
    status: str
    confidence: float
    extracted_text: Optional[str] = None
    format_valid: Optional[bool] = None


class DimensionsAnalysis(BaseModel):
    status: str
    confidence: float
    aspect_ratio: Optional[float] = None
    expected_aspect_ratio: Optional[float] = None
    deviation_percent: Optional[float] = None


class CNNClassification(BaseModel):
    result: str
    confidence: float
    model: str
    processing_time_ms: float


class FullAnalysis(BaseModel):
    cnn_classification: CNNClassification
    watermark: WatermarkAnalysis
    security_thread: SecurityThreadAnalysis
    color_analysis: ColorAnalysis
    texture_analysis: TextureAnalysis
    serial_number: SerialNumberAnalysis
    dimensions: DimensionsAnalysis


# ── Main response ────────────────────────────────────────────
class AnalysisResult(BaseModel):
    id: int
    result: str
    confidence: float
    currency_denomination: Optional[str] = None
    denomination_confidence: Optional[float] = None
    analysis: FullAnalysis
    ensemble_score: float
    annotated_image: str               # base64 data-URI
    processing_time_ms: int
    timestamp: str


# ── History ──────────────────────────────────────────────────
class HistoryItem(BaseModel):
    id: int
    result: str
    confidence: float
    denomination: Optional[str] = None
    thumbnail: str                     # base64 data-URI
    analyzed_at: str


class PaginationInfo(BaseModel):
    page: int
    limit: int
    total: int
    total_pages: int


class HistoryResponse(BaseModel):
    data: List[HistoryItem]
    pagination: PaginationInfo
