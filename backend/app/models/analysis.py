"""ORM model for currency analysis records."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, Float, Text, Enum, DateTime, Index,
)
from sqlalchemy.dialects.mysql import LONGTEXT, MEDIUMTEXT

from app.database import Base


class AnalysisRecord(Base):
    __tablename__ = "analysis_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    result = Column(Enum("REAL", "FAKE", name="result_enum"), nullable=False)
    confidence = Column(Float, nullable=False)
    currency_denomination = Column(String(10), nullable=True)
    denomination_confidence = Column(Float, nullable=True)

    # CNN classification
    cnn_result = Column(String(10), nullable=False)
    cnn_confidence = Column(Float, nullable=False)
    cnn_model = Column(String(50), nullable=False)
    cnn_time_ms = Column(Float, nullable=False)

    # Watermark
    watermark_status = Column(String(20), nullable=False, default="unknown")
    watermark_confidence = Column(Float, nullable=False, default=0.0)
    watermark_location = Column(Text, nullable=True)   # JSON string
    watermark_ssim = Column(Float, nullable=True)

    # Security Thread
    thread_status = Column(String(20), nullable=False, default="unknown")
    thread_confidence = Column(Float, nullable=False, default=0.0)
    thread_position = Column(String(50), nullable=True)
    thread_coordinates = Column(Text, nullable=True)    # JSON string

    # Color Analysis
    color_status = Column(String(20), nullable=False, default="unknown")
    color_confidence = Column(Float, nullable=False, default=0.0)
    color_bhattacharyya = Column(Float, nullable=True)
    color_dominant = Column(Text, nullable=True)         # JSON string

    # Texture Analysis
    texture_status = Column(String(20), nullable=False, default="unknown")
    texture_confidence = Column(Float, nullable=False, default=0.0)
    texture_contrast = Column(Float, nullable=True)
    texture_energy = Column(Float, nullable=True)
    texture_sharpness = Column(Float, nullable=True)

    # Serial Number
    serial_status = Column(String(20), nullable=False, default="unknown")
    serial_confidence = Column(Float, nullable=False, default=0.0)
    serial_text = Column(String(50), nullable=True)
    serial_format_valid = Column(Integer, nullable=True)  # 0/1

    # Dimensions
    dim_status = Column(String(20), nullable=False, default="unknown")
    dim_confidence = Column(Float, nullable=False, default=0.0)
    dim_aspect_ratio = Column(Float, nullable=True)
    dim_expected_ratio = Column(Float, nullable=True)
    dim_deviation = Column(Float, nullable=True)

    # Ensemble & meta
    ensemble_score = Column(Float, nullable=False)
    source = Column(Enum("upload", "camera", name="source_enum"), nullable=False, default="upload")
    original_image = Column(LONGTEXT, nullable=False)
    annotated_image = Column(LONGTEXT, nullable=False)
    thumbnail = Column(MEDIUMTEXT, nullable=False)
    processing_time_ms = Column(Integer, nullable=False)
    analyzed_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_result", "result"),
        Index("idx_analyzed_at", analyzed_at.desc()),
        Index("idx_denomination", "currency_denomination"),
    )
