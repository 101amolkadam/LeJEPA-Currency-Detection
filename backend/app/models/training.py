"""ORM models for training runs and model versions."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, Float, Text, Enum, DateTime, ForeignKey, Index,
)

from app.database import Base


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_type = Column(Enum("pretrain", "finetune", "full", name="run_type_enum"), nullable=False)
    status = Column(
        Enum("pending", "running", "completed", "failed", name="run_status_enum"),
        nullable=False, default="pending",
    )
    config = Column(Text, nullable=False)  # JSON string

    best_loss = Column(Float, nullable=True)
    best_accuracy = Column(Float, nullable=True)
    current_epoch = Column(Integer, nullable=False, default=0)
    total_epochs = Column(Integer, nullable=False)

    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    error_message = Column(Text, nullable=True)

    __table_args__ = (
        Index("idx_status", "status"),
    )


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(String(20), nullable=False, unique=True)
    model_path = Column(String(255), nullable=False)
    training_run_id = Column(Integer, ForeignKey("training_runs.id"), nullable=True)

    accuracy = Column(Float, nullable=True)
    precision_score = Column(Float, nullable=True)
    recall_score = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)

    is_active = Column(Integer, nullable=False, default=0)  # 0/1
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_active", "is_active"),
        Index("idx_version", "version"),
    )
