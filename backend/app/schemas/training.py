"""Pydantic schemas for training endpoints."""

from __future__ import annotations

from typing import Optional, List
from pydantic import BaseModel


class TrainingStartRequest(BaseModel):
    run_type: str = "full"             # "pretrain" | "finetune" | "full"
    pretrain_epochs: Optional[int] = None
    finetune_epochs: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None


class TrainingStatusResponse(BaseModel):
    id: int
    run_type: str
    status: str
    current_epoch: int
    total_epochs: int
    best_loss: Optional[float] = None
    best_accuracy: Optional[float] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None


class ModelVersionResponse(BaseModel):
    id: int
    version: str
    model_path: str
    accuracy: Optional[float] = None
    precision_score: Optional[float] = None
    recall_score: Optional[float] = None
    f1_score: Optional[float] = None
    is_active: bool
    created_at: str


class ModelListResponse(BaseModel):
    models: List[ModelVersionResponse]
