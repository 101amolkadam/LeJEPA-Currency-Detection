"""Training API endpoints — no password protection (open for dev)."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.training import TrainingRun, ModelVersion
from app.schemas.training import (
    TrainingStartRequest, TrainingStatusResponse,
    ModelVersionResponse, ModelListResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Track active training task
_training_task: asyncio.Task | None = None


async def _run_training(run_id: int, config: dict):
    """Background training task."""
    from app.database import _get_session_factory
    from app.ml.trainer import Trainer

    factory = _get_session_factory()
    async with factory() as db:
        try:
            # Update status to running
            stmt = update(TrainingRun).where(TrainingRun.id == run_id).values(
                status="running", started_at=datetime.utcnow()
            )
            await db.execute(stmt)
            await db.commit()

            trainer = Trainer(db_session=db)

            run_type = config.get("run_type", "full")
            if run_type == "pretrain":
                await trainer.pretrain(
                    epochs=config.get("pretrain_epochs"),
                    batch_size=config.get("batch_size"),
                    run_id=run_id,
                )
            elif run_type == "finetune":
                await trainer.finetune(
                    epochs=config.get("finetune_epochs"),
                    batch_size=config.get("batch_size"),
                    run_id=run_id,
                )
            else:
                model_path, metrics = await trainer.train_full(
                    pretrain_epochs=config.get("pretrain_epochs"),
                    finetune_epochs=config.get("finetune_epochs"),
                    batch_size=config.get("batch_size"),
                    run_id=run_id,
                )

                # Register model version
                from app.ml.model_registry import register_model
                version = datetime.utcnow().strftime("v%Y%m%d_%H%M%S")
                await register_model(
                    db, version=version, model_path=model_path,
                    training_run_id=run_id,
                    accuracy=metrics.get("accuracy"),
                    activate=True,
                )

            # Mark completed
            stmt = update(TrainingRun).where(TrainingRun.id == run_id).values(
                status="completed", completed_at=datetime.utcnow()
            )
            await db.execute(stmt)
            await db.commit()

            # Reload inference engine with new model
            from app.ml.inference import get_inference_engine
            get_inference_engine().reload_model()

            logger.info("Training run %d completed successfully.", run_id)

        except Exception as exc:
            logger.error("Training run %d failed: %s", run_id, exc, exc_info=True)
            try:
                stmt = update(TrainingRun).where(TrainingRun.id == run_id).values(
                    status="failed", error_message=str(exc),
                    completed_at=datetime.utcnow(),
                )
                await db.execute(stmt)
                await db.commit()
            except Exception:
                pass


@router.post("/start", response_model=TrainingStatusResponse)
async def start_training(
    request: TrainingStartRequest,
    db: AsyncSession = Depends(get_db),
):
    """Start model training in the background (no authentication required)."""
    global _training_task

    # Check if training is already running
    result = await db.execute(
        select(TrainingRun).where(TrainingRun.status == "running").limit(1)
    )
    active = result.scalar_one_or_none()
    if active:
        raise HTTPException(
            status_code=409,
            detail=f"Training run #{active.id} is already in progress.",
        )

    from app.config import get_settings
    settings = get_settings()

    config = {
        "run_type": request.run_type,
        "pretrain_epochs": request.pretrain_epochs or settings.PRETRAIN_EPOCHS,
        "finetune_epochs": request.finetune_epochs or settings.FINETUNE_EPOCHS,
        "batch_size": request.batch_size or settings.BATCH_SIZE,
        "learning_rate": request.learning_rate or settings.LEARNING_RATE,
    }

    total_epochs = config["pretrain_epochs"] + config["finetune_epochs"]
    if request.run_type == "pretrain":
        total_epochs = config["pretrain_epochs"]
    elif request.run_type == "finetune":
        total_epochs = config["finetune_epochs"]

    # Create training run record
    run = TrainingRun(
        run_type=request.run_type,
        status="pending",
        config=json.dumps(config),
        total_epochs=total_epochs,
        created_at=datetime.utcnow(),
    )
    db.add(run)
    await db.flush()
    await db.refresh(run)

    # Launch background task
    _training_task = asyncio.create_task(_run_training(run.id, config))

    return TrainingStatusResponse(
        id=run.id,
        run_type=run.run_type,
        status="pending",
        current_epoch=0,
        total_epochs=total_epochs,
    )


@router.get("/status/{run_id}", response_model=TrainingStatusResponse)
async def get_training_status(
    run_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get training run status."""
    result = await db.execute(
        select(TrainingRun).where(TrainingRun.id == run_id)
    )
    run = result.scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail="Training run not found")

    return TrainingStatusResponse(
        id=run.id,
        run_type=run.run_type,
        status=run.status,
        current_epoch=run.current_epoch,
        total_epochs=run.total_epochs,
        best_loss=run.best_loss,
        best_accuracy=run.best_accuracy,
        started_at=run.started_at.isoformat() + "Z" if run.started_at else None,
        completed_at=run.completed_at.isoformat() + "Z" if run.completed_at else None,
        error_message=run.error_message,
    )


@router.get("/models", response_model=ModelListResponse)
async def list_models(db: AsyncSession = Depends(get_db)):
    """List all registered model versions."""
    from app.ml.model_registry import list_models as list_model_versions
    models = await list_model_versions(db)

    return ModelListResponse(
        models=[
            ModelVersionResponse(
                id=m.id,
                version=m.version,
                model_path=m.model_path,
                accuracy=m.accuracy,
                precision_score=m.precision_score,
                recall_score=m.recall_score,
                f1_score=m.f1_score,
                is_active=bool(m.is_active),
                created_at=m.created_at.isoformat() + "Z",
            )
            for m in models
        ]
    )


@router.post("/models/{version}/activate")
async def activate_model_version(
    version: str,
    db: AsyncSession = Depends(get_db),
):
    """Activate a specific model version."""
    from app.ml.model_registry import activate_model
    success = await activate_model(db, version)
    if not success:
        raise HTTPException(status_code=404, detail=f"Model version '{version}' not found")

    # Reload inference engine
    from app.ml.inference import get_inference_engine
    get_inference_engine().reload_model()

    return {"message": f"Model {version} activated."}
