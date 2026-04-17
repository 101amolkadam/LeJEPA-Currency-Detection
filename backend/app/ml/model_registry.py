"""Model version registry — tracks and manages trained model versions."""

from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.training import ModelVersion

logger = logging.getLogger(__name__)


async def register_model(
    db: AsyncSession,
    version: str,
    model_path: str,
    training_run_id: int | None = None,
    accuracy: float | None = None,
    precision_score: float | None = None,
    recall_score: float | None = None,
    f1_score: float | None = None,
    activate: bool = True,
) -> ModelVersion:
    """Register a new model version in the database."""
    if activate:
        # Deactivate all existing models
        await db.execute(
            update(ModelVersion).values(is_active=0)
        )

    mv = ModelVersion(
        version=version,
        model_path=model_path,
        training_run_id=training_run_id,
        accuracy=accuracy,
        precision_score=precision_score,
        recall_score=recall_score,
        f1_score=f1_score,
        is_active=1 if activate else 0,
        created_at=datetime.utcnow(),
    )
    db.add(mv)
    await db.commit()
    await db.refresh(mv)
    logger.info("Registered model version '%s' (active=%s)", version, activate)
    return mv


async def get_active_model(db: AsyncSession) -> ModelVersion | None:
    """Get the currently active model version."""
    result = await db.execute(
        select(ModelVersion).where(ModelVersion.is_active == 1).limit(1)
    )
    return result.scalar_one_or_none()


async def activate_model(db: AsyncSession, version: str) -> bool:
    """Activate a specific model version (deactivates all others)."""
    result = await db.execute(
        select(ModelVersion).where(ModelVersion.version == version)
    )
    mv = result.scalar_one_or_none()
    if mv is None:
        return False

    await db.execute(update(ModelVersion).values(is_active=0))
    mv.is_active = 1
    await db.commit()
    logger.info("Activated model version '%s'", version)
    return True


async def list_models(db: AsyncSession) -> list[ModelVersion]:
    """List all registered model versions."""
    result = await db.execute(
        select(ModelVersion).order_by(ModelVersion.created_at.desc())
    )
    return list(result.scalars().all())
