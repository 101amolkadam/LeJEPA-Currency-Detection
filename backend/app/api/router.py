"""Main API router — groups all route modules under /api/v1."""

from fastapi import APIRouter

from app.api.analyze import router as analyze_router
from app.api.training import router as training_router

api_router = APIRouter()
api_router.include_router(analyze_router, prefix="/analyze", tags=["Analysis"])
api_router.include_router(training_router, prefix="/training", tags=["Training"])
