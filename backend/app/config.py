"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Database ──────────────────────────────────────────────
    DATABASE_URL: str = "mysql+aiomysql://root:root@localhost:3306/fake_currency_db"
    DB_HOST: str = "localhost"
    DB_PORT: int = 3306
    DB_USER: str = "root"
    DB_PASSWORD: str = "root"
    DB_NAME: str = "fake_currency_db"

    # ── Paths ─────────────────────────────────────────────────
    DATASET_PATH: str = "../dataset/archive/data/data"
    FEATURES_PATH: str = "../dataset/archive/Features/Features"
    MODEL_DIR: str = "./models"

    # ── Server ────────────────────────────────────────────────
    SERVER_PORT: int = 8001
    SECRET_KEY: str = "fake-currency-detection-secret-key-2026"
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

    # ── ML / Training ─────────────────────────────────────────
    IMAGE_SIZE: int = 224
    PATCH_SIZE: int = 16
    EMBED_DIM: int = 192          # ViT-Tiny
    ENCODER_DEPTH: int = 12
    ENCODER_HEADS: int = 3
    PREDICTOR_EMBED_DIM: int = 96
    PREDICTOR_DEPTH: int = 6
    PREDICTOR_HEADS: int = 3
    PRETRAIN_EPOCHS: int = 30
    FINETUNE_EPOCHS: int = 20
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 1.5e-4
    FINETUNE_LR: float = 1e-4
    # NOTE: DEVICE and NUM_WORKERS are auto-detected by app.ml.device

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def dataset_abs_path(self) -> Path:
        return Path(self.DATASET_PATH).resolve()

    @property
    def features_abs_path(self) -> Path:
        return Path(self.FEATURES_PATH).resolve()

    @property
    def model_dir_abs(self) -> Path:
        p = Path(self.MODEL_DIR).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
