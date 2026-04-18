"""Training orchestrator — LeJEPA pretraining + supervised fine-tuning."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from app.config import get_settings
from app.ml.device import detect_device, get_optimal_workers, get_pin_memory
from app.ml.lejepa.encoder import ViTEncoder
from app.ml.lejepa.model import LeJEPAPretrainModel, save_checkpoint, load_checkpoint
from app.ml.classifier import build_authenticity_classifier, build_denomination_classifier
from app.ml.dataset import create_dataloaders

logger = logging.getLogger(__name__)


class Trainer:
    """Orchestrates the two-phase training pipeline."""

    def __init__(self, db_session=None):
        self.settings = get_settings()
        self.device = detect_device()
        self.num_workers = get_optimal_workers()
        self.pin_memory = get_pin_memory(self.device)
        self.db_session = db_session

    # ──────────────────────────────────────────────────────────
    # Phase 1: Self-Supervised Pretraining
    # ──────────────────────────────────────────────────────────
    async def pretrain(
        self,
        epochs: int | None = None,
        batch_size: int | None = None,
        lr: float | None = None,
        run_id: int | None = None,
    ) -> str:
        """
        LeJEPA self-supervised pretraining.
        Returns path to saved encoder checkpoint.
        """
        epochs = epochs or self.settings.PRETRAIN_EPOCHS
        batch_size = batch_size or self.settings.BATCH_SIZE
        lr = lr or self.settings.LEARNING_RATE

        logger.info("Starting LeJEPA pretraining: %d epochs, batch=%d, lr=%.6f", epochs, batch_size, lr)

        # Build model
        model = LeJEPAPretrainModel(
            img_size=self.settings.IMAGE_SIZE,
            patch_size=self.settings.PATCH_SIZE,
            embed_dim=self.settings.EMBED_DIM,
            encoder_depth=self.settings.ENCODER_DEPTH,
            encoder_heads=self.settings.ENCODER_HEADS,
            predictor_embed_dim=self.settings.PREDICTOR_EMBED_DIM,
            predictor_depth=self.settings.PREDICTOR_DEPTH,
            predictor_heads=self.settings.PREDICTOR_HEADS,
        ).to(self.device)

        # Data
        train_loader, _ = create_dataloaders(
            self.settings.dataset_abs_path,
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            mode="pretrain",
        )

        # Optimizer
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.05)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        best_loss = float("inf")
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for images, _, _ in train_loader:
                images = images.to(self.device)

                result = model(images)
                loss = result["loss"]

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)

            if avg_loss < best_loss:
                best_loss = avg_loss

            logger.info(
                "Pretrain epoch %d/%d — loss: %.4f (pred: %.4f, sigreg: %.4f) — best: %.4f",
                epoch, epochs, avg_loss,
                result["pred_loss"].item(),
                result["sigreg_loss"].item(),
                best_loss,
            )

            # Update DB
            if self.db_session and run_id:
                await self._update_run(run_id, epoch, epochs, best_loss)

        # Save encoder checkpoint
        checkpoint_path = str(self.settings.model_dir_abs / "lejepa_encoder_pretrained.pth")
        save_checkpoint(model.encoder, checkpoint_path, {
            "type": "pretrained_encoder",
            "epochs": epochs,
            "best_loss": best_loss,
            "embed_dim": self.settings.EMBED_DIM,
        })
        logger.info("Encoder saved to %s", checkpoint_path)
        return checkpoint_path

    # ──────────────────────────────────────────────────────────
    # Phase 2: Supervised Fine-tuning
    # ──────────────────────────────────────────────────────────
    async def finetune(
        self,
        encoder_path: str | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
        lr: float | None = None,
        run_id: int | None = None,
    ) -> tuple[str, dict]:
        """
        Fine-tune the encoder + classification head.
        Returns (model_path, metrics_dict).
        """
        epochs = epochs or self.settings.FINETUNE_EPOCHS
        batch_size = batch_size or self.settings.BATCH_SIZE
        lr = lr or self.settings.FINETUNE_LR

        logger.info("Starting fine-tuning: %d epochs, batch=%d, lr=%.6f", epochs, batch_size, lr)

        # Build encoder
        encoder = ViTEncoder(
            img_size=self.settings.IMAGE_SIZE,
            patch_size=self.settings.PATCH_SIZE,
            embed_dim=self.settings.EMBED_DIM,
            depth=self.settings.ENCODER_DEPTH,
            num_heads=self.settings.ENCODER_HEADS,
        ).to(self.device)

        # Load pretrained weights if available
        if encoder_path and Path(encoder_path).exists():
            load_checkpoint(encoder, encoder_path)
            logger.info("Loaded pretrained encoder from %s", encoder_path)

        # Build classifiers
        auth_classifier = build_authenticity_classifier(encoder, freeze_encoder=False).to(self.device)
        denom_classifier = build_denomination_classifier(encoder, freeze_encoder=True).to(self.device)

        # Data
        train_loader, val_loader = create_dataloaders(
            self.settings.dataset_abs_path,
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            mode="finetune",
        )

        # Class weights for imbalanced dataset
        class_weights = train_loader.dataset.get_class_weights().to(self.device)

        # Optimizer (auth classifier trains encoder too)
        optimizer = AdamW([
            {"params": auth_classifier.encoder.parameters(), "lr": lr * 0.1},
            {"params": auth_classifier.head.parameters(), "lr": lr},
            {"params": denom_classifier.head.parameters(), "lr": lr},
        ], weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        criterion_auth = nn.CrossEntropyLoss(weight=class_weights)
        criterion_denom = nn.CrossEntropyLoss()

        best_acc = 0.0
        best_model_path = ""

        for epoch in range(1, epochs + 1):
            # ── Train ──
            auth_classifier.train()
            denom_classifier.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for images, labels, denoms in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                denoms = denoms.to(self.device)

                # Authenticity
                auth_logits = auth_classifier(images)
                loss_auth = criterion_auth(auth_logits, labels)

                # Denomination
                denom_logits = denom_classifier(images)
                loss_denom = criterion_denom(denom_logits, denoms)

                loss = loss_auth + 0.3 * loss_denom

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(auth_classifier.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                _, predicted = auth_logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            scheduler.step()

            # ── Validate ──
            auth_classifier.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels, _ in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    logits = auth_classifier(images)
                    _, predicted = logits.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            train_acc = correct / max(total, 1)
            val_acc = val_correct / max(val_total, 1)

            logger.info(
                "Finetune epoch %d/%d — loss: %.4f — train_acc: %.2f%% — val_acc: %.2f%%",
                epoch, epochs, train_loss / max(len(train_loader), 1),
                train_acc * 100, val_acc * 100,
            )

            if val_acc > best_acc:
                best_acc = val_acc
                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                best_model_path = str(self.settings.model_dir_abs / f"lejepa_classifier_{ts}.pth")
                save_checkpoint(auth_classifier, best_model_path, {
                    "type": "classifier",
                    "accuracy": val_acc,
                    "epoch": epoch,
                    "embed_dim": self.settings.EMBED_DIM,
                })

                # Also save denomination classifier
                denom_path = str(self.settings.model_dir_abs / f"lejepa_denom_{ts}.pth")
                save_checkpoint(denom_classifier, denom_path, {
                    "type": "denomination_classifier",
                    "epoch": epoch,
                })

            if self.db_session and run_id:
                await self._update_run(run_id, epoch, epochs, best_loss=None, best_accuracy=best_acc)

        metrics = {
            "accuracy": best_acc,
            "epochs": epochs,
        }
        logger.info("Fine-tuning complete. Best val accuracy: %.2f%%", best_acc * 100)
        return best_model_path, metrics

    # ──────────────────────────────────────────────────────────
    # Full pipeline
    # ──────────────────────────────────────────────────────────
    async def train_full(
        self,
        pretrain_epochs: int | None = None,
        finetune_epochs: int | None = None,
        batch_size: int | None = None,
        run_id: int | None = None,
    ) -> tuple[str, dict]:
        """Run pretrain → finetune pipeline."""
        encoder_path = await self.pretrain(
            epochs=pretrain_epochs,
            batch_size=batch_size,
            run_id=run_id,
        )
        model_path, metrics = await self.finetune(
            encoder_path=encoder_path,
            epochs=finetune_epochs,
            batch_size=batch_size,
            run_id=run_id,
        )
        return model_path, metrics

    async def _update_run(self, run_id: int, epoch: int, total: int,
                          best_loss: float | None = None, best_accuracy: float | None = None):
        """Update training_runs table."""
        if not self.db_session:
            return
        try:
            from sqlalchemy import update
            from app.models.training import TrainingRun
            values = {"current_epoch": epoch, "total_epochs": total}
            if best_loss is not None:
                values["best_loss"] = best_loss
            if best_accuracy is not None:
                values["best_accuracy"] = best_accuracy
            stmt = update(TrainingRun).where(TrainingRun.id == run_id).values(**values)
            await self.db_session.execute(stmt)
            await self.db_session.commit()
        except Exception as e:
            logger.warning("Failed to update training run %d: %s", run_id, e)
