"""CLI training script — run from backend/ directory with: uv run python scripts/train.py"""

import asyncio
import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main():
    parser = argparse.ArgumentParser(description="Train LeJEPA currency detection model")
    parser.add_argument("--mode", choices=["pretrain", "finetune", "full"], default="full",
                        help="Training mode (default: full)")
    parser.add_argument("--pretrain-epochs", type=int, default=None,
                        help="Number of pretraining epochs")
    parser.add_argument("--finetune-epochs", type=int, default=None,
                        help="Number of fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate")
    args = parser.parse_args()

    from app.ml.trainer import Trainer

    trainer = Trainer()

    if args.mode == "pretrain":
        path = await trainer.pretrain(
            epochs=args.pretrain_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        print(f"\nPretrained encoder saved to: {path}")

    elif args.mode == "finetune":
        # Look for existing pretrained encoder
        from app.config import get_settings
        settings = get_settings()
        encoder_path = str(settings.model_dir_abs / "lejepa_encoder_pretrained.pth")
        if not os.path.exists(encoder_path):
            encoder_path = None
            print("No pretrained encoder found — training from scratch")

        path, metrics = await trainer.finetune(
            encoder_path=encoder_path,
            epochs=args.finetune_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        print(f"\nClassifier saved to: {path}")
        print(f"Metrics: {metrics}")

    else:  # full
        path, metrics = await trainer.train_full(
            pretrain_epochs=args.pretrain_epochs,
            finetune_epochs=args.finetune_epochs,
            batch_size=args.batch_size,
        )
        print(f"\nFull training complete!")
        print(f"Model saved to: {path}")
        print(f"Metrics: {metrics}")


if __name__ == "__main__":
    asyncio.run(main())
