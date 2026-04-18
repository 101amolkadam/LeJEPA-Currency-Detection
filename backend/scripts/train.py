"""CLI training script — run from backend/ directory.

Usage:
    uv run python scripts/train.py                          # full 50-epoch training
    uv run python scripts/train.py --mode full              # same as above
    uv run python scripts/train.py --pretrain-epochs 5 --finetune-epochs 3  # quick run
    uv run python scripts/train.py --mode pretrain          # pretrain only
    uv run python scripts/train.py --mode finetune          # finetune only

Hardware is auto-detected: GPU (CUDA) preferred, CPU fallback.
"""

import asyncio
import argparse
import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_banner():
    """Print startup banner with hardware and dataset info."""
    from app.ml.device import detect_device, get_optimal_workers, get_optimal_batch_size
    from app.config import get_settings

    device = detect_device()
    workers = get_optimal_workers()
    batch = get_optimal_batch_size(device)
    settings = get_settings()

    dataset_path = settings.dataset_abs_path
    model_dir = settings.model_dir_abs

    # Count dataset images
    total_images = 0
    class_counts = {"real": 0, "fake": 0}
    if dataset_path.exists():
        for label in ["real", "fake"]:
            label_dir = dataset_path / label
            if label_dir.exists():
                count = sum(1 for f in label_dir.rglob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
                class_counts[label] = count
                total_images += count

    print()
    print("=" * 60)
    print("  LeJEPA Currency Detection — Training")
    print("=" * 60)
    print(f"  Device          : {device}")
    print(f"  DataLoader workers: {workers}")
    print(f"  Default batch   : {batch}")
    print(f"  Dataset path    : {dataset_path}")
    print(f"  Total images    : {total_images}")
    print(f"    Real          : {class_counts['real']}")
    print(f"    Fake          : {class_counts['fake']}")
    print(f"  Model dir       : {model_dir}")
    print("=" * 60)
    print()


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {int(s)}s"
    else:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h)}h {int(m)}m {int(s)}s"


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

    # Print hardware + dataset banner
    print_banner()

    from app.ml.trainer import Trainer

    trainer = Trainer()
    start_time = time.perf_counter()

    if args.mode == "pretrain":
        path = await trainer.pretrain(
            epochs=args.pretrain_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        elapsed = time.perf_counter() - start_time
        print(f"\n{'=' * 60}")
        print(f"  Pretraining complete!")
        print(f"  Encoder saved   : {path}")
        print(f"  Duration        : {format_duration(elapsed)}")
        print(f"{'=' * 60}")

    elif args.mode == "finetune":
        from app.config import get_settings
        settings = get_settings()
        encoder_path = str(settings.model_dir_abs / "lejepa_encoder_pretrained.pth")
        if not os.path.exists(encoder_path):
            encoder_path = None
            print("⚠ No pretrained encoder found — training from scratch")

        path, metrics = await trainer.finetune(
            encoder_path=encoder_path,
            epochs=args.finetune_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        elapsed = time.perf_counter() - start_time
        print(f"\n{'=' * 60}")
        print(f"  Fine-tuning complete!")
        print(f"  Classifier saved: {path}")
        print(f"  Best accuracy   : {metrics['accuracy'] * 100:.2f}%")
        print(f"  Duration        : {format_duration(elapsed)}")
        print(f"{'=' * 60}")

    else:  # full
        path, metrics = await trainer.train_full(
            pretrain_epochs=args.pretrain_epochs,
            finetune_epochs=args.finetune_epochs,
            batch_size=args.batch_size,
        )
        elapsed = time.perf_counter() - start_time
        print(f"\n{'=' * 60}")
        print(f"  Full training complete!")
        print(f"  Classifier saved: {path}")
        print(f"  Best accuracy   : {metrics['accuracy'] * 100:.2f}%")
        print(f"  Duration        : {format_duration(elapsed)}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
