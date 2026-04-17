"""Model evaluation script — run from backend/ directory with: uv run python scripts/evaluate.py"""

import asyncio
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main():
    parser = argparse.ArgumentParser(description="Evaluate LeJEPA currency detection model")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to classifier checkpoint (uses latest if not specified)")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    import torch
    from app.config import get_settings
    from app.ml.lejepa.encoder import ViTEncoder
    from app.ml.lejepa.model import load_checkpoint
    from app.ml.classifier import build_authenticity_classifier
    from app.ml.dataset import create_dataloaders

    settings = get_settings()
    device = torch.device(settings.DEVICE)

    # Find model
    model_path = args.model_path
    if model_path is None:
        model_files = sorted(
            settings.model_dir_abs.glob("lejepa_classifier_*.pth"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not model_files:
            print("ERROR: No trained model found. Run training first.")
            return
        model_path = str(model_files[0])

    print(f"Evaluating: {model_path}")

    # Build model
    encoder = ViTEncoder(
        img_size=settings.IMAGE_SIZE,
        patch_size=settings.PATCH_SIZE,
        embed_dim=settings.EMBED_DIM,
        depth=settings.ENCODER_DEPTH,
        num_heads=settings.ENCODER_HEADS,
    )
    model = build_authenticity_classifier(encoder, freeze_encoder=False)
    load_checkpoint(model, model_path)
    model.to(device)
    model.eval()

    # Load validation data
    _, val_loader = create_dataloaders(
        settings.dataset_abs_path,
        batch_size=args.batch_size,
        num_workers=settings.NUM_WORKERS,
        mode="finetune",
    )

    # Evaluate
    correct = 0
    total = 0
    tp = fp = tn = fn = 0

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            _, predicted = logits.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Confusion matrix (label 1 = real = positive)
            for p, l in zip(predicted.cpu().tolist(), labels.cpu().tolist()):
                if p == 1 and l == 1:
                    tp += 1
                elif p == 1 and l == 0:
                    fp += 1
                elif p == 0 and l == 0:
                    tn += 1
                elif p == 0 and l == 1:
                    fn += 1

    accuracy = correct / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    print(f"\n{'='*50}")
    print(f"  Evaluation Results")
    print(f"{'='*50}")
    print(f"  Samples evaluated: {total}")
    print(f"  Accuracy:          {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"  Precision:         {precision:.4f}")
    print(f"  Recall:            {recall:.4f}")
    print(f"  F1 Score:          {f1:.4f}")
    print(f"{'='*50}")
    print(f"  Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                FAKE    REAL")
    print(f"  Actual FAKE   {tn:5d}   {fp:5d}")
    print(f"  Actual REAL   {fn:5d}   {tp:5d}")
    print(f"{'='*50}")


if __name__ == "__main__":
    asyncio.run(main())
