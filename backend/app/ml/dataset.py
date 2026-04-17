"""PyTorch Dataset for Indian currency images."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

from app.ml.classifier import DENOMINATION_LABELS


class CurrencyDataset(Dataset):
    """
    Loads images from:
        dataset_root/
            fake/{10,20,50,100,200,500,2000}/  *.jpg|*.png
            real/{10,20,50,100,200,500,2000}/  *.jpg|*.png

    Returns (image_tensor, label, denomination_idx)
        label: 0 = fake, 1 = real
        denomination_idx: index into DENOMINATION_LABELS
    """

    def __init__(
        self,
        dataset_root: str | Path,
        transform: Optional[transforms.Compose] = None,
        split: str = "all",  # "all", "train", "val"
        val_ratio: float = 0.15,
        seed: int = 42,
    ):
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.transform = transform or self._default_transform()
        self.samples: list[tuple[Path, int, int]] = []  # (path, label, denom_idx)

        # Scan directory
        for label_name, label_id in [("fake", 0), ("real", 1)]:
            label_dir = self.dataset_root / label_name
            if not label_dir.exists():
                continue
            for denom in DENOMINATION_LABELS:
                denom_dir = label_dir / denom
                if not denom_dir.exists():
                    continue
                denom_idx = DENOMINATION_LABELS.index(denom)
                for img_path in sorted(denom_dir.iterdir()):
                    if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                        self.samples.append((img_path, label_id, denom_idx))

        # Train / val split
        if split in ("train", "val"):
            import random
            rng = random.Random(seed)
            indices = list(range(len(self.samples)))
            rng.shuffle(indices)
            n_val = int(len(indices) * val_ratio)
            if split == "val":
                indices = indices[:n_val]
            else:
                indices = indices[n_val:]
            self.samples = [self.samples[i] for i in indices]

    @staticmethod
    def _default_transform() -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @staticmethod
    def train_transform() -> transforms.Compose:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @staticmethod
    def pretrain_transform() -> transforms.Compose:
        """Transform for self-supervised pretraining (stronger augmentation)."""
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        path, label, denom_idx = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Return a blank image on error
            img = Image.new("RGB", (224, 224), (128, 128, 128))

        if self.transform:
            img = self.transform(img)

        return img, label, denom_idx

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for imbalanced dataset."""
        labels = [s[1] for s in self.samples]
        from collections import Counter
        counts = Counter(labels)
        total = len(labels)
        weights = torch.tensor([
            total / (len(counts) * counts.get(i, 1))
            for i in range(max(counts.keys()) + 1)
        ], dtype=torch.float32)
        return weights

    def get_sampler(self) -> WeightedRandomSampler:
        """Weighted sampler to handle class imbalance during training."""
        labels = [s[1] for s in self.samples]
        class_weights = self.get_class_weights()
        sample_weights = [class_weights[l] for l in labels]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )


def create_dataloaders(
    dataset_root: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    mode: str = "finetune",   # "pretrain" or "finetune"
) -> tuple[DataLoader, DataLoader]:
    """Create train + val dataloaders."""
    if mode == "pretrain":
        train_tf = CurrencyDataset.pretrain_transform()
    else:
        train_tf = CurrencyDataset.train_transform()

    val_tf = CurrencyDataset._default_transform()

    train_ds = CurrencyDataset(dataset_root, transform=train_tf, split="train")
    val_ds = CurrencyDataset(dataset_root, transform=val_tf, split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_ds.get_sampler(),
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader
