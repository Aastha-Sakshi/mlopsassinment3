"""Data loaders for the canonical processed split directories."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config import ProjectConfig


def build_transform(config: ProjectConfig, training: bool):
    operations = []
    if training and config.augmentation.enabled:
        if config.augmentation.horizontal_flip:
            operations.append(transforms.RandomHorizontalFlip())
        operations.append(transforms.RandomRotation(config.augmentation.rotation_degrees))
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(config.image.normalization.mean, config.image.normalization.std),
        ]
    )
    return transforms.Compose(operations)


def build_loader(
    data_root: Path,
    split: str,
    config: ProjectConfig,
    shuffle: bool,
    num_workers: int = 2,
) -> DataLoader:
    split_dir = data_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Processed split does not exist: {split_dir}")
    dataset = datasets.ImageFolder(split_dir, transform=build_transform(config, training=split == "train"))
    if tuple(dataset.classes) != config.image.class_labels:
        raise ValueError(f"Expected classes {config.image.class_labels}, found {tuple(dataset.classes)}")
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
