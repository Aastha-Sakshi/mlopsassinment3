"""Datasets and dataloaders backed by the processed Phase 2C output."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.config import ProjectConfig
from src.data.contracts import SplitName
from src.data.splits import read_manifest
from src.training.transforms import build_eval_transform, build_train_transform


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class DatasetSummary:
    split_counts: dict[str, int]
    class_counts: dict[str, dict[str, int]]


class ProcessedSplitDataset(Dataset[tuple[Tensor, int]]):
    """Read one processed split from `split/label/*.jpg`."""

    def __init__(self, split_root: Path, class_labels: tuple[str, str], transform, manifest_path: Path) -> None:
        if not split_root.is_dir():
            raise FileNotFoundError(f"Processed split directory does not exist: {split_root}")

        self.transform = transform
        self.class_labels = tuple(label.casefold() for label in class_labels)
        self.class_to_index = {label: index for index, label in enumerate(self.class_labels)}
        self.samples: list[tuple[Path, int]] = []

        for row in read_manifest(manifest_path):
            label = row.label.casefold()
            if label not in self.class_to_index:
                raise ValueError(f"Unknown class label in {manifest_path}: {row.label}")
            image_path = split_root / label / Path(row.relative_path).name
            if not image_path.is_file():
                raise FileNotFoundError(f"Manifest-selected processed image is missing: {image_path}")
            self.samples.append((image_path, self.class_to_index[label]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        image_path, label_index = self.samples[index]
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, label_index


def build_datasets(processed_root: Path, config: ProjectConfig) -> dict[SplitName, ProcessedSplitDataset]:
    return {
        "train": ProcessedSplitDataset(
            split_root=processed_root / "train",
            class_labels=config.image.class_labels,
            transform=build_train_transform(config),
            manifest_path=config.dataset.training_manifests_dir / "train.csv",
        ),
        "validation": ProcessedSplitDataset(
            split_root=processed_root / "validation",
            class_labels=config.image.class_labels,
            transform=build_eval_transform(config),
            manifest_path=config.dataset.training_manifests_dir / "validation.csv",
        ),
        "test": ProcessedSplitDataset(
            split_root=processed_root / "test",
            class_labels=config.image.class_labels,
            transform=build_eval_transform(config),
            manifest_path=config.dataset.training_manifests_dir / "test.csv",
        ),
    }


def build_dataloaders(
    datasets: dict[SplitName, ProcessedSplitDataset],
    config: ProjectConfig,
) -> dict[SplitName, DataLoader[tuple[Tensor, int]]]:
    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.training.num_workers,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
        ),
    }


def summarize_datasets(datasets: dict[SplitName, ProcessedSplitDataset]) -> DatasetSummary:
    return DatasetSummary(
        split_counts={split_name: len(dataset) for split_name, dataset in datasets.items()},
        class_counts={
            split_name: dict(
                sorted(Counter(dataset.class_labels[label_index] for _, label_index in dataset.samples).items())
            )
            for split_name, dataset in datasets.items()
        },
    )
