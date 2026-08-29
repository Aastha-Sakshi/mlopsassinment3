"""Small records that make dataset processing and split manifests explicit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


SplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class DatasetIdentity:
    """Identifies the source and contents of a dataset used in a run."""

    source: str
    fingerprint: str | None = None


@dataclass(frozen=True)
class ImageSample:
    """One labelled image before it is assigned to a train/validation/test split."""

    image_path: Path
    label: str
    source_id: str


@dataclass(frozen=True)
class SplitRecord:
    """One stable entry in a future split manifest."""

    source_id: str
    relative_path: Path
    label: str
    split: SplitName


@dataclass(frozen=True)
class SplitSummary:
    """Counts recorded after a deterministic split is created."""

    train_count: int
    validation_count: int
    test_count: int
    seed: int
