"""Deterministic, duplicate-aware dataset split manifests.

The functions here deliberately use only the Python standard library.  Split
membership is a small, reviewable Git-tracked contract; images remain outside
Git and are tracked by DVC.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class ManifestRow:
    relative_path: str
    label: str


def _normalise_label(name: str) -> str:
    return name.casefold()


def _allocate_counts(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    """Allocate a class count using largest remainder, with a stable tie-break."""
    exact = [total * ratio for ratio in ratios]
    counts = [int(value) for value in exact]
    remaining = total - sum(counts)
    order = sorted(range(3), key=lambda index: (-(exact[index] - counts[index]), index))
    for index in order[:remaining]:
        counts[index] += 1
    return tuple(counts)  # type: ignore[return-value]


def resolve_exact_duplicates(
    data_dir: Path, validation_report: dict[str, Any]
) -> tuple[list[ManifestRow], list[dict[str, Any]]]:
    """Return eligible images and an auditable exact-duplicate exclusion list.

    Same-label exact duplicates retain the lexicographically first path.  A
    duplicate group spanning labels is ambiguous, so every member is excluded
    rather than allowing a contradictory label into any split.
    """
    exclusions: list[dict[str, Any]] = []
    excluded_paths: set[str] = set()

    for group in validation_report.get("exact_duplicate_groups", []):
        paths = sorted(str(path).replace("\\", "/") for path in group)
        labels = {_normalise_label(Path(path).parts[0]) for path in paths}
        if len(labels) > 1:
            excluded_paths.update(paths)
            exclusions.append({"reason": "cross_label_exact_duplicate", "paths": paths})
            continue

        canonical = paths[0]
        for duplicate in paths[1:]:
            excluded_paths.add(duplicate)
            exclusions.append(
                {
                    "reason": "same_label_exact_duplicate",
                    "path": duplicate,
                    "canonical_path": canonical,
                }
            )

    rows: list[ManifestRow] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(data_dir).as_posix()
        if relative_path in excluded_paths:
            continue
        parts = Path(relative_path).parts
        if len(parts) != 2:
            raise ValueError(f"Expected <class>/<file> layout, found: {relative_path}")
        rows.append(ManifestRow(relative_path=relative_path, label=_normalise_label(parts[0])))

    return rows, exclusions


def build_stratified_splits(
    rows: list[ManifestRow], seed: int, ratios: tuple[float, float, float]
) -> dict[str, list[ManifestRow]]:
    if len(ratios) != 3 or any(ratio < 0 for ratio in ratios) or sum(ratios) != 1:
        raise ValueError("Split ratios must be three non-negative values that sum to 1.")

    by_label: dict[str, list[ManifestRow]] = {}
    for row in rows:
        by_label.setdefault(row.label, []).append(row)

    splits = {name: [] for name in SPLIT_NAMES}
    for label in sorted(by_label):
        label_rows = sorted(by_label[label], key=lambda row: row.relative_path)
        random.Random(f"{seed}:{label}").shuffle(label_rows)
        train_count, validation_count, test_count = _allocate_counts(len(label_rows), ratios)
        boundaries = (train_count, train_count + validation_count, train_count + validation_count + test_count)
        splits["train"].extend(label_rows[: boundaries[0]])
        splits["validation"].extend(label_rows[boundaries[0] : boundaries[1]])
        splits["test"].extend(label_rows[boundaries[1] : boundaries[2]])

    return {name: sorted(split, key=lambda row: row.relative_path) for name, split in splits.items()}


def write_manifests(
    output_dir: Path,
    splits: dict[str, list[ManifestRow]],
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in SPLIT_NAMES:
        with (output_dir / f"{split_name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["relative_path", "label"])
            writer.writeheader()
            writer.writerows(
                {"relative_path": row.relative_path, "label": row.label}
                for row in splits[split_name]
            )
    (output_dir / "manifest_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_manifest(path: Path) -> list[ManifestRow]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [ManifestRow(**row) for row in csv.DictReader(handle)]


def validate_manifests(data_dir: Path, manifests_dir: Path) -> dict[str, Any]:
    """Raise a ValueError if manifest membership violates a core invariant."""
    rows_by_split = {name: read_manifest(manifests_dir / f"{name}.csv") for name in SPLIT_NAMES}
    path_sets = {name: {row.relative_path for row in rows} for name, rows in rows_by_split.items()}
    if any(len(path_sets[name]) != len(rows_by_split[name]) for name in SPLIT_NAMES):
        raise ValueError("A manifest contains duplicate paths.")
    if path_sets["train"] & path_sets["validation"] or path_sets["train"] & path_sets["test"] or path_sets["validation"] & path_sets["test"]:
        raise ValueError("Split manifests overlap.")

    missing = sorted(
        row.relative_path
        for rows in rows_by_split.values()
        for row in rows
        if not (data_dir / row.relative_path).is_file()
    )
    if missing:
        raise ValueError(f"Manifest paths missing from raw data: {missing[:5]}")

    return {
        "total_rows": sum(len(rows) for rows in rows_by_split.values()),
        "split_counts": {name: len(rows) for name, rows in rows_by_split.items()},
        "class_counts": {
            name: dict(sorted(Counter(row.label for row in rows).items()))
            for name, rows in rows_by_split.items()
        },
    }
