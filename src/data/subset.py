"""Deterministic manifest-only selection for the baseline training subset."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from src.data.splits import SPLIT_NAMES, ManifestRow, read_manifest


ALGORITHM_VERSION = "sha256-ranked-per-existing-split-class-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_balanced_subset(
    source_manifests_dir: Path,
    processed_dvc_path: Path,
    output_dir: Path,
    selection_seed: int = 20260829,
    per_class_quotas: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Select stable, balanced membership without copying image data."""
    quotas = per_class_quotas or {"train": 4992, "validation": 624, "test": 624}
    source_metadata = json.loads(
        (source_manifests_dir / "manifest_metadata.json").read_text(encoding="utf-8")
    )
    dvc_payload = yaml.safe_load(processed_dvc_path.read_text(encoding="utf-8"))
    dvc_output = dvc_payload["outs"][0]
    selected_by_split: dict[str, list[ManifestRow]] = {}

    for split in SPLIT_NAMES:
        rows = read_manifest(source_manifests_dir / f"{split}.csv")
        by_label: dict[str, list[ManifestRow]] = {}
        for row in rows:
            by_label.setdefault(row.label, []).append(row)
        if set(by_label) != {"cat", "dog"}:
            raise ValueError(f"Expected cat and dog labels in {split}, found {sorted(by_label)}")
        selected: list[ManifestRow] = []
        for label in ("cat", "dog"):
            ranked = sorted(
                by_label[label],
                key=lambda row: (
                    hashlib.sha256(
                        f"{selection_seed}\0{split}\0{label}\0{row.relative_path}".encode("utf-8")
                    ).hexdigest(),
                    row.relative_path,
                ),
            )
            quota = quotas[split]
            if len(ranked) < quota:
                raise ValueError(f"Not enough {label} rows in {split}: need {quota}, found {len(ranked)}")
            selected.extend(ranked[:quota])
        selected_by_split[split] = sorted(selected, key=lambda row: row.relative_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_hashes: dict[str, str] = {}
    for split in SPLIT_NAMES:
        path = output_dir / f"{split}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["relative_path", "label"])
            writer.writeheader()
            writer.writerows(
                {"relative_path": row.relative_path, "label": row.label}
                for row in selected_by_split[split]
            )
        manifest_hashes[path.name] = sha256_file(path)

    combined_hash = hashlib.sha256(
        "".join(f"{name}:{manifest_hashes[name]}\n" for name in sorted(manifest_hashes)).encode("utf-8")
    ).hexdigest()
    class_counts = {
        split: dict(sorted(Counter(row.label for row in rows).items()))
        for split, rows in selected_by_split.items()
    }
    metadata = {
        "description": (
            "The full Kaggle dataset is the authoritative DVC-versioned dataset, while a "
            "deterministic balanced 50% subset is used for the assignment's baseline training "
            "experiment to reduce iteration time."
        ),
        "algorithm_version": ALGORITHM_VERSION,
        "selection_seed": selection_seed,
        "source_split_seed": source_metadata["split_seed"],
        "split_ratios": source_metadata["split_ratios"],
        "source_dataset_fingerprint_sha256": source_metadata["dataset_fingerprint_sha256"],
        "source_processed_dvc": {
            "path": Path(*processed_dvc_path.parts[-3:]).as_posix(),
            "hash": dvc_output["md5"],
            "hash_type": dvc_output.get("hash", "md5"),
            "size_bytes": dvc_output["size"],
            "file_count": dvc_output["nfiles"],
        },
        "source_manifest_hashes_sha256": {
            f"{split}.csv": sha256_file(source_manifests_dir / f"{split}.csv")
            for split in SPLIT_NAMES
        },
        "subset_manifest_hashes_sha256": manifest_hashes,
        "subset_manifest_combined_sha256": combined_hash,
        "subset_size": sum(len(rows) for rows in selected_by_split.values()),
        "samples_per_class": 6240,
        "split_counts": {split: len(rows) for split, rows in selected_by_split.items()},
        "class_counts": class_counts,
        "selection_scope": "rank independently within each existing split and class",
        "images_copied": False,
    }
    (output_dir / "subset_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def validate_subset(manifests_dir: Path) -> dict[str, Any]:
    rows = {split: read_manifest(manifests_dir / f"{split}.csv") for split in SPLIT_NAMES}
    paths = {split: {row.relative_path for row in split_rows} for split, split_rows in rows.items()}
    if any(len(paths[split]) != len(rows[split]) for split in SPLIT_NAMES):
        raise ValueError("A subset manifest contains duplicate paths.")
    if any(paths[left] & paths[right] for index, left in enumerate(SPLIT_NAMES) for right in SPLIT_NAMES[index + 1 :]):
        raise ValueError("Subset manifests overlap.")
    counts = {split: Counter(row.label for row in split_rows) for split, split_rows in rows.items()}
    expected = {"train": 4992, "validation": 624, "test": 624}
    for split, per_class in expected.items():
        if counts[split] != Counter({"cat": per_class, "dog": per_class}):
            raise ValueError(f"Unexpected class balance in {split}: {dict(counts[split])}")
    metadata = json.loads((manifests_dir / "subset_metadata.json").read_text(encoding="utf-8"))
    actual_hashes = {f"{split}.csv": sha256_file(manifests_dir / f"{split}.csv") for split in SPLIT_NAMES}
    if actual_hashes != metadata["subset_manifest_hashes_sha256"]:
        raise ValueError("Subset manifest hashes do not match metadata.")
    return {
        "total": sum(len(split_rows) for split_rows in rows.values()),
        "split_counts": {split: len(split_rows) for split, split_rows in rows.items()},
        "class_counts": {split: dict(sorted(count.items())) for split, count in counts.items()},
        "combined_sha256": metadata["subset_manifest_combined_sha256"],
    }
