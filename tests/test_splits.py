from __future__ import annotations

from pathlib import Path

from src.data.splits import (
    ManifestRow,
    build_stratified_splits,
    resolve_exact_duplicates,
    validate_manifests,
    write_manifests,
)


def _touch(root: Path, relative_path: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"image")


def test_duplicate_resolution_excludes_conflicts_and_keeps_one_same_label(tmp_path: Path) -> None:
    for relative_path in ("Cat/a.jpg", "Cat/b.jpg", "Cat/c.jpg", "Dog/a.jpg", "Dog/b.jpg"):
        _touch(tmp_path, relative_path)
    report = {
        "exact_duplicate_groups": [
            ["Cat/a.jpg", "Cat/b.jpg"],
            ["Cat/c.jpg", "Dog/a.jpg"],
        ]
    }

    rows, exclusions = resolve_exact_duplicates(tmp_path, report)

    assert {(row.relative_path, row.label) for row in rows} == {("Cat/a.jpg", "cat"), ("Dog/b.jpg", "dog")}
    assert len(exclusions) == 2


def test_split_generation_is_deterministic_and_non_overlapping(tmp_path: Path) -> None:
    rows = [
        *[ManifestRow(f"Cat/{index}.jpg", "cat") for index in range(10)],
        *[ManifestRow(f"Dog/{index}.jpg", "dog") for index in range(10)],
    ]
    first = build_stratified_splits(rows, seed=42, ratios=(0.8, 0.1, 0.1))
    second = build_stratified_splits(rows, seed=42, ratios=(0.8, 0.1, 0.1))
    assert first == second
    write_manifests(tmp_path, first, {"dataset_fingerprint_sha256": "test"})
    for row in rows:
        _touch(tmp_path / "raw", row.relative_path)
    summary = validate_manifests(tmp_path / "raw", tmp_path)
    assert summary["split_counts"] == {"train": 16, "validation": 2, "test": 2}
