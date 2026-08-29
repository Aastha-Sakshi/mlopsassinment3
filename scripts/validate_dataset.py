"""Validate a downloaded image dataset without changing its contents.

This Phase 2A command records the exact files that were inspected, verifies
that Pillow can read them, and writes a small JSON report for later DVC and
training decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, UnidentifiedImageError


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def file_sha256(path: Path) -> str:
    """Return a SHA-256 hash without loading a whole file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(path: Path) -> tuple[str, tuple[int, int]]:
    """Verify an image is readable and return its mode and dimensions."""
    try:
        with Image.open(path) as image:
            image.load()
            return image.mode, image.size
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(str(error)) from error


def validate_dataset(data_dir: Path) -> dict:
    """Inspect images under class directories and return a JSON-ready report."""
    if not data_dir.is_dir():
        raise ValueError(f"Dataset directory does not exist: {data_dir}")

    image_paths = sorted(
        path for path in data_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise ValueError(f"No supported image files found under {data_dir}")

    class_counts: Counter[str] = Counter()
    format_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    invalid_files: list[dict[str, str]] = []
    hashes: dict[str, list[str]] = defaultdict(list)
    fingerprint = hashlib.sha256()

    for image_path in image_paths:
        relative_path = image_path.relative_to(data_dir)
        class_name = relative_path.parts[0] if len(relative_path.parts) > 1 else "UNCLASSIFIED"
        content_hash = file_sha256(image_path)
        fingerprint.update(f"{relative_path.as_posix()}|{image_path.stat().st_size}|{content_hash}\n".encode())
        hashes[content_hash].append(relative_path.as_posix())

        try:
            mode, size = inspect_image(image_path)
        except ValueError as error:
            invalid_files.append({"path": relative_path.as_posix(), "error": str(error)})
            continue

        class_counts[class_name] += 1
        format_counts[image_path.suffix.lower()] += 1
        mode_counts[mode] += 1
        dimensions[f"{size[0]}x{size[1]}"] += 1

    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    return {
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "data_dir": str(data_dir),
        "total_image_files": len(image_paths),
        "valid_image_files": len(image_paths) - len(invalid_files),
        "class_counts": dict(sorted(class_counts.items())),
        "format_counts": dict(sorted(format_counts.items())),
        "mode_counts": dict(sorted(mode_counts.items())),
        "dimension_counts": dict(sorted(dimensions.items())),
        "invalid_files": invalid_files,
        "exact_duplicate_groups": duplicate_groups,
        "exact_duplicate_file_count": sum(len(group) for group in duplicate_groups),
        "dataset_fingerprint_sha256": fingerprint.hexdigest(),
        "notes": [
            "Duplicate detection uses exact file-byte SHA-256 hashes.",
            "Near-duplicate or perceptual duplicate detection is not included in this Phase 2A scan.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an image dataset without modifying it.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = validate_dataset(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Validated {report['valid_image_files']} readable images.")
    print(f"Invalid images: {len(report['invalid_files'])}")
    print(f"Exact duplicate groups: {len(report['exact_duplicate_groups'])}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
