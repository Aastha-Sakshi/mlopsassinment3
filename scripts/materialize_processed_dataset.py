"""Materialize the canonical 224x224 RGB dataset from committed manifests."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageFile

# Phase 2A accepted every file as readable, including one JPEG that emitted a
# truncated-read warning.  Preserve that accepted dataset decision here.
ImageFile.LOAD_TRUNCATED_IMAGES = True


def rows(manifests_dir: Path):
    for split in ("train", "validation", "test"):
        with (manifests_dir / f"{split}.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                yield split, row["relative_path"], row["label"]


def materialize_one(source_root: Path, destination_root: Path, item: tuple[str, str, str]) -> None:
    split, relative_path, label = item
    source = source_root / relative_path
    destination = destination_root / split / label / Path(relative_path).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR).save(
            destination, format="JPEG", quality=95, optimize=True
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--manifests-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    items = list(rows(args.manifests_dir))
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(lambda item: materialize_one(args.source_dir, args.output_dir, item), items))
    full_metadata_path = args.manifests_dir / "manifest_metadata.json"
    subset_metadata_path = args.manifests_dir / "subset_metadata.json"
    if full_metadata_path.is_file():
        metadata = json.loads(full_metadata_path.read_text(encoding="utf-8"))
    elif subset_metadata_path.is_file():
        subset_metadata = json.loads(subset_metadata_path.read_text(encoding="utf-8"))
        metadata = {
            "dataset_fingerprint_sha256": subset_metadata["source_dataset_fingerprint_sha256"],
            "subset_manifest_combined_sha256": subset_metadata["subset_manifest_combined_sha256"],
            "source_processed_dvc_hash": subset_metadata["source_processed_dvc"]["hash"],
            "selection_seed": subset_metadata["selection_seed"],
            "split_seed": subset_metadata["source_split_seed"],
        }
    else:
        raise FileNotFoundError("Manifest metadata is missing.")
    metadata.update({"preprocessing": {"size": [224, 224], "mode": "RGB", "format": "JPEG", "resampling": "BILINEAR"}, "count": len(items)})
    (args.output_dir / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Materialized {len(items)} images to {args.output_dir}")


if __name__ == "__main__":
    main()
