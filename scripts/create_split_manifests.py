"""Create deterministic, duplicate-aware train/validation/test manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.splits import (
    SPLIT_NAMES,
    build_stratified_splits,
    resolve_exact_duplicates,
    validate_manifests,
    write_manifests,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--dataset-source", required=True)
    parser.add_argument("--dataset-version", required=True)
    args = parser.parse_args()

    report = json.loads(args.validation_report.read_text(encoding="utf-8"))
    rows, exclusions = resolve_exact_duplicates(args.data_dir, report)
    ratios = (args.train_ratio, args.validation_ratio, args.test_ratio)
    splits = build_stratified_splits(rows, args.seed, ratios)
    metadata = {
        "dataset_source": args.dataset_source,
        "dataset_version": args.dataset_version,
        "data_dir": args.data_dir.as_posix(),
        "validation_report": args.validation_report.as_posix(),
        "dataset_fingerprint_sha256": report["dataset_fingerprint_sha256"],
        "split_seed": args.seed,
        "split_ratios": dict(zip(SPLIT_NAMES, ratios, strict=True)),
        "duplicate_policy": {
            "same_label_exact_duplicates": "keep lexicographically first path",
            "cross_label_exact_duplicates": "exclude every path in the group",
            "near_duplicates": "not assessed in Phase 2A",
        },
        "exclusions": exclusions,
    }
    write_manifests(args.output_dir, splits, metadata)
    summary = validate_manifests(args.data_dir, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
