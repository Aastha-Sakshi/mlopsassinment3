"""Create the Git-tracked deterministic 50% baseline subset manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.subset import build_balanced_subset, validate_subset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifests-dir", type=Path, required=True)
    parser.add_argument("--processed-dvc", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-seed", type=int, default=20260829)
    args = parser.parse_args()
    build_balanced_subset(
        args.source_manifests_dir,
        args.processed_dvc,
        args.output_dir,
        args.selection_seed,
    )
    print(json.dumps(validate_subset(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
