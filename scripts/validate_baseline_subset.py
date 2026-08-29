"""Validate deterministic baseline subset membership and recorded hashes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.subset import validate_subset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_subset(args.manifests_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
