"""Verify the structural invariants of committed split manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.splits import validate_manifests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifests-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_manifests(args.data_dir, args.manifests_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
