"""Validate production model checksums, metadata, and loadability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.models.bundle import load_model_bundle, validate_model_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=Path("models/production"))
    args = parser.parse_args()
    validate_model_bundle(args.bundle_dir)
    loaded = load_model_bundle(args.bundle_dir)
    print(
        json.dumps(
            {
                "model_type": loaded.metadata.model_type,
                "model_version": loaded.metadata.model_version,
                "dataset_fingerprint": loaded.metadata.dataset_fingerprint,
                "processed_dvc_hash": loaded.metadata.processed_dvc_hash,
                "subset_manifest_hash": loaded.metadata.subset_manifest_hash,
                "mlflow_run_id": loaded.metadata.mlflow_run_id,
                "evaluation_summary": loaded.metadata.evaluation_summary,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
