"""Validate that the baseline MLflow run is locally inspectable and complete."""

from __future__ import annotations

import argparse
import json

from mlflow.tracking import MlflowClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default="sqlite:///mlruns/mlflow.db")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    client = MlflowClient(tracking_uri=args.tracking_uri)
    run = client.get_run(args.run_id)
    required_metrics = {
        "train_loss",
        "train_accuracy",
        "validation_loss",
        "validation_accuracy",
        "accuracy",
        "f1_macro",
        "test_loss",
    }
    missing_metrics = required_metrics - set(run.data.metrics)
    if missing_metrics:
        raise ValueError(f"MLflow run is missing metrics: {sorted(missing_metrics)}")
    required_params = {"dataset_fingerprint_sha256", "processed_dvc_hash", "subset_manifest_hash"}
    missing_params = required_params - set(run.data.params)
    if missing_params:
        raise ValueError(f"MLflow run is missing parameters: {sorted(missing_params)}")
    artifact_paths = sorted(item.path for item in client.list_artifacts(args.run_id))
    required_artifacts = {"confusion_matrix.png", "loss_curve.png", "run_summary.json", "model_bundle", "dataset"}
    if not required_artifacts.issubset(artifact_paths):
        raise ValueError(f"MLflow run artifact roots are incomplete: {artifact_paths}")
    print(
        json.dumps(
            {
                "run_id": run.info.run_id,
                "status": run.info.status,
                "artifact_uri": run.info.artifact_uri,
                "metrics": run.data.metrics,
                "traceability_params": {key: run.data.params[key] for key in sorted(required_params)},
                "artifact_roots": artifact_paths,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
