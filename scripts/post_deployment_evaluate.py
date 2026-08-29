"""Evaluate a small balanced labeled batch against the deployed API and log it to MLflow."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import mlflow
import requests

from src.config import load_config
from src.data.splits import read_manifest
from src.training.mlflow_utils import configure_mlflow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--samples-per-class", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("artifacts/post_deployment/evaluation.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    rows = read_manifest(config.dataset.training_manifests_dir / "test.csv")
    selected = []
    for label in config.image.class_labels:
        selected.extend(sorted((row for row in rows if row.label == label), key=lambda row: row.relative_path)[: args.samples_per_class])

    predictions: list[dict[str, object]] = []
    confusion: Counter[tuple[str, str]] = Counter()
    for row in selected:
        image_path = config.dataset.processed_dir / "test" / row.label / Path(row.relative_path).name
        with image_path.open("rb") as handle:
            response = requests.post(
                f"{args.base_url}/predict",
                files={"file": (image_path.name, handle, "image/jpeg")},
                timeout=30,
            )
        response.raise_for_status()
        payload = response.json()
        predicted = str(payload["label"])
        confusion[(row.label, predicted)] += 1
        predictions.append({"image": row.relative_path, "true_label": row.label, "predicted_label": predicted})

    correct = sum(item["true_label"] == item["predicted_label"] for item in predictions)
    accuracy = correct / len(predictions)
    metrics_response = requests.get(f"{args.base_url}/metrics", timeout=15)
    metrics_response.raise_for_status()
    result = {
        "base_url": args.base_url,
        "sample_count": len(predictions),
        "samples_per_class": args.samples_per_class,
        "accuracy": accuracy,
        "confusion": {f"{truth}->{prediction}": count for (truth, prediction), count in sorted(confusion.items())},
        "service_metrics": metrics_response.json(),
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    configure_mlflow(config)
    mlflow.set_experiment(f"{config.mlflow.experiment_name}-post-deployment")
    with mlflow.start_run(run_name="post-deployment-balanced-batch"):
        mlflow.log_params({"samples_per_class": args.samples_per_class, "base_url": args.base_url})
        mlflow.log_metrics({"post_deployment_accuracy": accuracy, "post_deployment_sample_count": len(predictions)})
        mlflow.log_artifact(str(args.output))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
