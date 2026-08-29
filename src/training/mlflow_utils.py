"""Local MLflow setup and logging helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from src.config import ProjectConfig


@dataclass(frozen=True)
class MlflowContext:
    tracking_uri: str
    experiment_id: str


def configure_mlflow(config: ProjectConfig) -> MlflowContext:
    tracking_uri = config.mlflow.tracking_uri
    artifact_location = config.mlflow.artifact_location.resolve()
    artifact_location.mkdir(parents=True, exist_ok=True)
    if tracking_uri.startswith("sqlite:///"):
        sqlite_path = Path(tracking_uri.removeprefix("sqlite:///"))
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(config.mlflow.experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            name=config.mlflow.experiment_name,
            artifact_location=artifact_location.as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(config.mlflow.experiment_name)
    return MlflowContext(tracking_uri=tracking_uri, experiment_id=experiment_id)


def build_param_payload(config: ProjectConfig, dataset_counts: dict[str, int], device_name: str) -> dict[str, Any]:
    return {
        "project_name": config.name,
        "model_type": config.training.model_type,
        "random_seed": config.random_seed,
        "batch_size": config.training.batch_size,
        "epochs": config.training.epochs,
        "learning_rate": config.training.learning_rate,
        "num_workers": config.training.num_workers,
        "dataset_source": config.dataset.source,
        "dataset_version": config.dataset.version,
        "train_examples": dataset_counts["train"],
        "validation_examples": dataset_counts["validation"],
        "test_examples": dataset_counts["test"],
        "device": device_name,
    }
