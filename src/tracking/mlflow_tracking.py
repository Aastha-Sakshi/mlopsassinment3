"""Small MLflow adapter used by the training entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import mlflow


def start_run(tracking_uri: str, experiment_name: str, run_name: str | None = None):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name)


def log_parameters(parameters: Mapping[str, object]) -> None:
    mlflow.log_params(dict(parameters))


def log_epoch(epoch: int, metrics: Mapping[str, float]) -> None:
    mlflow.log_metrics(dict(metrics), step=epoch)


def log_artifacts(paths: list[Path]) -> None:
    for path in paths:
        mlflow.log_artifact(str(path))
