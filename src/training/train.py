"""Train the baseline Cats vs Dogs model from the processed Phase 2C dataset."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch import nn

from src.config import ProjectConfig, load_config
from src.models.artifact import ModelArtifactMetadata, NormalizationMetadata
from src.models.bundle import save_model_bundle
from src.models.factory import build_model
from src.training.datasets import build_dataloaders, build_datasets, summarize_datasets
from src.training.metrics import build_evaluation_summary, save_confusion_matrix, save_loss_curve
from src.training.mlflow_utils import build_param_payload, configure_mlflow
from src.training.runner import evaluate_model, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--disable-mlflow", action="store_true")
    parser.add_argument("--promote-to-production", action="store_true")
    parser.add_argument("--execution-environment", default="local")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    processed_dir = args.processed_dir or config.dataset.processed_dir
    ensure_processed_dataset_ready(processed_dir)
    set_random_seed(config.random_seed)

    datasets = build_datasets(processed_dir, config)
    dataloaders = build_dataloaders(datasets, config)
    dataset_summary = summarize_datasets(datasets)
    subset_metadata = load_subset_metadata(config.dataset.training_manifests_dir)
    validate_baseline_counts(dataset_summary.split_counts, dataset_summary.class_counts)
    device = resolve_device(args.device)
    model = build_model(config.training.model_type, num_classes=len(config.image.class_labels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    criterion = nn.CrossEntropyLoss()

    run_timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"{config.training.model_type}-{run_timestamp}"
    run_dir = config.training.run_output_dir / run_name
    checkpoint_dir = config.training.checkpoint_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mlflow_context = None if args.disable_mlflow else configure_mlflow(config)
    mlflow_run = (
        nullcontext(None)
        if mlflow_context is None
        else mlflow.start_run(
            experiment_id=mlflow_context.experiment_id,
            run_name=run_name,
            tags={"execution_environment": args.execution_environment},
        )
    )

    with mlflow_run as active_run:
        if active_run is not None:
            mlflow.log_params(build_param_payload(config, dataset_summary.split_counts, str(device)))
            mlflow.log_param("dataset_fingerprint_sha256", load_dataset_metadata(processed_dir)["dataset_fingerprint_sha256"])
            mlflow.log_param("processed_dvc_hash", subset_metadata["source_processed_dvc"]["hash"])
            mlflow.log_param("subset_manifest_hash", subset_metadata["subset_manifest_combined_sha256"])
            mlflow.log_param("subset_selection_seed", subset_metadata["selection_seed"])
            mlflow.log_param("source_split_seed", subset_metadata["source_split_seed"])

        train_losses: list[float] = []
        validation_losses: list[float] = []
        best_validation_accuracy = -1.0
        best_state_dict = copy.deepcopy(model.state_dict())

        for epoch_index in range(config.training.epochs):
            train_result = train_one_epoch(model, dataloaders["train"], optimizer, criterion, device)
            validation_result = evaluate_model(model, dataloaders["validation"], criterion, device)
            train_losses.append(train_result.loss)
            validation_losses.append(validation_result.loss)

            if active_run is not None:
                mlflow.log_metrics(
                    {
                        "train_loss": train_result.loss,
                        "train_accuracy": train_result.accuracy,
                        "validation_loss": validation_result.loss,
                        "validation_accuracy": validation_result.accuracy,
                    },
                    step=epoch_index,
                )

            if validation_result.accuracy >= best_validation_accuracy:
                best_validation_accuracy = validation_result.accuracy
                best_state_dict = copy.deepcopy(model.state_dict())
                torch.save(best_state_dict, checkpoint_dir / "best_model_state.pt")

        model.load_state_dict(best_state_dict)
        test_result = evaluate_model(model, dataloaders["test"], criterion, device)
        evaluation_summary, confusion = build_evaluation_summary(
            test_result.true_labels,
            test_result.predicted_labels,
            config.image.class_labels,
        )
        evaluation_summary["test_loss"] = test_result.loss

        loss_curve_path = run_dir / "loss_curve.png"
        confusion_path = run_dir / "confusion_matrix.png"
        run_summary_path = run_dir / "run_summary.json"
        save_loss_curve(train_losses, validation_losses, loss_curve_path)
        save_confusion_matrix(confusion, config.image.class_labels, confusion_path)

        metadata = build_artifact_metadata(
            config=config,
            config_path=args.config,
            dataset_metadata=load_dataset_metadata(processed_dir),
            subset_metadata=subset_metadata,
            evaluation_summary=evaluation_summary,
            mlflow_run_id=None if active_run is None else active_run.info.run_id,
            model_version=run_name,
        )
        bundle_dir = run_dir / "model_bundle"
        save_model_bundle(bundle_dir, model, metadata)
        if args.promote_to_production:
            save_model_bundle(config.artifacts.production_dir, model, metadata)

        summary_payload = {
            "run_name": run_name,
            "processed_dir": processed_dir.as_posix(),
            "device": str(device),
            "execution_environment": args.execution_environment,
            "dataset_counts": dataset_summary.split_counts,
            "class_counts": dataset_summary.class_counts,
            "subset_manifest_hash": subset_metadata["subset_manifest_combined_sha256"],
            "processed_dvc_hash": subset_metadata["source_processed_dvc"]["hash"],
            "evaluation_summary": evaluation_summary,
            "checkpoint_path": (checkpoint_dir / "best_model_state.pt").as_posix(),
            "bundle_dir": bundle_dir.as_posix(),
            "production_bundle_updated": args.promote_to_production,
            "mlflow_enabled": active_run is not None,
            "mlflow_run_id": None if active_run is None else active_run.info.run_id,
        }
        run_summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        if active_run is not None:
            mlflow.log_metrics(evaluation_summary)
            mlflow.log_artifact(str(loss_curve_path))
            mlflow.log_artifact(str(confusion_path))
            mlflow.log_artifact(str(run_summary_path))
            mlflow.log_artifact(str(config.dataset.training_manifests_dir / "subset_metadata.json"), artifact_path="dataset")
            for split_name in ("train", "validation", "test"):
                mlflow.log_artifact(
                    str(config.dataset.training_manifests_dir / f"{split_name}.csv"),
                    artifact_path="dataset/manifests",
                )
            mlflow.log_artifacts(str(bundle_dir), artifact_path="model_bundle")

        print(json.dumps(summary_payload, indent=2, sort_keys=True))


def build_artifact_metadata(
    config: ProjectConfig,
    config_path: Path,
    dataset_metadata: dict[str, object],
    subset_metadata: dict[str, object],
    evaluation_summary: dict[str, float],
    mlflow_run_id: str | None,
    model_version: str,
) -> ModelArtifactMetadata:
    return ModelArtifactMetadata(
        model_type=config.training.model_type,
        model_version=model_version,
        class_labels=config.image.class_labels,
        input_size=config.image.size,
        color_mode=config.image.mode,
        normalization=NormalizationMetadata(
            mean=config.image.normalization.mean,
            std=config.image.normalization.std,
        ),
        config_hash=sha256_file(config_path),
        dataset_fingerprint=str(dataset_metadata["dataset_fingerprint_sha256"]),
        processed_dvc_hash=str(subset_metadata["source_processed_dvc"]["hash"]),  # type: ignore[index]
        subset_manifest_hash=str(subset_metadata["subset_manifest_combined_sha256"]),
        training_seed=config.random_seed,
        framework_version=torch.__version__,
        mlflow_run_id=mlflow_run_id,
        evaluation_summary=evaluation_summary,
        created_at=datetime.now(UTC).isoformat(),
    )


def ensure_processed_dataset_ready(processed_dir: Path) -> None:
    if not processed_dir.is_dir():
        raise FileNotFoundError(f"Processed dataset directory does not exist: {processed_dir}")
    metadata_path = processed_dir / "dataset_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Processed dataset metadata is missing: {metadata_path}")


def load_dataset_metadata(processed_dir: Path) -> dict[str, object]:
    return json.loads((processed_dir / "dataset_metadata.json").read_text(encoding="utf-8"))


def load_subset_metadata(manifests_dir: Path) -> dict[str, object]:
    return json.loads((manifests_dir / "subset_metadata.json").read_text(encoding="utf-8"))


def validate_baseline_counts(
    split_counts: dict[str, int],
    class_counts: dict[str, dict[str, int]],
) -> None:
    expected_splits = {"train": 9984, "validation": 1248, "test": 1248}
    expected_per_class = {"train": 4992, "validation": 624, "test": 624}
    if split_counts != expected_splits:
        raise ValueError(f"Baseline subset split counts are invalid: {split_counts}")
    for split, count in expected_per_class.items():
        if class_counts[split] != {"cat": count, "dog": count}:
            raise ValueError(f"Baseline subset class counts are invalid for {split}: {class_counts[split]}")


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
