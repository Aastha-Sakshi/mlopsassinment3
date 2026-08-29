"""Small, explicit loader for the project's YAML configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when a configuration file is missing required valid values."""


@dataclass(frozen=True)
class SplitRatios:
    train: float
    validation: float
    test: float


@dataclass(frozen=True)
class NormalizationConfig:
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


@dataclass(frozen=True)
class DatasetConfig:
    source: str
    version: int
    raw_dir: Path
    validation_report: Path
    processed_dir: Path
    manifests_dir: Path
    training_manifests_dir: Path
    split_ratios: SplitRatios


@dataclass(frozen=True)
class ImageConfig:
    size: int
    mode: str
    class_labels: tuple[str, str]
    normalization: NormalizationConfig


@dataclass(frozen=True)
class AugmentationConfig:
    enabled: bool
    horizontal_flip: bool
    rotation_degrees: int


@dataclass(frozen=True)
class TrainingConfig:
    model_type: str
    batch_size: int
    epochs: int
    learning_rate: float
    num_workers: int
    checkpoint_dir: Path
    run_output_dir: Path


@dataclass(frozen=True)
class ArtifactConfig:
    production_dir: Path


@dataclass(frozen=True)
class MlflowConfig:
    experiment_name: str
    tracking_uri: str
    artifact_location: Path


@dataclass(frozen=True)
class RuntimeConfig:
    host: str
    port: int
    model_bundle_path: Path


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    random_seed: int
    dataset: DatasetConfig
    image: ImageConfig
    augmentation: AugmentationConfig
    training: TrainingConfig
    artifacts: ArtifactConfig
    mlflow: MlflowConfig
    runtime: RuntimeConfig


def load_config(path: str | Path) -> ProjectConfig:
    """Load and validate a project configuration from a YAML file."""
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            raw_config = yaml.safe_load(config_file)
    except yaml.YAMLError as error:
        raise ConfigurationError(f"Invalid YAML in {config_path}: {error}") from error

    if not isinstance(raw_config, dict):
        raise ConfigurationError("Configuration root must be a mapping.")

    return _build_project_config(raw_config)


def _build_project_config(raw: dict[str, Any]) -> ProjectConfig:
    project = _mapping(raw, "project")
    dataset = _mapping(raw, "dataset")
    image = _mapping(raw, "image")
    augmentation = _mapping(raw, "augmentation")
    training = _mapping(raw, "training")
    artifacts = _mapping(raw, "artifacts")
    mlflow = _mapping(raw, "mlflow")
    runtime = _mapping(raw, "runtime")
    ratios = _mapping(dataset, "split_ratios")
    normalization = _mapping(image, "normalization")

    split_ratios = SplitRatios(
        train=_positive_float(ratios, "train"),
        validation=_positive_float(ratios, "validation"),
        test=_positive_float(ratios, "test"),
    )
    if abs(sum((split_ratios.train, split_ratios.validation, split_ratios.test)) - 1.0) > 1e-9:
        raise ConfigurationError("dataset.split_ratios must add up to 1.0.")

    image_size = _positive_int(image, "size")
    if image_size != 224:
        raise ConfigurationError("image.size must be 224 for this assignment.")

    image_mode = _string(image, "mode")
    if image_mode != "RGB":
        raise ConfigurationError("image.mode must be RGB for this assignment.")

    class_labels = _labels(image, "class_labels")
    normalization_config = NormalizationConfig(
        mean=_three_floats(normalization, "mean"),
        std=_three_floats(normalization, "std"),
    )
    if any(value <= 0 for value in normalization_config.std):
        raise ConfigurationError("image.normalization.std values must be greater than zero.")

    rotation_degrees = _non_negative_int(augmentation, "rotation_degrees")
    learning_rate = _positive_float(training, "learning_rate")

    return ProjectConfig(
        name=_string(project, "name"),
        random_seed=_non_negative_int(project, "random_seed"),
        dataset=DatasetConfig(
            source=_string(dataset, "source"),
            version=_positive_int(dataset, "version"),
            raw_dir=Path(_string(dataset, "raw_dir")),
            validation_report=Path(_string(dataset, "validation_report")),
            processed_dir=Path(_string(dataset, "processed_dir")),
            manifests_dir=Path(_string(dataset, "manifests_dir")),
            training_manifests_dir=Path(
                _string(dataset, "training_manifests_dir")
                if "training_manifests_dir" in dataset
                else "data/manifests/baseline_50"
            ),
            split_ratios=split_ratios,
        ),
        image=ImageConfig(
            size=image_size,
            mode=image_mode,
            class_labels=class_labels,
            normalization=normalization_config,
        ),
        augmentation=AugmentationConfig(
            enabled=_boolean(augmentation, "enabled"),
            horizontal_flip=_boolean(augmentation, "horizontal_flip"),
            rotation_degrees=rotation_degrees,
        ),
        training=TrainingConfig(
            model_type=_string(training, "model_type"),
            batch_size=_positive_int(training, "batch_size"),
            epochs=_positive_int(training, "epochs"),
            learning_rate=learning_rate,
            num_workers=_non_negative_int(training, "num_workers"),
            checkpoint_dir=Path(_string(training, "checkpoint_dir")),
            run_output_dir=Path(_string(training, "run_output_dir")),
        ),
        artifacts=ArtifactConfig(production_dir=Path(_string(artifacts, "production_dir"))),
        mlflow=MlflowConfig(
            experiment_name=_string(mlflow, "experiment_name"),
            tracking_uri=_string(mlflow, "tracking_uri"),
            artifact_location=Path(_string(mlflow, "artifact_location")),
        ),
        runtime=RuntimeConfig(
            host=_string(runtime, "host"),
            port=_positive_int(runtime, "port"),
            model_bundle_path=Path(_string(runtime, "model_bundle_path")),
        ),
    )


def _mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ConfigurationError(f"{key} must be a mapping.")
    return value


def _string(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{key} must be a non-empty string.")
    return value


def _boolean(raw: dict[str, Any], key: str) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise ConfigurationError(f"{key} must be true or false.")
    return value


def _positive_int(raw: dict[str, Any], key: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigurationError(f"{key} must be a positive integer.")
    return value


def _non_negative_int(raw: dict[str, Any], key: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ConfigurationError(f"{key} must be a non-negative integer.")
    return value


def _positive_float(raw: dict[str, Any], key: str) -> float:
    value = raw.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ConfigurationError(f"{key} must be a positive number.")
    return float(value)


def _three_floats(raw: dict[str, Any], key: str) -> tuple[float, float, float]:
    value = raw.get(key)
    if not isinstance(value, list) or len(value) != 3:
        raise ConfigurationError(f"{key} must contain exactly three numbers.")
    if any(not isinstance(item, (int, float)) or isinstance(item, bool) for item in value):
        raise ConfigurationError(f"{key} must contain only numbers.")
    return (float(value[0]), float(value[1]), float(value[2]))


def _labels(raw: dict[str, Any], key: str) -> tuple[str, str]:
    value = raw.get(key)
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigurationError(f"{key} must contain exactly two class labels.")
    if any(not isinstance(label, str) or not label.strip() for label in value):
        raise ConfigurationError(f"{key} must contain non-empty strings.")
    if value[0] == value[1]:
        raise ConfigurationError(f"{key} must contain distinct class labels.")
    return (value[0], value[1])
