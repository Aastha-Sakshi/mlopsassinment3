"""Metadata contract for a production model bundle.

No trained model is created in this foundation milestone. This module only
defines the information that must accompany one before it can be served.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class ArtifactMetadataError(ValueError):
    """Raised when model metadata is incomplete or inconsistent."""


@dataclass(frozen=True)
class NormalizationMetadata:
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


@dataclass(frozen=True)
class ModelArtifactMetadata:
    """Metadata required to understand and safely load a model artifact.

    Required fields describe how the model must be loaded and interpreted.
    Optional fields become available after the full training/evaluation pipeline
    exists.
    """

    model_type: str
    model_version: str
    class_labels: tuple[str, str]
    input_size: int
    color_mode: str
    normalization: NormalizationMetadata
    config_hash: str | None = None
    dataset_fingerprint: str | None = None
    processed_dvc_hash: str | None = None
    subset_manifest_hash: str | None = None
    training_seed: int | None = None
    framework_version: str | None = None
    mlflow_run_id: str | None = None
    evaluation_summary: dict[str, float] | None = None
    created_at: str | None = None

    def __post_init__(self) -> None:
        if not self.model_type:
            raise ArtifactMetadataError("model_type is required.")
        if not self.model_version:
            raise ArtifactMetadataError("model_version is required.")
        if len(self.class_labels) != 2 or len(set(self.class_labels)) != 2:
            raise ArtifactMetadataError("class_labels must contain two distinct labels.")
        if self.input_size != 224:
            raise ArtifactMetadataError("input_size must be 224 for this assignment.")
        if self.color_mode != "RGB":
            raise ArtifactMetadataError("color_mode must be RGB for this assignment.")
        if any(value <= 0 for value in self.normalization.std):
            raise ArtifactMetadataError("normalization std values must be greater than zero.")


def metadata_from_dict(raw: dict[str, Any]) -> ModelArtifactMetadata:
    """Validate a decoded metadata mapping and return the typed representation."""
    if not isinstance(raw, dict):
        raise ArtifactMetadataError("Model metadata must be a JSON object.")

    normalization = raw.get("normalization")
    if not isinstance(normalization, dict):
        raise ArtifactMetadataError("normalization must be an object.")

    try:
        mean = _three_numbers(normalization, "mean")
        std = _three_numbers(normalization, "std")
        labels = _two_labels(raw, "class_labels")
        metadata = ModelArtifactMetadata(
            model_type=_required_string(raw, "model_type"),
            model_version=_required_string(raw, "model_version"),
            class_labels=labels,
            input_size=_required_int(raw, "input_size"),
            color_mode=_required_string(raw, "color_mode"),
            normalization=NormalizationMetadata(mean=mean, std=std),
            config_hash=_optional_string(raw, "config_hash"),
            dataset_fingerprint=_optional_string(raw, "dataset_fingerprint"),
            processed_dvc_hash=_optional_string(raw, "processed_dvc_hash"),
            subset_manifest_hash=_optional_string(raw, "subset_manifest_hash"),
            training_seed=_optional_int(raw, "training_seed"),
            framework_version=_optional_string(raw, "framework_version"),
            mlflow_run_id=_optional_string(raw, "mlflow_run_id"),
            evaluation_summary=_optional_metrics(raw, "evaluation_summary"),
            created_at=_optional_string(raw, "created_at"),
        )
    except TypeError as error:
        raise ArtifactMetadataError(str(error)) from error

    return metadata


def load_metadata(path: str | Path) -> ModelArtifactMetadata:
    """Load and validate metadata.json from a future production model bundle."""
    metadata_path = Path(path)
    if not metadata_path.is_file():
        raise ArtifactMetadataError(f"Model metadata file does not exist: {metadata_path}")

    try:
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ArtifactMetadataError(f"Invalid JSON in {metadata_path}: {error}") from error

    return metadata_from_dict(raw)


def metadata_to_dict(metadata: ModelArtifactMetadata) -> dict[str, Any]:
    """Return JSON-ready metadata using lists for the tuple fields."""
    result = asdict(metadata)
    result["class_labels"] = list(metadata.class_labels)
    result["normalization"] = {
        "mean": list(metadata.normalization.mean),
        "std": list(metadata.normalization.std),
    }
    return result


def _required_string(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ArtifactMetadataError(f"{key} must be a non-empty string.")
    return value


def _optional_string(raw: dict[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ArtifactMetadataError(f"{key} must be a non-empty string when provided.")
    return value


def _required_int(raw: dict[str, Any], key: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ArtifactMetadataError(f"{key} must be an integer.")
    return value


def _optional_int(raw: dict[str, Any], key: str) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ArtifactMetadataError(f"{key} must be an integer when provided.")
    return value


def _three_numbers(raw: dict[str, Any], key: str) -> tuple[float, float, float]:
    value = raw.get(key)
    if not isinstance(value, list) or len(value) != 3:
        raise ArtifactMetadataError(f"normalization.{key} must contain three numbers.")
    if any(not isinstance(item, (int, float)) or isinstance(item, bool) for item in value):
        raise ArtifactMetadataError(f"normalization.{key} must contain only numbers.")
    return (float(value[0]), float(value[1]), float(value[2]))


def _two_labels(raw: dict[str, Any], key: str) -> tuple[str, str]:
    value = raw.get(key)
    if not isinstance(value, list) or len(value) != 2:
        raise ArtifactMetadataError(f"{key} must contain two labels.")
    if any(not isinstance(label, str) or not label.strip() for label in value):
        raise ArtifactMetadataError(f"{key} must contain non-empty strings.")
    return (value[0], value[1])


def _optional_metrics(raw: dict[str, Any], key: str) -> dict[str, float] | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ArtifactMetadataError(f"{key} must be an object when provided.")
    if any(not isinstance(metric, str) or not isinstance(score, (int, float)) for metric, score in value.items()):
        raise ArtifactMetadataError(f"{key} must map metric names to numbers.")
    return {metric: float(score) for metric, score in value.items()}
