from datetime import datetime, timezone

import pytest
import torch

from src.models.artifact import ModelArtifactMetadata, NormalizationMetadata
from src.models.bundle import save_model_bundle, validate_model_bundle
from src.models.simple_cnn import SimpleCNN


def _metadata() -> ModelArtifactMetadata:
    return ModelArtifactMetadata(
        model_type="simple_cnn",
        model_version="test",
        class_labels=("cat", "dog"),
        input_size=224,
        color_mode="RGB",
        normalization=NormalizationMetadata(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def test_model_bundle_round_trip_and_checksum(tmp_path):
    save_model_bundle(tmp_path, SimpleCNN(), _metadata())
    validate_model_bundle(tmp_path)
    state = torch.load(tmp_path / "model.pt", weights_only=True)["state_dict"]
    assert "classifier.1.weight" in state


def test_model_bundle_detects_tampering(tmp_path):
    save_model_bundle(tmp_path, SimpleCNN(), _metadata())
    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        validate_model_bundle(tmp_path)
