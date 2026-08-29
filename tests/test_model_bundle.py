from __future__ import annotations

import torch

from src.models.artifact import ModelArtifactMetadata, NormalizationMetadata
from src.models.bundle import load_model_bundle, save_model_bundle
from src.models.factory import build_model


def test_model_bundle_round_trip(tmp_path):
    model = build_model("simple_cnn", num_classes=2)
    metadata = ModelArtifactMetadata(
        model_type="simple_cnn",
        model_version="test-version",
        class_labels=("cat", "dog"),
        input_size=224,
        color_mode="RGB",
        normalization=NormalizationMetadata(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        dataset_fingerprint="abc123",
        training_seed=42,
    )

    bundle_dir = tmp_path / "bundle"
    save_model_bundle(bundle_dir, model, metadata)
    loaded = load_model_bundle(bundle_dir)

    assert loaded.metadata.model_type == "simple_cnn"
    assert loaded.metadata.model_version == "test-version"
    outputs = loaded.model(torch.zeros(1, 3, 224, 224))
    assert outputs.shape == (1, 2)


def test_build_model_rejects_unknown_model_type():
    try:
        build_model("unknown", num_classes=2)
    except ValueError as error:
        assert "Unsupported model_type" in str(error)
    else:
        raise AssertionError("Expected ValueError for unsupported model type")
