from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.api.main import create_app


def _write_config(path: Path, bundle_path: Path) -> None:
    path.write_text(
        f"""
project:
  name: cats-dogs-mlops
  random_seed: 42
dataset:
  source: bhavikjikadara/dog-and-cat-classification-dataset
  version: 1
  raw_dir: data/downloads/PetImages
  validation_report: artifacts/dataset_validation.json
  processed_dir: data/processed/PetImages224
  manifests_dir: data/manifests
  split_ratios:
    train: 0.8
    validation: 0.1
    test: 0.1
image:
  size: 224
  mode: RGB
  class_labels: [cat, dog]
  normalization:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
augmentation:
  enabled: true
  horizontal_flip: true
  rotation_degrees: 15
training:
  model_type: simple_cnn
  batch_size: 16
  epochs: 5
  learning_rate: 0.001
  num_workers: 0
  checkpoint_dir: models/checkpoints
  run_output_dir: artifacts/training
artifacts:
  production_dir: models/production
mlflow:
  experiment_name: cats-vs-dogs
  tracking_uri: "sqlite:///{(bundle_path.parent / 'mlruns.db').as_posix()}"
  artifact_location: "{(bundle_path.parent / 'mlartifacts').as_posix()}"
runtime:
  host: 0.0.0.0
  port: 8000
  model_bundle_path: "{bundle_path.as_posix()}"
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_health_reports_unloaded_model_when_bundle_is_missing(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, tmp_path / "missing_bundle")
    with TestClient(create_app(config_path)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": False, "model_type": None, "model_version": None}


def test_predict_returns_503_without_loaded_bundle(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, tmp_path / "missing_bundle")
    with TestClient(create_app(config_path)) as client:
        response = client.post("/predict", files={"file": ("image.jpg", b"not-an-image", "image/jpeg")})

    assert response.status_code == 503


def test_metrics_start_at_zero(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, tmp_path / "missing_bundle")
    with TestClient(create_app(config_path)) as client:
        response = client.get("/metrics")
    assert response.status_code == 200
    assert response.json() == {
        "prediction_request_count": 0,
        "prediction_error_count": 0,
        "average_prediction_latency_ms": 0.0,
    }
