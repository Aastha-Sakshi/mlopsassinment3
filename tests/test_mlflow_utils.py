from __future__ import annotations

from pathlib import Path

import mlflow

from src.config import load_config
from src.training.mlflow_utils import build_param_payload, configure_mlflow


def test_configure_mlflow_creates_local_experiment(tmp_path: Path):
    config_text = f"""
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
  experiment_name: temp-experiment
  tracking_uri: "sqlite:///{(tmp_path / 'mlruns.db').as_posix()}"
  artifact_location: "{(tmp_path / 'artifacts').as_posix()}"
runtime:
  host: 0.0.0.0
  port: 8000
  model_bundle_path: models/production
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_text.strip() + "\n", encoding="utf-8")
    config = load_config(config_path)

    context = configure_mlflow(config)
    experiment = mlflow.get_experiment_by_name("temp-experiment")

    assert context.tracking_uri.startswith("sqlite:///")
    assert experiment is not None
    assert experiment.artifact_location is not None
    assert "artifacts" in experiment.artifact_location


def test_build_param_payload_flattens_run_inputs():
    config = load_config("configs/base.yaml")
    payload = build_param_payload(
        config,
        dataset_counts={"train": 10, "validation": 2, "test": 2},
        device_name="cpu",
    )

    assert payload["model_type"] == "simple_cnn"
    assert payload["train_examples"] == 10
    assert payload["device"] == "cpu"
