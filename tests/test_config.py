from pathlib import Path

import pytest

from src.config import ConfigurationError, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"


def test_base_config_loads_with_assignment_image_contract():
    config = load_config(BASE_CONFIG)

    assert config.name == "cats-dogs-mlops"
    assert config.image.size == 224
    assert config.image.mode == "RGB"
    assert config.image.class_labels == ("cat", "dog")
    assert config.dataset.split_ratios.train == 0.8
    assert config.dataset.source == "bhavikjikadara/dog-and-cat-classification-dataset"
    assert config.dataset.version == 1
    assert config.dataset.raw_dir.as_posix() == "data/downloads/PetImages"
    assert config.dataset.processed_dir.as_posix() == "data/processed/PetImages224"
    assert config.dataset.training_manifests_dir.as_posix() == "data/manifests/baseline_50"
    assert config.training.num_workers == 0
    assert config.training.checkpoint_dir.as_posix() == "models/checkpoints"
    assert config.training.run_output_dir.as_posix() == "artifacts/training"
    assert config.mlflow.tracking_uri == "sqlite:///mlruns/mlflow.db"
    assert config.mlflow.artifact_location.as_posix() == "mlartifacts"


def test_config_rejects_split_ratios_that_do_not_add_to_one(tmp_path):
    invalid_config = BASE_CONFIG.read_text(encoding="utf-8").replace("test: 0.1", "test: 0.2")
    config_path = tmp_path / "invalid-ratios.yaml"
    config_path.write_text(invalid_config, encoding="utf-8")

    with pytest.raises(ConfigurationError, match="add up to 1.0"):
        load_config(config_path)


def test_config_rejects_non_rgb_image_mode(tmp_path):
    invalid_config = BASE_CONFIG.read_text(encoding="utf-8").replace("mode: RGB", "mode: L")
    config_path = tmp_path / "invalid-mode.yaml"
    config_path.write_text(invalid_config, encoding="utf-8")

    with pytest.raises(ConfigurationError, match="must be RGB"):
        load_config(config_path)
