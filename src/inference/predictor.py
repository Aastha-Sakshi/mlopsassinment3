"""Load a trained bundle and run deterministic predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from src.config import ProjectConfig
from src.models.bundle import LoadedModelBundle, load_model_bundle
from src.preprocessing.image_contract import decode_image, normalize_array, prepare_image_array


@dataclass(frozen=True)
class PredictionResult:
    label: str
    probabilities: dict[str, float]
    model_version: str


class ModelPredictor:
    def __init__(self, config: ProjectConfig, bundle_dir: Path) -> None:
        self.config = config
        self.bundle_dir = bundle_dir
        self.loaded_bundle: LoadedModelBundle | None = None

    def is_ready(self) -> bool:
        return self.loaded_bundle is not None

    def load(self) -> None:
        self.loaded_bundle = load_model_bundle(self.bundle_dir)

    def predict_bytes(self, image_bytes: bytes) -> PredictionResult:
        if self.loaded_bundle is None:
            raise RuntimeError("Model bundle is not loaded.")

        metadata = self.loaded_bundle.metadata
        image = decode_image(image_bytes)
        image_array = prepare_image_array(image)
        normalized = normalize_array(
            image_array,
            mean=metadata.normalization.mean,
            std=metadata.normalization.std,
        )
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            logits = self.loaded_bundle.model(tensor)
            probabilities = F.softmax(logits, dim=1).squeeze(0)
        label_index = int(probabilities.argmax().item())
        labels = list(metadata.class_labels)
        return PredictionResult(
            label=labels[label_index],
            probabilities={label: round(float(probabilities[index]), 6) for index, label in enumerate(labels)},
            model_version=metadata.model_version,
        )
