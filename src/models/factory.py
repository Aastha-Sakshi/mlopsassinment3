"""Small model factory used by training and inference."""

from __future__ import annotations

from torch import nn

from src.models.simple_cnn import SimpleCNN


def build_model(model_type: str, num_classes: int) -> nn.Module:
    if model_type == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)
    raise ValueError(f"Unsupported model_type: {model_type}")
