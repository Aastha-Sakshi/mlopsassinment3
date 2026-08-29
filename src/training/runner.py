"""Train and evaluate the baseline model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EpochResult:
    loss: float
    accuracy: float


@dataclass(frozen=True)
class EvaluationResult:
    loss: float
    accuracy: float
    true_labels: list[int]
    predicted_labels: list[int]


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader[tuple[Tensor, int]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> EpochResult:
    model.train()
    total_loss = 0.0
    total_examples = 0
    total_correct = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())

    return EpochResult(
        loss=total_loss / max(total_examples, 1),
        accuracy=total_correct / max(total_examples, 1),
    )


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader[tuple[Tensor, int]],
    criterion: nn.Module,
    device: torch.device,
) -> EvaluationResult:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    total_correct = 0
    true_labels: list[int] = []
    predicted_labels: list[int] = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        total_correct += int((predictions == labels).sum().item())
        true_labels.extend(labels.cpu().tolist())
        predicted_labels.extend(predictions.cpu().tolist())

    return EvaluationResult(
        loss=total_loss / max(total_examples, 1),
        accuracy=total_correct / max(total_examples, 1),
        true_labels=true_labels,
        predicted_labels=predicted_labels,
    )
