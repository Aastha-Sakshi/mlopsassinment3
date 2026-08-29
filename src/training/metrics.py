"""Metrics and plots for baseline training runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


matplotlib.use("Agg")


def build_evaluation_summary(
    true_labels: list[int],
    predicted_labels: list[int],
    class_labels: tuple[str, str],
) -> tuple[dict[str, float], np.ndarray]:
    matrix = confusion_matrix(true_labels, predicted_labels, labels=list(range(len(class_labels))))
    summary: dict[str, float] = {
        "accuracy": float(np.mean(np.asarray(true_labels) == np.asarray(predicted_labels))),
        "precision_macro": float(
            precision_score(true_labels, predicted_labels, average="macro", zero_division=0)
        ),
        "recall_macro": float(recall_score(true_labels, predicted_labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(true_labels, predicted_labels, average="macro", zero_division=0)),
    }
    per_class_precision = precision_score(true_labels, predicted_labels, average=None, zero_division=0)
    per_class_recall = recall_score(true_labels, predicted_labels, average=None, zero_division=0)
    per_class_f1 = f1_score(true_labels, predicted_labels, average=None, zero_division=0)
    for index, label in enumerate(class_labels):
        summary[f"precision_{label}"] = float(per_class_precision[index])
        summary[f"recall_{label}"] = float(per_class_recall[index])
        summary[f"f1_{label}"] = float(per_class_f1[index])
    return summary, matrix


def save_loss_curve(train_losses: list[float], validation_losses: list[float], output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    epochs = list(range(1, len(train_losses) + 1))
    plt.plot(epochs, train_losses, label="train_loss")
    plt.plot(epochs, validation_losses, label="validation_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_confusion_matrix(
    matrix: np.ndarray | list[list[int]],
    class_labels: tuple[str, str],
    output_path: Path,
) -> None:
    matrix = np.asarray(matrix)
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = list(range(len(class_labels)))
    plt.xticks(ticks, class_labels)
    plt.yticks(ticks, class_labels)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            plt.text(column_index, row_index, str(matrix[row_index, column_index]), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
