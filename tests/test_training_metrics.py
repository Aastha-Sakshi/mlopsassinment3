from __future__ import annotations

from pathlib import Path

from src.training.metrics import build_evaluation_summary, save_confusion_matrix, save_loss_curve


def test_build_evaluation_summary_includes_macro_and_per_class_metrics():
    summary, matrix = build_evaluation_summary(
        true_labels=[0, 0, 1, 1],
        predicted_labels=[0, 1, 1, 1],
        class_labels=("cat", "dog"),
    )

    assert matrix.shape == (2, 2)
    assert summary["accuracy"] == 0.75
    assert "precision_cat" in summary
    assert "f1_dog" in summary


def test_metric_plots_are_written(tmp_path: Path):
    loss_curve_path = tmp_path / "loss_curve.png"
    confusion_path = tmp_path / "confusion_matrix.png"

    save_loss_curve([0.9, 0.7], [1.0, 0.8], loss_curve_path)
    save_confusion_matrix(matrix=[[4, 1], [0, 5]], class_labels=("cat", "dog"), output_path=confusion_path)

    assert loss_curve_path.is_file()
    assert confusion_path.is_file()
