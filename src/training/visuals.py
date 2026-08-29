"""Dependency-light PNG evidence for MLflow and submission artifacts."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def save_loss_curve(train_loss: list[float], validation_loss: list[float], output: Path) -> None:
    canvas = Image.new("RGB", (800, 500), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 20), "Loss by epoch", fill="black")
    values = train_loss + validation_loss
    maximum = max(values, default=1.0) or 1.0
    for index, series in enumerate((train_loss, validation_loss)):
        color = "#2563eb" if index == 0 else "#dc2626"
        points = []
        for epoch, value in enumerate(series):
            x = 60 + epoch * 700 / max(len(series) - 1, 1)
            y = 450 - value * 380 / maximum
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=color, width=4)
    draw.text((600, 20), "blue=train  red=validation", fill="black")
    canvas.save(output)


def save_confusion_matrix(matrix: list[list[int]], labels: tuple[str, str], output: Path) -> None:
    canvas = Image.new("RGB", (600, 600), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((150, 25), "Confusion matrix (rows=true, columns=predicted)", fill="black")
    maximum = max((value for row in matrix for value in row), default=1) or 1
    for row in range(2):
        for column in range(2):
            value = matrix[row][column]
            shade = 255 - int(180 * value / maximum)
            box = (120 + column * 200, 120 + row * 200, 320 + column * 200, 320 + row * 200)
            draw.rectangle(box, fill=(shade, shade, 255), outline="black", width=2)
            draw.text((205 + column * 200, 205 + row * 200), str(value), fill="black")
    draw.text((180, 540), f"columns: {labels[0]}, {labels[1]}", fill="black")
    draw.text((15, 350), f"rows: {labels[0]}, {labels[1]}", fill="black")
    canvas.save(output)
