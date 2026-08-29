"""Training and evaluation transforms for the processed dataset."""

from __future__ import annotations

from torchvision import transforms

from src.config import ProjectConfig


def build_train_transform(config: ProjectConfig) -> transforms.Compose:
    steps: list[transforms.Transform] = [transforms.Lambda(lambda image: image.convert(config.image.mode))]
    if config.augmentation.enabled and config.augmentation.horizontal_flip:
        steps.append(transforms.RandomHorizontalFlip())
    if config.augmentation.enabled and config.augmentation.rotation_degrees > 0:
        steps.append(transforms.RandomRotation(config.augmentation.rotation_degrees))
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=list(config.image.normalization.mean),
                std=list(config.image.normalization.std),
            ),
        ]
    )
    return transforms.Compose(steps)


def build_eval_transform(config: ProjectConfig) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert(config.image.mode)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=list(config.image.normalization.mean),
                std=list(config.image.normalization.std),
            ),
        ]
    )
