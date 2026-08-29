"""Canonical deterministic image preparation for later training and inference."""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, UnidentifiedImageError


@dataclass(frozen=True)
class ImageContract:
    """The fixed image format required by this assignment."""

    size: int = 224
    mode: str = "RGB"
    channel_order: str = "HWC"

    def __post_init__(self) -> None:
        if self.size != 224:
            raise ValueError("This assignment requires images resized to 224 x 224.")
        if self.mode != "RGB":
            raise ValueError("This assignment requires RGB images.")
        if self.channel_order != "HWC":
            raise ValueError("Prepared NumPy arrays use HWC channel ordering.")


DEFAULT_IMAGE_CONTRACT = ImageContract()


def decode_image(image_bytes: bytes) -> Image.Image:
    """Decode bytes into an in-memory image with a clear error for bad input."""
    if not image_bytes:
        raise ValueError("Image bytes are empty.")

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()
            return image.copy()
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("Input is not a readable image.") from error


def prepare_image(image: Image.Image, contract: ImageContract = DEFAULT_IMAGE_CONTRACT) -> Image.Image:
    """Convert an image to RGB and resize it deterministically to the contract size."""
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image instance.")

    rgb_image = image.convert(contract.mode)
    return rgb_image.resize((contract.size, contract.size), Image.Resampling.BILINEAR)


def image_to_array(image: Image.Image) -> np.ndarray:
    """Convert a prepared image to a float32 HWC array with values from 0 to 1."""
    array = np.asarray(image, dtype=np.float32) / 255.0
    expected_shape = (image.height, image.width, 3)
    if array.shape != expected_shape:
        raise ValueError(f"Expected an HWC RGB array with shape {expected_shape}, got {array.shape}.")
    return array


def normalize_array(
    image_array: np.ndarray,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> np.ndarray:
    """Apply deterministic per-channel normalization to an HWC float array."""
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("image_array must have HWC shape with exactly three channels.")
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("mean and std must each contain three channel values.")
    if any(value <= 0 for value in std):
        raise ValueError("std values must be greater than zero.")

    mean_array = np.asarray(mean, dtype=np.float32)
    std_array = np.asarray(std, dtype=np.float32)
    return (image_array - mean_array) / std_array


def prepare_image_array(
    image: Image.Image,
    contract: ImageContract = DEFAULT_IMAGE_CONTRACT,
) -> np.ndarray:
    """Prepare an image without augmentation or normalization.

    Training augmentation and tensor conversion are intentionally deferred to
    later code. Validation, test, and inference will share this deterministic
    image-preparation step.
    """
    return image_to_array(prepare_image(image, contract))
