import io

import numpy as np
import pytest
from PIL import Image

from src.preprocessing.image_contract import (
    DEFAULT_IMAGE_CONTRACT,
    decode_image,
    image_to_array,
    normalize_array,
    prepare_image,
    prepare_image_array,
)


def test_prepare_image_converts_grayscale_to_rgb_and_resizes():
    grayscale_image = Image.new("L", (80, 40), color=128)

    prepared = prepare_image(grayscale_image)

    assert prepared.mode == "RGB"
    assert prepared.size == (224, 224)


def test_prepare_image_array_returns_hwc_float32_values_between_zero_and_one():
    rgba_image = Image.new("RGBA", (50, 100), color=(10, 20, 30, 100))

    image_array = prepare_image_array(rgba_image)

    assert image_array.shape == (224, 224, 3)
    assert image_array.dtype == np.float32
    assert image_array.min() >= 0.0
    assert image_array.max() <= 1.0


def test_decode_image_reads_representative_png_bytes():
    image = Image.new("RGB", (12, 12), color=(20, 30, 40))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    decoded = decode_image(buffer.getvalue())

    assert decoded.mode == "RGB"
    assert decoded.size == (12, 12)


def test_decode_image_rejects_non_image_bytes():
    with pytest.raises(ValueError, match="not a readable image"):
        decode_image(b"this is not an image")


def test_normalize_array_uses_one_value_per_rgb_channel():
    image = Image.new("RGB", (224, 224), color=(255, 128, 0))
    image_array = image_to_array(image)

    normalized = normalize_array(image_array, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    assert normalized.shape == (224, 224, 3)
    np.testing.assert_allclose(normalized[0, 0], np.array([1.0, 0.0039216, -1.0]), atol=1e-6)
    assert DEFAULT_IMAGE_CONTRACT.channel_order == "HWC"
