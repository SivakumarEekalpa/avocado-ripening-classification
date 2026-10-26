import numpy as np
import pytest
import torch
from PIL import Image

from src.preprocessing import imagenet_tensor, read_bgr, to_rgb_float


def test_read_bgr_resizes_and_accepts_unicode_paths(tmp_path):
    image_path = tmp_path / "avocado_ñ.png"
    Image.new("RGB", (19, 13), (10, 80, 180)).save(image_path)

    image = read_bgr(image_path, size=(32, 24))
    float_image = to_rgb_float(image_path, size=(32, 24))

    assert image.shape == (24, 32, 3)
    assert float_image.shape == (24, 32, 3)
    assert float_image.dtype == np.float32
    assert 0 <= float_image.min() <= float_image.max() <= 1


def test_read_bgr_reports_corrupt_image(tmp_path):
    image_path = tmp_path / "broken.png"
    image_path.write_bytes(b"invalid image bytes")

    with pytest.raises(ValueError, match="Unreadable or unsupported image"):
        read_bgr(image_path)


def test_imagenet_tensor_returns_normalized_rgb_tensor(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (16, 12), (255, 128, 0)).save(image_path)

    image = imagenet_tensor(image_path, size=32)

    assert image.shape == (3, 32, 32)
    assert image.dtype == torch.float32
