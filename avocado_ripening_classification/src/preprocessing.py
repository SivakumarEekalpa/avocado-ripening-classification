"""Shared image decoding and preprocessing helpers."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


def read_bgr(path, size=(224, 224), denoise=False):
    """Read an image using Pillow (Unicode-path safe) and return resized BGR pixels."""
    try:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise ValueError(f"Unreadable or unsupported image: {path}") from exc

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if denoise:
        bgr = cv2.fastNlMeansDenoisingColored(bgr, None, 3, 3, 7, 21)
    return cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)


def to_rgb_float(path, size=(224, 224)):
    image = read_bgr(path, size=size)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def imagenet_tensor(path, size=224):
    import torch
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    try:
        with Image.open(Path(path)) as image:
            rgb_image = image.convert("RGB")
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise ValueError(f"Unreadable or unsupported image: {path}") from exc
    return transform(rgb_image)
