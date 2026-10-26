"""Dataset discovery, validation, duplicate filtering, and stratified splitting."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

from .utils import CLASS_NAMES, list_images

logger = logging.getLogger(__name__)

CLASS_ALIASES = {
    "underripe": "underripe",
    "hard": "underripe",
    "breaking": "breaking",
    "firm": "breaking",
    "ripe": "ripe",
    "soft": "ripe",
    "overripe": "overripe",
    "decay": "overripe",
    "decayed": "overripe",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_readable_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
    except (OSError, ValueError, UnidentifiedImageError):
        return False
    return True


def discover_dataset(
    data_dir: str | Path,
    *,
    validate_images: bool = True,
    require_all_classes: bool = True,
) -> pd.DataFrame:
    """Return one row per unique image with canonical labels and SHA-256 hashes.

    Invalid images are skipped when ``validate_images`` is true. Byte-identical
    images are deduplicated; a duplicate assigned conflicting labels is an error.
    Set validation and class completeness off in EDA to inspect incomplete or
    corrupted datasets instead of stopping early.
    """
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    rows: list[dict[str, str]] = []
    seen_hashes: dict[str, tuple[str, str]] = {}
    invalid_count = 0
    duplicate_count = 0

    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        normalized_name = (
            folder.name.lower().replace(" ", "").replace("-", "").replace("_", "")
        )
        label = CLASS_ALIASES.get(normalized_name)
        if label is None:
            logger.warning("Ignoring unrecognized class directory: %s", folder)
            continue

        for image_path in list_images(folder):
            if validate_images and not _is_readable_image(image_path):
                logger.warning("Skipping unreadable image: %s", image_path)
                invalid_count += 1
                continue

            image_hash = _file_sha256(image_path)
            previous = seen_hashes.get(image_hash)
            if previous is not None:
                previous_label, previous_path = previous
                if previous_label != label:
                    raise ValueError(
                        "Identical image content has conflicting labels: "
                        f"{previous_path} ({previous_label}) and "
                        f"{image_path} ({label})"
                    )
                duplicate_count += 1
                continue

            seen_hashes[image_hash] = (label, str(image_path.resolve()))
            rows.append(
                {
                    "path": str(image_path.resolve()),
                    "label": label,
                    "image_sha256": image_hash,
                }
            )

    if invalid_count:
        logger.warning("Skipped %d unreadable image(s) under %s", invalid_count, root)
    if duplicate_count:
        logger.warning("Removed %d byte-identical duplicate image(s)", duplicate_count)

    frame = pd.DataFrame(rows, columns=["path", "label", "image_sha256"])
    if frame.empty:
        raise ValueError(
            f"No supported images found under {root}. Expected class folders: "
            f"{', '.join(CLASS_NAMES)}"
        )

    missing_classes = sorted(set(CLASS_NAMES) - set(frame["label"]))
    if require_all_classes and missing_classes:
        raise ValueError(
            "Dataset is missing required classes: "
            f"{', '.join(missing_classes)}. Found: "
            f"{frame['label'].value_counts().sort_index().to_dict()}"
        )
    return frame.reset_index(drop=True)


def stratified_split(
    df: pd.DataFrame, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return stratified train/validation/test frames in approximately 70/15/15."""
    if df.empty:
        raise ValueError("Cannot split an empty dataset.")
    if "label" not in df:
        raise ValueError("Dataset frame must contain a 'label' column.")

    class_counts = df["label"].value_counts()
    missing_classes = sorted(set(CLASS_NAMES) - set(class_counts.index))
    if missing_classes:
        raise ValueError(
            f"Cannot split dataset; missing required classes: {', '.join(missing_classes)}"
        )
    too_small = class_counts[class_counts < 6]
    if not too_small.empty:
        counts = ", ".join(f"{name}={count}" for name, count in too_small.items())
        raise ValueError(
            "A stratified 70/15/15 split requires at least 6 unique images per "
            f"class; found {counts}. Collect more images or document a different split."
        )

    train, temporary = train_test_split(
        df,
        test_size=0.30,
        random_state=seed,
        stratify=df["label"],
    )
    validation, test = train_test_split(
        temporary,
        test_size=0.50,
        random_state=seed,
        stratify=temporary["label"],
    )
    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def save_split_manifest(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    output: str | Path = "data/processed/split_manifest.csv",
) -> pd.DataFrame:
    """Save the exact split membership used for a training run."""
    manifest = pd.concat(
        [
            train.assign(split="train"),
            validation.assign(split="validation"),
            test.assign(split="test"),
        ],
        ignore_index=True,
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)
    return manifest
