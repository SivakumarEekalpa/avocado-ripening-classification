from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from src.dataset import discover_dataset, save_split_manifest, stratified_split

CLASS_FOLDERS = {
    "hard": "underripe",
    "firm": "breaking",
    "soft": "ripe",
    "decay": "overripe",
}


def _write_class_images(root: Path, images_per_class=12):
    for class_index, folder_name in enumerate(CLASS_FOLDERS):
        folder = root / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        for image_index in range(images_per_class):
            color = (
                class_index * 40 + image_index,
                image_index * 7,
                class_index * 25 + image_index * 3,
            )
            Image.new("RGB", (12, 10), color).save(folder / f"image_{image_index}.png")


def test_discover_maps_aliases_and_deduplicates_exact_content(tmp_path):
    _write_class_images(tmp_path)
    original = tmp_path / "hard" / "image_0.png"
    (tmp_path / "hard" / "duplicate.png").write_bytes(original.read_bytes())

    dataset = discover_dataset(tmp_path)

    assert len(dataset) == 48
    assert set(dataset.label) == {
        "underripe",
        "breaking",
        "ripe",
        "overripe",
    }
    assert dataset.image_sha256.is_unique
    assert dataset.path.is_unique


def test_corrupt_images_are_skipped_for_training_and_visible_to_eda(tmp_path):
    _write_class_images(tmp_path)
    corrupt_path = tmp_path / "hard" / "corrupt.png"
    corrupt_path.write_bytes(b"not an image")

    training_data = discover_dataset(tmp_path)
    eda_candidates = discover_dataset(
        tmp_path,
        validate_images=False,
        require_all_classes=False,
    )

    assert str(corrupt_path.resolve()) not in set(training_data.path)
    assert str(corrupt_path.resolve()) in set(eda_candidates.path)


def test_identical_content_with_conflicting_labels_is_rejected(tmp_path):
    _write_class_images(tmp_path)
    original = tmp_path / "hard" / "image_0.png"
    conflicting_copy = tmp_path / "firm" / "same_content.png"
    conflicting_copy.write_bytes(original.read_bytes())

    with pytest.raises(ValueError, match="conflicting labels"):
        discover_dataset(tmp_path)


def test_missing_class_is_reported(tmp_path):
    for class_index, folder in enumerate(("hard", "firm", "soft")):
        directory = tmp_path / folder
        directory.mkdir(parents=True)
        Image.new("RGB", (4, 4), (class_index + 1, 2, 3)).save(
            directory / "sample.png"
        )

    with pytest.raises(ValueError, match="missing required classes"):
        discover_dataset(tmp_path)


def test_stratified_split_is_disjoint_and_approximately_70_15_15(tmp_path):
    _write_class_images(tmp_path)
    dataset = discover_dataset(tmp_path)

    train, validation, test = stratified_split(dataset, seed=17)

    paths = [set(frame.path) for frame in (train, validation, test)]
    assert paths[0].isdisjoint(paths[1])
    assert paths[0].isdisjoint(paths[2])
    assert paths[1].isdisjoint(paths[2])
    assert len(train) + len(validation) + len(test) == len(dataset)
    assert abs(len(train) / len(dataset) - 0.70) < 0.04
    assert abs(len(validation) / len(dataset) - 0.15) < 0.04
    assert abs(len(test) / len(dataset) - 0.15) < 0.04
    for split in (train, validation, test):
        assert set(split.label) == set(dataset.label)


def test_stratified_split_rejects_too_few_samples_per_class():
    dataset = pd.DataFrame(
        {
            "path": [f"{label}_{index}" for label in CLASS_FOLDERS.values() for index in range(5)],
            "label": [label for label in CLASS_FOLDERS.values() for _ in range(5)],
        }
    )

    with pytest.raises(ValueError, match="at least 6 unique images per class"):
        stratified_split(dataset)


def test_split_manifest_preserves_membership_and_labels(tmp_path):
    _write_class_images(tmp_path)
    train, validation, test = stratified_split(discover_dataset(tmp_path))
    manifest_path = tmp_path / "processed" / "split_manifest.csv"

    manifest = save_split_manifest(train, validation, test, manifest_path)

    assert manifest_path.is_file()
    assert manifest.split.value_counts().to_dict() == {
        "train": len(train),
        "test": len(test),
        "validation": len(validation),
    }
    assert len(pd.read_csv(manifest_path)) == len(train) + len(validation) + len(test)
