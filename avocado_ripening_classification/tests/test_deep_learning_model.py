import pandas as pd
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.deep_learning_model import (
    AvocadoImageDataset,
    build_model,
    load_mobilenet,
    train_mobilenet,
)


def test_mobilenet_head_and_frozen_backbone_support_single_image():
    model = build_model(num_classes=4, pretrained=False, freeze_backbone=True)
    model.train()
    logits = model(torch.zeros(1, 3, 224, 224))

    assert logits.shape == (1, 4)
    assert all(not parameter.requires_grad for parameter in model.features.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())


def test_dataset_converts_input_to_rgb_and_normalized_tensor(tmp_path):
    image_path = tmp_path / "gray.png"
    Image.new("L", (20, 16), 120).save(image_path)
    frame = pd.DataFrame(
        {"path": [str(image_path)], "label": ["ripe"]}
    )
    dataset = AvocadoImageDataset(frame, {"ripe": 0}, train=False, size=32)

    image, label = dataset[0]

    assert image.shape == (3, 32, 32)
    assert label == 0
    assert image.dtype == torch.float32


def test_checkpoint_round_trip_uses_saved_class_order(tmp_path):
    class_names = ["breaking", "overripe", "ripe", "underripe"]
    model = build_model(num_classes=len(class_names), pretrained=False)
    checkpoint_path = tmp_path / "checkpoint.pth"
    torch.save(
        {
            "architecture": "mobilenet_v2",
            "state_dict": model.state_dict(),
            "class_names": class_names,
            "image_size": 224,
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        checkpoint_path,
    )

    loaded_model, loaded_classes, device = load_mobilenet(
        checkpoint_path,
        device="cpu",
    )

    assert loaded_classes == class_names
    assert device.type == "cpu"
    assert not loaded_model.training


def test_training_rejects_bad_parameters_before_model_creation():
    frame = pd.DataFrame(
        {"path": ["unused.png"], "label": ["ripe"]}
    )

    with pytest.raises(ValueError, match="num_workers cannot be negative"):
        train_mobilenet(
            frame,
            frame,
            frame,
            num_workers=-1,
        )
