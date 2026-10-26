"""MobileNetV2 transfer learning with deterministic validation and safe loading."""
from pathlib import Path
import copy
import time

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from .evaluation import evaluate_predictions
from .utils import save_json, set_seed


class AvocadoImageDataset(Dataset):
    def __init__(self, frame, class_to_idx, train=False, size=224):
        self.frame = frame.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        operations = [transforms.Resize((size, size))]
        if train:
            operations.extend(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(12),
                    transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
                    transforms.ColorJitter(
                        brightness=0.15,
                        contrast=0.15,
                        saturation=0.10,
                    ),
                ]
            )
        operations.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )
        self.transform = transforms.Compose(operations)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row.path) as image:
            image = image.convert("RGB")
        return self.transform(image), self.class_to_idx[row.label]


def build_model(num_classes=4, pretrained=True, freeze_backbone=True):
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

    input_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.25),
        nn.Linear(input_features, 256),
        nn.LayerNorm(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.25),
        nn.Linear(256, num_classes),
    )
    return model


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _epoch(model, loader, criterion, optimizer, device, training):
    model.train(training)
    if training and all(not parameter.requires_grad for parameter in model.features.parameters()):
        model.features.eval()
    total_loss = torch.zeros((), device=device)
    true_labels = []
    predictions = []
    probability_rows = []
    _synchronize(device)
    started = time.perf_counter()

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.detach() * len(labels)
            if not training:
                true_labels.extend(labels.detach().cpu().numpy())
                predictions.extend(logits.argmax(dim=1).detach().cpu().numpy())
                probability_rows.extend(
                    torch.softmax(logits, dim=1).detach().cpu().numpy()
                )

    _synchronize(device)
    elapsed = time.perf_counter() - started
    mean_loss = (total_loss / len(loader.dataset)).item()
    return (
        mean_loss,
        np.asarray(true_labels),
        np.asarray(predictions),
        np.asarray(probability_rows),
        elapsed,
    )


def train_mobilenet(
    train_df,
    val_df,
    test_df,
    model_path: str | Path = "models/mobilenetv2.pth",
    metrics_path: str | Path = "outputs/metrics/mobilenetv2_metrics.json",
    epochs=15,
    batch_size=32,
    patience=4,
    pretrained=True,
    num_workers=0,
    seed=42,
):
    """Train MobileNetV2; select epochs on validation loss, evaluate test once."""
    for split_name, frame in (
        ("training", train_df),
        ("validation", val_df),
        ("test", test_df),
    ):
        if frame.empty:
            raise ValueError(f"The {split_name} split cannot be empty.")
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if patience < 1:
        raise ValueError("patience must be at least 1.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    set_seed(seed)
    class_names = sorted(train_df.label.unique())
    expected_classes = set(class_names)
    for split_name, frame in (("validation", val_df), ("test", test_df)):
        unexpected = set(frame.label.unique()) - expected_classes
        missing = expected_classes - set(frame.label.unique())
        if unexpected or missing:
            raise ValueError(
                f"{split_name} labels do not match training labels; "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}."
            )

    class_to_index = {label: index for index, label in enumerate(class_names)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        AvocadoImageDataset(train_df, class_to_index, train=True),
        shuffle=True,
        **loader_options,
    )
    validation_loader = DataLoader(
        AvocadoImageDataset(val_df, class_to_index),
        shuffle=False,
        **loader_options,
    )
    test_loader = DataLoader(
        AvocadoImageDataset(test_df, class_to_index),
        shuffle=False,
        **loader_options,
    )

    model = build_model(len(class_names), pretrained=pretrained).to(device)
    class_counts = train_df.label.value_counts()
    class_weights = torch.tensor(
        [
            len(train_df) / (len(class_names) * class_counts[label])
            for label in class_names
        ],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    best_validation_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    history = []
    training_started = time.perf_counter()

    for epoch_number in range(epochs):
        train_loss, _, _, _, _ = _epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            training=True,
        )
        validation_loss, val_true, val_predictions, _, _ = _epoch(
            model,
            validation_loader,
            criterion,
            optimizer,
            device,
            training=False,
        )
        scheduler.step(validation_loss)
        history.append(
            {
                "epoch": epoch_number + 1,
                "train_loss": train_loss,
                "val_loss": validation_loss,
                "val_accuracy": float(accuracy_score(val_true, val_predictions)),
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"Epoch {epoch_number + 1}/{epochs}: "
            f"train_loss={train_loss:.4f} val_loss={validation_loss:.4f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print("Early stopping.")
                break

    training_seconds = time.perf_counter() - training_started
    model.load_state_dict(best_state)
    model_file = Path(model_path)
    metrics_file = Path(metrics_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
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
        model_file,
    )

    _, val_true, val_predictions, val_probabilities, _ = _epoch(
        model,
        validation_loader,
        criterion,
        optimizer,
        device,
        training=False,
    )
    test_loss, test_true, test_predictions, test_probabilities, test_elapsed = _epoch(
        model,
        test_loader,
        criterion,
        optimizer,
        device,
        training=False,
    )
    index_to_class = {index: label for index, label in enumerate(class_names)}
    val_true_names = [index_to_class[index] for index in val_true]
    val_prediction_names = [index_to_class[index] for index in val_predictions]
    test_true_names = [index_to_class[index] for index in test_true]
    test_prediction_names = [index_to_class[index] for index in test_predictions]
    validation_metrics = evaluate_predictions(
        val_true_names,
        val_prediction_names,
        class_names,
        y_proba=val_probabilities,
    )
    test_metrics = evaluate_predictions(
        test_true_names,
        test_prediction_names,
        class_names,
        y_proba=test_probabilities,
        output_prefix=metrics_file.parent.parent / "confusion_matrices" / "mobilenetv2",
    )
    metrics = {
        **test_metrics,
        "model": "mobilenetv2",
        "validation_macro_f1": validation_metrics["f1_macro"],
        "training_seconds": round(training_seconds, 3),
        "inference_ms_per_image": round(test_elapsed * 1000 / len(test_df), 4),
        "model_size_mb": round(model_file.stat().st_size / (1024**2), 4),
        "test_loss": test_loss,
        "device": str(device),
        "class_names": class_names,
        "history": history,
    }
    save_json(metrics, metrics_file)
    return model, metrics


def load_mobilenet(checkpoint_path, device=None):
    """Load a project checkpoint using PyTorch's restricted weights-only unpickler."""
    target_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=target_device,
        weights_only=True,
    )
    class_names = checkpoint["class_names"]
    model = build_model(
        num_classes=len(class_names),
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(target_device).eval()
    return model, class_names, target_device
