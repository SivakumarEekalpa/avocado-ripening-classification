import json

import numpy as np
from PIL import Image

from src.classical_model import train_classical
from src.dataset import discover_dataset, stratified_split
from src.deep_learning_model import train_mobilenet, load_mobilenet
from main_demo import predict_classical, predict_dl, save_prediction_summary


CLASS_NAMES = ["underripe", "breaking", "ripe", "overripe"]


def _make_synthetic_dataset(root):
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        for image_index in range(8):
            image = np.zeros((48, 48, 3), dtype=np.uint8)
            image[:, :, 0] = 30 + class_index * 45 + image_index
            image[:, :, 1] = 170 - class_index * 30 + image_index * 2
            image[:, :, 2] = 20 + class_index * 35 + image_index * 3
            image[image_index : image_index + 4, :, 1] = 40 + class_index * 35
            Image.fromarray(image).save(class_dir / f"sample_{image_index}.png")


def test_both_training_approaches_write_evaluable_artifacts(tmp_path):
    dataset_dir = tmp_path / "dataset"
    _make_synthetic_dataset(dataset_dir)
    dataset = discover_dataset(dataset_dir)
    train_df, validation_df, test_df = stratified_split(dataset)

    classical = train_classical(
        train_df,
        validation_df,
        test_df,
        output_dir=tmp_path / "models",
        metrics_dir=tmp_path / "outputs" / "metrics",
        cv_splits=2,
    )
    _, deep_metrics = train_mobilenet(
        train_df,
        validation_df,
        test_df,
        model_path=tmp_path / "models" / "mobilenetv2.pth",
        metrics_path=tmp_path / "outputs" / "metrics" / "mobilenetv2_metrics.json",
        epochs=1,
        batch_size=8,
        patience=1,
        pretrained=False,
    )
    loaded_model, saved_class_names, _ = load_mobilenet(
        tmp_path / "models" / "mobilenetv2.pth",
        device="cpu",
    )
    sample_image = test_df.iloc[0].path
    classical_predictions = {
        name: predict_classical(tmp_path / "models" / f"{name}_model.joblib", sample_image)
        for name in ("svm", "rf")
    }
    deep_prediction, deep_confidence, deep_latency = predict_dl(
        tmp_path / "models" / "mobilenetv2.pth",
        sample_image,
    )
    prediction_path = save_prediction_summary(
        sample_image,
        {
            "svm": {
                "predicted_class": classical_predictions["svm"][0],
                "confidence": classical_predictions["svm"][1],
            },
            "rf": {
                "predicted_class": classical_predictions["rf"][0],
                "confidence": classical_predictions["rf"][1],
            },
            "mobilenetv2": {
                "predicted_class": deep_prediction,
                "confidence": deep_confidence,
                "inference_ms": deep_latency,
            },
        },
        tmp_path / "outputs" / "predictions" / "sample.json",
    )

    assert set(classical.model) == {"svm", "rf"}
    assert all(classical.validation_macro_f1.between(0, 1))
    assert deep_metrics["model"] == "mobilenetv2"
    assert 0 <= deep_metrics["validation_macro_f1"] <= 1
    assert deep_metrics["inference_ms_per_image"] > 0
    assert (tmp_path / "outputs" / "confusion_matrices" / "svm_confusion_matrix.png").is_file()
    assert (tmp_path / "outputs" / "confusion_matrices" / "rf_confusion_matrix.png").is_file()
    assert (tmp_path / "outputs" / "confusion_matrices" / "mobilenetv2_confusion_matrix.png").is_file()
    assert saved_class_names == sorted(CLASS_NAMES)
    assert not loaded_model.training
    assert all(prediction in CLASS_NAMES for prediction, _ in classical_predictions.values())
    assert all(
        0 <= confidence <= 1
        for _, confidence in classical_predictions.values()
    )
    assert deep_prediction in CLASS_NAMES
    assert 0 <= deep_confidence <= 1
    assert deep_latency > 0
    saved_predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    assert set(saved_predictions["predictions"]) == {"svm", "rf", "mobilenetv2"}
