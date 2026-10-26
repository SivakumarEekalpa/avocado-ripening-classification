"""Shared classification metrics, confusion matrices, and model comparison."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from .utils import save_json


def evaluate_predictions(
    y_true,
    y_pred,
    labels,
    y_proba=None,
    output_prefix=None,
    metrics_path=None,
):
    """Calculate macro/weighted classification metrics and optionally save a plot."""
    labels = list(labels)
    if not labels:
        raise ValueError("At least one class label is required for evaluation.")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same number of values.")

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }

    if y_proba is not None:
        probabilities = np.asarray(y_proba)
        if probabilities.ndim != 2 or probabilities.shape != (len(y_true), len(labels)):
            raise ValueError(
                "y_proba must have shape (number of samples, number of labels); "
                f"received {probabilities.shape}."
            )
        if set(y_true) == set(labels):
            ordered_labels = sorted(labels)
            ordered_columns = [labels.index(label) for label in ordered_labels]
            metrics["roc_auc_ovr_macro"] = float(
                roc_auc_score(
                    y_true,
                    probabilities[:, ordered_columns],
                    labels=ordered_labels,
                    multi_class="ovr",
                    average="macro",
                )
            )
        else:
            metrics["roc_auc_ovr_macro"] = None
            metrics["roc_auc_note"] = (
                "ROC-AUC was not computed because the evaluation labels do not "
                "contain every configured class."
            )

    if output_prefix is not None:
        prefix = Path(output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        figure, axis = plt.subplots(figsize=(7, 6))
        image = axis.imshow(matrix, cmap="Greens")
        axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        axis.set_yticks(range(len(labels)), labels)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.set_title("Confusion Matrix")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        figure.savefig(f"{prefix}_confusion_matrix.png", dpi=160)
        plt.close(figure)
        if (
            y_proba is not None
            and probabilities.shape == (len(y_true), len(labels))
            and set(y_true) == set(labels)
            and len(labels) > 2
        ):
            binary_truth = label_binarize(y_true, classes=labels)
            roc_figure, roc_axis = plt.subplots(figsize=(7, 6))
            for class_index, label in enumerate(labels):
                false_positive_rate, true_positive_rate, _ = roc_curve(
                    binary_truth[:, class_index],
                    probabilities[:, class_index],
                )
                roc_axis.plot(
                    false_positive_rate,
                    true_positive_rate,
                    label=str(label),
                )
            roc_axis.plot([0, 1], [0, 1], linestyle="--", color="grey")
            roc_axis.set(
                xlabel="False positive rate",
                ylabel="True positive rate",
                title="One-vs-rest ROC curves",
                xlim=(0, 1),
                ylim=(0, 1),
            )
            roc_axis.legend(title="Class")
            roc_figure.tight_layout()
            roc_figure.savefig(f"{prefix}_roc_curve.png", dpi=160)
            plt.close(roc_figure)
    if metrics_path is not None:
        save_json(metrics, metrics_path)
    return metrics


def compare_metrics(metric_files, output_csv="outputs/metrics/model_comparison.csv"):
    """Compare only supplied run artifacts; model selection uses validation macro F1."""
    rows = []
    for metric_file in metric_files:
        with open(metric_file, encoding="utf-8") as handle:
            data = json.load(handle)
        rows.append(
            {
                key: data.get(key)
                for key in (
                    "model",
                    "accuracy",
                    "precision_macro",
                    "recall_macro",
                    "f1_macro",
                    "f1_weighted",
                    "validation_macro_f1",
                    "training_seconds",
                    "inference_ms_per_image",
                    "model_size_mb",
                )
            }
        )

    columns = [
        "model",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "validation_macro_f1",
        "training_seconds",
        "inference_ms_per_image",
        "model_size_mb",
    ]
    comparison = pd.DataFrame(rows, columns=columns)
    if not comparison.empty:
        comparison["selection_rank"] = comparison["validation_macro_f1"].rank(
            ascending=False,
            method="min",
            na_option="bottom",
        )
        comparison["selected_best"] = (
            comparison["selection_rank"].eq(1)
            & comparison["validation_macro_f1"].notna()
        )
    else:
        comparison["selection_rank"] = pd.Series(dtype="float64")
        comparison["selected_best"] = pd.Series(dtype="bool")

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)
    return comparison
