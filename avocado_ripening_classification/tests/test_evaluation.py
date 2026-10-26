import json

import numpy as np

from src.evaluation import compare_metrics, evaluate_predictions

LABELS = ["underripe", "breaking", "ripe", "overripe"]


def test_evaluation_returns_classification_metrics_and_saves_plots(tmp_path):
    truth = LABELS * 2
    predictions = ["underripe", "breaking", "ripe", "ripe"] * 2
    probabilities = np.array(
        [
            [0.85, 0.05, 0.05, 0.05],
            [0.10, 0.75, 0.10, 0.05],
            [0.05, 0.10, 0.80, 0.05],
            [0.05, 0.10, 0.55, 0.30],
        ]
        * 2
    )
    prefix = tmp_path / "evaluation"

    metrics = evaluate_predictions(
        truth,
        predictions,
        LABELS,
        y_proba=probabilities,
        output_prefix=prefix,
        metrics_path=tmp_path / "evaluation_metrics.json",
    )

    assert set(
        (
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "f1_weighted",
            "classification_report",
            "roc_auc_ovr_macro",
        )
    ).issubset(metrics)
    assert 0 <= metrics["roc_auc_ovr_macro"] <= 1
    assert (tmp_path / "evaluation_confusion_matrix.png").is_file()
    assert (tmp_path / "evaluation_roc_curve.png").is_file()
    assert (tmp_path / "evaluation_metrics.json").is_file()


def test_compare_metrics_selects_with_validation_not_test_score(tmp_path):
    lower_validation = tmp_path / "lower_validation.json"
    higher_validation = tmp_path / "higher_validation.json"
    lower_validation.write_text(
        json.dumps(
            {
                "model": "test-score-winner",
                "validation_macro_f1": 0.70,
                "f1_macro": 0.99,
            }
        ),
        encoding="utf-8",
    )
    higher_validation.write_text(
        json.dumps(
            {
                "model": "validation-winner",
                "validation_macro_f1": 0.80,
                "f1_macro": 0.10,
            }
        ),
        encoding="utf-8",
    )

    comparison = compare_metrics(
        [lower_validation, higher_validation],
        tmp_path / "comparison.csv",
    )

    selected = comparison.loc[comparison.selected_best, "model"].tolist()
    assert selected == ["validation-winner"]
    assert (tmp_path / "comparison.csv").is_file()
