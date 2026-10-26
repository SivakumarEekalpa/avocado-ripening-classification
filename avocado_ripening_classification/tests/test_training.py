import json

import pandas as pd
import pytest

import train


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--cv-splits", "1"], "--cv-splits must be at least 2."),
        (["--num-workers", "-1"], "--num-workers cannot be negative."),
    ],
)
def test_training_cli_rejects_invalid_arguments(
    monkeypatch,
    capsys,
    arguments,
    message,
):
    monkeypatch.setattr("sys.argv", ["train.py", *arguments])

    with pytest.raises(SystemExit) as error:
        train.main()

    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_training_orchestrates_both_approaches_and_comparison(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(train, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(train, "set_seed", lambda *_: None)
    frame = pd.DataFrame(
        {
            "path": ["one.png", "two.png"],
            "label": ["ripe", "overripe"],
        }
    )
    monkeypatch.setattr(train, "discover_dataset", lambda *_: frame)
    monkeypatch.setattr(
        train,
        "stratified_split",
        lambda data, seed: (data, data, data),
    )
    monkeypatch.setattr(train, "save_split_manifest", lambda *args: None)

    def save_model_metrics(metrics_dir, model_name, validation_f1):
        metrics = {
            "model": model_name,
            "accuracy": 0.8,
            "precision_macro": 0.8,
            "recall_macro": 0.8,
            "f1_macro": 0.8,
            "f1_weighted": 0.8,
            "validation_macro_f1": validation_f1,
            "training_seconds": 1.0,
            "inference_ms_per_image": 1.0,
            "model_size_mb": 1.0,
        }
        (metrics_dir / f"{model_name}_metrics.json").write_text(
            json.dumps(metrics),
            encoding="utf-8",
        )
        return metrics

    def train_classical(*args, output_dir, metrics_dir, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        save_model_metrics(metrics_dir, "svm", 0.75)
        save_model_metrics(metrics_dir, "rf", 0.7)
        return pd.DataFrame(
            [
                {
                    "model": name,
                    "accuracy": 0.8,
                    "f1_macro": 0.8,
                    "validation_macro_f1": score,
                    "training_seconds": 1.0,
                }
                for name, score in (("svm", 0.75), ("rf", 0.7))
            ]
        )

    def train_mobilenet(*args, metrics_path, **kwargs):
        metrics = save_model_metrics(
            metrics_path.parent,
            "mobilenetv2",
            0.9,
        )
        return None, metrics

    monkeypatch.setattr(train, "train_classical", train_classical)
    monkeypatch.setattr(train, "train_mobilenet", train_mobilenet)
    monkeypatch.setattr(
        "sys.argv",
        ["train.py", "--epochs", "1", "--cv-splits", "2", "--no-pretrained"],
    )

    train.main()

    assert (tmp_path / "outputs" / "metrics" / "model_comparison.csv").is_file()
    assert "Candidate selected by validation macro F1: mobilenetv2" in capsys.readouterr().out
