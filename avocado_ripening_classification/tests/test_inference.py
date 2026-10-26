import json

import main_demo
from PIL import Image


def test_prediction_summary_saves_json_under_project_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(main_demo, "PROJECT_ROOT", tmp_path)
    predictions = {
        "svm": {"predicted_class": "ripe", "confidence": 0.8},
        "mobilenetv2": {
            "predicted_class": "ripe",
            "confidence": 0.9,
            "inference_ms": 2.5,
        },
    }

    output_path = main_demo.save_prediction_summary(
        tmp_path / "avocado.png",
        predictions,
    )

    assert output_path == tmp_path / "outputs" / "predictions" / "avocado_prediction.json"
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["image"].endswith("avocado.png")
    assert data["predictions"] == predictions


def test_prediction_summary_skips_empty_results(tmp_path):
    assert main_demo.save_prediction_summary(tmp_path / "avocado.png", {}) is None


def test_demo_runs_available_models_and_writes_predictions(
    tmp_path,
    monkeypatch,
    capsys,
):
    image_path = tmp_path / "test_avocado.png"
    Image.new("RGB", (8, 8), (80, 120, 40)).save(image_path)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    for model_name in ("svm", "rf", "mobilenetv2"):
        (models_dir / f"{model_name}_model.joblib").touch()
    (models_dir / "mobilenetv2.pth").touch()
    monkeypatch.setattr(main_demo, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "main_demo.py",
            "--image",
            str(image_path),
            "--models-dir",
            str(models_dir),
            "--no-display",
        ],
    )
    monkeypatch.setattr(main_demo, "predict_classical", lambda *_: ("ripe", 0.8))
    monkeypatch.setattr(
        main_demo,
        "predict_dl",
        lambda *_: ("ripe", 0.9, 2.5),
    )

    main_demo.main()

    output_path = tmp_path / "outputs" / "predictions" / "test_avocado_prediction.json"
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert set(data["predictions"]) == {"svm", "rf", "mobilenetv2"}
    assert "Saved prediction summary" in capsys.readouterr().out
