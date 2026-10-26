"""Classical feature extraction, cross-validated tuning, and final evaluation."""
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .evaluation import evaluate_predictions
from .feature_extraction import extract_features
from .utils import save_json


def make_feature_matrix(df):
    if df.empty:
        raise ValueError("Cannot extract features from an empty dataset split.")
    features = np.vstack([extract_features(path) for path in df["path"]])
    labels = df["label"].to_numpy()
    return features, labels


def _predict_with_latency(model, features):
    model.predict(features[:1])
    started = time.perf_counter()
    predictions = model.predict(features)
    elapsed = time.perf_counter() - started
    return predictions, elapsed * 1000 / len(features)


def train_classical(
    train_df,
    val_df,
    test_df,
    output_dir: str | Path = "models",
    metrics_dir: str | Path = "outputs/metrics",
    cv_splits=3,
    seed=42,
):
    """Fit SVM and Random Forest; use test data only for final reporting."""
    for split_name, frame in (
        ("training", train_df),
        ("validation", val_df),
        ("test", test_df),
    ):
        if frame.empty:
            raise ValueError(f"The {split_name} split cannot be empty.")
    if cv_splits < 2:
        raise ValueError("cv_splits must be at least 2.")

    train_counts = train_df["label"].value_counts()
    effective_splits = min(cv_splits, int(train_counts.min()) // 2)
    if effective_splits < 2:
        raise ValueError(
            "Cross-validated SVM probability calibration requires at least 4 "
            "training images per class."
        )
    if effective_splits != cv_splits:
        print(
            f"Reducing CV folds from {cv_splits} to {effective_splits}: "
            "the smallest training class has fewer samples."
        )

    output_path = Path(output_dir)
    metrics_path = Path(metrics_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics_path.mkdir(parents=True, exist_ok=True)

    x_train, y_train = make_feature_matrix(train_df)
    x_validation, y_validation = make_feature_matrix(val_df)
    x_test, y_test = make_feature_matrix(test_df)
    cross_validation = StratifiedKFold(
        n_splits=effective_splits,
        shuffle=True,
        random_state=seed,
    )
    candidates = {
        "svm": (
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "clf",
                        CalibratedClassifierCV(
                            estimator=SVC(
                                class_weight="balanced",
                                random_state=seed,
                            ),
                            method="sigmoid",
                            cv=2,
                            ensemble=False,
                        ),
                    ),
                ]
            ),
            {
                "clf__estimator__C": [0.1, 1, 10],
                "clf__estimator__kernel": ["rbf"],
                "clf__estimator__gamma": ["scale", "auto"],
            },
        ),
        "rf": (
            Pipeline(
                [
                    (
                        "clf",
                        RandomForestClassifier(
                            class_weight="balanced",
                            random_state=seed,
                            n_jobs=-1,
                        ),
                    )
                ]
            ),
            {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [None, 20],
                "clf__min_samples_leaf": [1, 2],
            },
        ),
    }

    results = []
    for name, (pipeline, parameter_grid) in candidates.items():
        started = time.perf_counter()
        search = GridSearchCV(
            pipeline,
            parameter_grid,
            cv=cross_validation,
            scoring="f1_macro",
            n_jobs=-1,
            refit=True,
            verbose=1,
            error_score="raise",
        )
        search.fit(x_train, y_train)
        training_seconds = time.perf_counter() - started
        model = search.best_estimator_

        validation_predictions = model.predict(x_validation)
        validation_metrics = evaluate_predictions(
            y_validation,
            validation_predictions,
            model.classes_,
        )
        test_predictions, inference_ms = _predict_with_latency(model, x_test)
        test_probabilities = (
            model.predict_proba(x_test) if hasattr(model, "predict_proba") else None
        )
        metrics = evaluate_predictions(
            y_test,
            test_predictions,
            model.classes_,
            y_proba=test_probabilities,
            output_prefix=metrics_path.parent / "confusion_matrices" / name,
        )
        bundle_path = output_path / f"{name}_model.joblib"
        joblib.dump(
            {
                "pipeline": model,
                "feature_version": 1,
                "classes": list(model.classes_),
            },
            bundle_path,
        )
        metrics.update(
            {
                "model": name,
                "training_seconds": round(training_seconds, 3),
                "inference_ms_per_image": round(inference_ms, 4),
                "model_size_mb": round(bundle_path.stat().st_size / (1024**2), 4),
                "best_params": search.best_params_,
                "validation_macro_f1": validation_metrics["f1_macro"],
                "cv_best_macro_f1": float(search.best_score_),
            }
        )
        save_json(metrics, metrics_path / f"{name}_metrics.json")
        results.append(metrics)

    return pd.DataFrame(results)
