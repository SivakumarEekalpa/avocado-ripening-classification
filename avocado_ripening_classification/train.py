"""Train both assignment approaches and compare only artifacts from this run."""
import argparse
from pathlib import Path

from src.classical_model import train_classical
from src.dataset import discover_dataset, save_split_manifest, stratified_split
from src.deep_learning_model import train_mobilenet
from src.evaluation import compare_metrics
from src.utils import set_seed

PROJECT_ROOT = Path(__file__).resolve().parent


def _project_path(path):
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw", help="Dataset root directory.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers; 0 is the safest default on Windows and notebooks.",
    )
    parser.add_argument(
        "--skip-dl",
        action="store_true",
        help="Train classical models only (not a complete assignment submission).",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Initialize MobileNetV2 randomly instead of downloading ImageNet weights.",
    )
    args = parser.parse_args()
    if args.cv_splits < 2:
        parser.error("--cv-splits must be at least 2.")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative.")

    set_seed(args.seed)
    data_dir = _project_path(args.data)
    data = discover_dataset(data_dir)
    print("Unique readable images by class:\n", data.label.value_counts().sort_index())
    train_df, validation_df, test_df = stratified_split(data, seed=args.seed)
    save_split_manifest(
        train_df,
        validation_df,
        test_df,
        _project_path("data/processed/split_manifest.csv"),
    )

    metrics_dir = _project_path("outputs/metrics")
    model_dir = _project_path("models")
    classical_results = train_classical(
        train_df,
        validation_df,
        test_df,
        output_dir=model_dir,
        metrics_dir=metrics_dir,
        cv_splits=args.cv_splits,
        seed=args.seed,
    )
    print(
        "Classical test metrics:\n",
        classical_results[
            ["model", "accuracy", "f1_macro", "validation_macro_f1", "training_seconds"]
        ].to_string(index=False),
    )

    metric_files = [
        metrics_dir / "svm_metrics.json",
        metrics_dir / "rf_metrics.json",
    ]
    if not args.skip_dl:
        _, deep_metrics = train_mobilenet(
            train_df,
            validation_df,
            test_df,
            model_path=model_dir / "mobilenetv2.pth",
            metrics_path=metrics_dir / "mobilenetv2_metrics.json",
            epochs=args.epochs,
            pretrained=not args.no_pretrained,
            num_workers=args.num_workers,
            seed=args.seed,
        )
        print(
            "MobileNetV2 test metrics: "
            f"accuracy={deep_metrics['accuracy']:.4f}, "
            f"macro_f1={deep_metrics['f1_macro']:.4f}, "
            f"validation_macro_f1={deep_metrics['validation_macro_f1']:.4f}"
        )
        metric_files.append(metrics_dir / "mobilenetv2_metrics.json")
    else:
        print("Deep learning training skipped; this run is not assignment-complete.")

    comparison = compare_metrics(
        metric_files,
        output_csv=metrics_dir / "model_comparison.csv",
    )
    print("\nCurrent-run model comparison:\n", comparison.to_string(index=False))
    eligible = comparison[comparison.validation_macro_f1.notna()]
    if not eligible.empty:
        best = eligible.sort_values(
            ["validation_macro_f1", "model"],
            ascending=[False, True],
        ).iloc[0]
        print(
            f"\nCandidate selected by validation macro F1: {best['model']} "
            f"(validation macro F1={best['validation_macro_f1']:.4f}; "
            f"held-out test macro F1={best['f1_macro']:.4f})"
        )


if __name__ == "__main__":
    main()
