from __future__ import annotations

import argparse
from pathlib import Path

from ml.features import DEFAULT_FEATURE_COLUMNS, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the matching model from blocking candidates.")
    parser.add_argument("--train-dir", type=Path, default=Path("student_resource/dataset/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args()

    model, feature_cols = train_model(args.train_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.output_dir / "matching_model.joblib"
    feature_path = args.output_dir / "matching_model_features.txt"

    import joblib

    joblib.dump(model, model_path)
    feature_path.write_text("\n".join(feature_cols), encoding="utf-8")

    print(f"Saved model to {model_path}")
    print(f"Saved feature columns to {feature_path}")
    print(f"Applied features: {', '.join(feature_cols)}")


if __name__ == "__main__":
    main()
