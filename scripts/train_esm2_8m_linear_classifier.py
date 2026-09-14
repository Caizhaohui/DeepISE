# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "polars", "scikit-learn"]
# ///
# ─── How to run ───
# PYTHONPATH=python uv run scripts/train_esm2_8m_linear_classifier.py

import json
from pathlib import Path

import numpy as np
import polars as pl
from deepise_ml.benchmark.evaluation import find_fdr_threshold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def main() -> None:
    embeddings_dir = Path("data/processed/embeddings")
    split_dir = Path("data/splits/cluster30")
    model_dir = Path("benchmark/models")
    output_path = model_dir / "esm2_8m_linear_classifier.npz"
    metadata_path = model_dir / "esm2_8m_linear_classifier.json"

    train_embeddings = np.load(embeddings_dir / "esm2_8m_train.npy")
    validation_embeddings = np.load(embeddings_dir / "esm2_8m_val.npy")
    train_labels = pl.read_parquet(split_dir / "train_combined.parquet")["label"].to_numpy()
    validation_labels = pl.read_parquet(split_dir / "validation_combined.parquet")["label"].to_numpy()

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_embeddings)
    scaled_validation = scaler.transform(validation_embeddings)
    classifier = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    classifier.fit(scaled_train, train_labels)
    validation_scores = classifier.predict_proba(scaled_validation)[:, 1]
    threshold, achieved_fdr, validation_recall = find_fdr_threshold(
        validation_labels,
        validation_scores,
        target_fdr=0.05,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        mean=np.asarray(scaler.mean_, dtype=np.float32),
        scale=np.asarray(scaler.scale_, dtype=np.float32),
        coefficients=np.asarray(classifier.coef_[0], dtype=np.float32),
        intercept=np.asarray(classifier.intercept_[0], dtype=np.float32),
        threshold=np.asarray(threshold, dtype=np.float32),
    )
    metadata_path.write_text(
        json.dumps(
            {
                "artifact_version": 1,
                "classifier": "sklearn.linear_model.LogisticRegression",
                "model_key": "esm2_8m",
                "embedding_dimension": int(train_embeddings.shape[1]),
                "pooling": "mean",
                "dataset_split": "cluster30",
                "calibration_split": "validation",
                "target_fdr": 0.05,
                "achieved_fdr": float(achieved_fdr),
                "validation_recall": float(validation_recall),
                "random_seed": 42,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
