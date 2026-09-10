"""Unit tests for cluster-aware dataset splitting."""

import pytest
import polars as pl
from deepise_ml.dataset.split import cluster_aware_split


def test_cluster_integrity_no_overlap():
    rows = []
    # Create 10 clusters with different sizes
    for c_idx in range(1, 11):
        c_id = f"cluster30_{c_idx:04d}"
        size = 3 if c_idx % 2 == 0 else 1
        for m in range(size):
            rows.append({
                "tpase_id": f"t_{c_idx}_{m}",
                "is_name": f"IS_{c_idx}",
                "family": f"FAM_{c_idx % 3}",
                "protein_sequence": "AAAA",
                "protein_length": 4,
                "protein_sha256": f"hash_{c_idx}_{m}",
                "cluster30_id": c_id,
            })

    df = pl.DataFrame(rows)
    train_df, val_df, test_df, manifest = cluster_aware_split(df, ratios=(0.70, 0.15, 0.15), seed=42)

    train_clusters = set(train_df["cluster30_id"].to_list())
    val_clusters = set(val_df["cluster30_id"].to_list())
    test_clusters = set(test_df["cluster30_id"].to_list())

    # ZERO cluster overlap
    assert len(train_clusters & val_clusters) == 0
    assert len(train_clusters & test_clusters) == 0
    assert len(val_clusters & test_clusters) == 0

    # Total counts match
    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    assert manifest["counts"]["total"] == len(df)
