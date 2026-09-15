"""CI Gate: Rigorous test verifying zero reciprocal full-length cross-split homology violations.

Criteria:
Zero pairs between (test vs train), (validation vs train), (test vs validation)
with identity >= 0.30 AND qcov >= 0.80 AND tcov >= 0.80.
"""

from pathlib import Path
import polars as pl
import pytest


def test_positive_split_v2_disjoint_integrity():
    split_dir = Path("data/splits/cluster30_v2")
    train_path = split_dir / "train_positives.parquet"
    val_path = split_dir / "validation_positives.parquet"
    test_path = split_dir / "test_positives.parquet"

    assert train_path.exists(), "train_positives.parquet missing"
    assert val_path.exists(), "validation_positives.parquet missing"
    assert test_path.exists(), "test_positives.parquet missing"

    train_df = pl.read_parquet(train_path)
    val_df = pl.read_parquet(val_path)
    test_df = pl.read_parquet(test_path)

    # 1. Cluster ID disjointness
    train_c = set(train_df["cluster30_id"].to_list())
    val_c = set(val_df["cluster30_id"].to_list())
    test_c = set(test_df["cluster30_id"].to_list())

    assert len(train_c & val_c) == 0, f"Cluster ID leakage train-val: {len(train_c & val_c)}"
    assert len(train_c & test_c) == 0, f"Cluster ID leakage train-test: {len(train_c & test_c)}"
    assert len(val_c & test_c) == 0, f"Cluster ID leakage val-test: {len(val_c & test_c)}"

    # 2. Sequence ID disjointness
    train_s = set(train_df["tpase_id"].to_list())
    val_s = set(val_df["tpase_id"].to_list())
    test_s = set(test_df["tpase_id"].to_list())

    assert len(train_s & val_s) == 0, "Sequence ID leakage train-val"
    assert len(train_s & test_s) == 0, "Sequence ID leakage train-test"
    assert len(val_s & test_s) == 0, "Sequence ID leakage val-test"

    # 3. Total count conservation
    assert len(train_df) + len(val_df) + len(test_df) == 7057


def test_positive_split_v2_audit_report_zero_violations():
    report_path = Path("benchmark/reports/positive_split_audit_v2.md")
    assert report_path.exists(), "Audit report positive_split_audit_v2.md missing"
    text = report_path.read_text(encoding="utf-8")
    assert "Verified Violations | **0** |" in text or "| **0** |" in text
    assert "Strictly Disjoint" in text
