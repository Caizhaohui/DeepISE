"""Unit tests for Homology v2 metrics and reciprocal coverage auditing."""

import tempfile
from pathlib import Path
import pytest
from deepise_ml.dataset.leakage import audit_homology_v2


def test_homology_v2_exact_classification():
    with tempfile.TemporaryDirectory() as tmpdir:
        tsv_path = Path(tmpdir) / "search.tsv"
        # query, target, fident, alnlen, qcov, tcov, evalue, bits
        tsv_content = (
            "q_exact\tt_exact\t1.000\t300\t1.000\t1.000\t0.0\t600.0\n"
            "q_close\tt_close\t0.350\t300\t0.850\t0.850\t1e-20\t200.0\n"
            # Domain only: 95% identity, but qcov 0.15, tcov 0.15 (reciprocal < 80%)
            "q_domain\tt_domain\t0.950\t45\t0.150\t0.150\t1e-10\t90.0\n"
            # Remote 20 strict: 18% identity, qcov 85%, tcov 85%
            "q_remote20\tt_remote20\t0.180\t300\t0.850\t0.850\t1e-5\t50.0\n"
            # Remote 30 strict: 25% identity, qcov 90%, tcov 90%
            "q_remote30\tt_remote30\t0.250\t300\t0.900\t0.900\t1e-8\t75.0\n"
        )
        tsv_path.write_text(tsv_content)

        query_ids = [
            "q_exact",
            "q_close",
            "q_domain",
            "q_remote20",
            "q_remote30",
            "q_orphan",  # Zero hits
        ]

        df, summary = audit_homology_v2(
            tsv_path,
            query_ids,
            identity_threshold=0.30,
            coverage_threshold=0.80,
        )

        assert summary["total_queries"] == 6
        assert summary["l1_exact_count"] == 1
        assert summary["l2_close_count"] == 1
        assert summary["l3_domain_only_count"] == 1
        assert summary["no_full_length_count"] == 1  # q_orphan
        assert not summary["pass_strict_split"]  # Has L1 and L2

        # Check per-query classification
        res = {row["test_id"]: row for row in df.iter_rows(named=True)}
        assert res["q_exact"]["homology_class"] == "L1_EXACT"
        assert res["q_close"]["homology_class"] == "L2_CLOSE"
        assert res["q_domain"]["homology_class"] == "L3_DOMAIN_ONLY"
        assert res["q_remote20"]["homology_class"] == "REMOTE_20_STRICT"
        assert res["q_remote30"]["homology_class"] == "REMOTE_30_STRICT"
        assert res["q_orphan"]["homology_class"] == "NO_FULL_LENGTH_HOMOLOG"


def test_max_identity_vs_bitscore_selection():
    """Ensure max_full_length_identity chooses the highest identity hit, not merely highest bitscore."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tsv_path = Path(tmpdir) / "search.tsv"
        # Hit 1: high bitscore (longer match), lower identity (25%)
        # Hit 2: lower bitscore (shorter full-length match), higher identity (28%)
        tsv_content = (
            "q1\tt_high_bits\t0.250\t400\t0.950\t0.950\t1e-30\t350.0\n"
            "q1\tt_high_id\t0.280\t350\t0.820\t0.820\t1e-25\t310.0\n"
        )
        tsv_path.write_text(tsv_content)

        df, summary = audit_homology_v2(
            tsv_path,
            ["q1"],
            identity_threshold=0.30,
            coverage_threshold=0.80,
        )

        row = df.to_dicts()[0]
        assert row["max_full_length_identity"] == pytest.approx(0.280)
        assert row["max_identity_train_id"] == "t_high_id"
        assert row["best_bitscore_identity"] == pytest.approx(0.250)
        assert row["best_bitscore_train_id"] == "t_high_bits"
        assert row["homology_class"] == "REMOTE_30_STRICT"
