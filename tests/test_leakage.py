"""Unit tests for leakage audit verification."""

import pytest
import tempfile
from pathlib import Path
from deepise_ml.dataset.leakage import audit_leakage


def test_leakage_audit_passing():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        query_faa = tmp / "query.faa"
        search_tsv = tmp / "search.tsv"

        query_faa.write_text(">q1\nAAAA\n>q2\nCCCC\n")
        # q1 hits t1 with 25% identity, q2 hits t2 with 50% identity but only 30% coverage
        search_tsv.write_text(
            "q1\tt1\t0.25\t100\t0.90\t0.90\t1e-5\t50.0\n"
            "q2\tt2\t0.50\t30\t0.30\t0.30\t1e-2\t25.0\n"
        )

        hits_df, violations_df, summary = audit_leakage(
            search_tsv=search_tsv,
            query_faa=query_faa,
            identity_threshold=0.30,
            coverage_threshold=0.80,
        )

        assert summary["pass_audit"] is True
        assert summary["total_violations"] == 0
        assert len(violations_df) == 0


def test_leakage_audit_detects_violation():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        query_faa = tmp / "query.faa"
        search_tsv = tmp / "search.tsv"

        query_faa.write_text(">q1\nAAAA\n")
        # q1 hits t1 with 45% identity and 85% coverage -> violation!
        search_tsv.write_text(
            "q1\tt1\t0.45\t100\t0.85\t0.85\t1e-10\t120.0\n"
        )

        hits_df, violations_df, summary = audit_leakage(
            search_tsv=search_tsv,
            query_faa=query_faa,
            identity_threshold=0.30,
            coverage_threshold=0.80,
        )

        assert summary["pass_audit"] is False
        assert summary["total_violations"] == 1
        assert len(violations_df) == 1
