"""Unit tests for Phase-1 models and evaluation suite."""

import tempfile
from pathlib import Path
import numpy as np
import polars as pl
from deepise_ml.benchmark.evaluation import (
    find_fdr_threshold,
    compute_binary_metrics_at_threshold,
    compute_overall_metrics,
)
from deepise_ml.models.blast import parse_blast_predictions
from deepise_ml.models.mmseqs import parse_mmseqs_predictions
from deepise_ml.models.hmmer import parse_hmmsearch_predictions


def test_find_fdr_threshold():
    # 5 positives with scores [10, 9, 8, 7, 6]
    # 5 negatives with scores [5, 4, 3, 2, 1]
    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_scores = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])

    thresh, fdr, rec = find_fdr_threshold(y_true, y_scores, target_fdr=0.05)
    # Threshold 6.0 yields TP=5, FP=0 -> FDR=0.0 <= 0.05, Recall=1.0
    assert thresh == 6.0
    assert fdr == 0.0
    assert rec == 1.0


def test_binary_metrics_at_threshold():
    y_true = np.array([1, 1, 0, 0])
    y_scores = np.array([0.9, 0.8, 0.7, 0.2])
    m = compute_binary_metrics_at_threshold(y_true, y_scores, threshold=0.75)

    assert m["tp"] == 2
    assert m["fp"] == 0
    assert m["tn"] == 2
    assert m["fn"] == 0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0


def test_parse_predictions_parsers():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        query_faa = tmp / "queries.faa"
        query_faa.write_text(">q1\nAAA\n>q2\nCCC\n")

        # 1. BLAST TSV
        blast_tsv = tmp / "blast.tsv"
        blast_tsv.write_text("q1\tt1\t95.0\t100\t90.0\t1e-20\t250.0\n")
        df_blast = parse_blast_predictions(blast_tsv, query_faa)
        assert len(df_blast) == 2
        assert df_blast.filter(pl.col("seq_id") == "q1")["score"][0] == 250.0
        assert df_blast.filter(pl.col("seq_id") == "q2")["score"][0] == 0.0

        # 2. MMseqs TSV
        mmseqs_tsv = tmp / "mmseqs.tsv"
        mmseqs_tsv.write_text("q1\tt1\t0.95\t100\t0.90\t0.90\t1e-20\t300.0\n")
        df_mmseqs = parse_mmseqs_predictions(mmseqs_tsv, query_faa)
        assert len(df_mmseqs) == 2
        assert df_mmseqs.filter(pl.col("seq_id") == "q1")["score"][0] == 300.0
        assert df_mmseqs.filter(pl.col("seq_id") == "q2")["score"][0] == 0.0

        # 3. HMMER tblout
        hmmer_tbl = tmp / "hmmer.tbl"
        hmmer_tbl.write_text(
            "# target name\n"
            "q1          -          Tpase_IS3  -            1.2e-45   180.5   0.1\n"
        )
        df_hmmer = parse_hmmsearch_predictions(hmmer_tbl, query_faa)
        assert len(df_hmmer) == 2
        assert df_hmmer.filter(pl.col("seq_id") == "q1")["score"][0] == 180.5
        assert df_hmmer.filter(pl.col("seq_id") == "q2")["score"][0] == 0.0
