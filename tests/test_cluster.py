"""Unit tests for cluster parsing and statistics."""

import tempfile
from pathlib import Path
import polars as pl
from deepise_ml.dataset.cluster import parse_cluster_tsv, compute_cluster_statistics


def test_parse_cluster_tsv_and_statistics():
    with tempfile.TemporaryDirectory() as tmpdir:
        tsv_path = Path(tmpdir) / "clusters.tsv"
        tsv_path.write_text(
            "rep1\trep1\n"
            "rep1\tmem1\n"
            "rep1\tmem2\n"
            "rep2\trep2\n"
            "rep2\tmem3\n"
            "rep3\trep3\n"
        )

        member_to_cluster, cluster_df = parse_cluster_tsv(tsv_path, prefix="test_clu_")

        assert len(member_to_cluster) == 6
        assert len(cluster_df["cluster_id"].unique()) == 3
        assert member_to_cluster["mem1"] == member_to_cluster["rep1"]
        assert member_to_cluster["mem3"] == member_to_cluster["rep2"]

        tpase_df = pl.DataFrame([
            {"tpase_id": "rep1", "family": "IS3", "protein_length": 300},
            {"tpase_id": "mem1", "family": "IS3", "protein_length": 305},
            {"tpase_id": "mem2", "family": "IS3", "protein_length": 298},
            {"tpase_id": "rep2", "family": "IS5", "protein_length": 400},
            {"tpase_id": "mem3", "family": "IS5", "protein_length": 395},
            {"tpase_id": "rep3", "family": "IS1", "protein_length": 250},
        ])

        stats = compute_cluster_statistics(cluster_df, tpase_df)
        assert stats["total_sequences"] == 6
        assert stats["num_clusters"] == 3
        assert stats["num_singletons"] == 1
        assert stats["max_cluster_size"] == 3
