"""Unit tests for exact deduplication logic."""

import pytest
from deepise_ml.schemas.records import CanonicalTpaseRecord, compute_sha256
from deepise_ml.dataset.deduplicate import deduplicate_tpases


def test_deduplicate_identical_sequences():
    seq1 = "MKKITLALSSAA"
    seq2 = "MKKITLALSSAA"
    seq3 = "MAGGYTR"

    rec1 = CanonicalTpaseRecord(
        tpase_id="t1",
        is_name="IS1A",
        family="IS1",
        protein_sequence=seq1,
        protein_length=len(seq1),
        protein_sha256=compute_sha256(seq1),
    )
    rec2 = CanonicalTpaseRecord(
        tpase_id="t2",
        is_name="IS1B",
        family="IS1",
        protein_sequence=seq2,
        protein_length=len(seq2),
        protein_sha256=compute_sha256(seq2),
    )
    rec3 = CanonicalTpaseRecord(
        tpase_id="t3",
        is_name="IS2A",
        family="IS2",
        protein_sequence=seq3,
        protein_length=len(seq3),
        protein_sha256=compute_sha256(seq3),
    )

    reps, cluster_df = deduplicate_tpases([rec1, rec2, rec3])

    assert len(reps) == 2
    assert len(cluster_df) == 3
    # Check that t1 and t2 share the same exact cluster ID
    t1_c = cluster_df.filter(cluster_df["member_tpase_id"] == "t1")["exact_cluster_id"][0]
    t2_c = cluster_df.filter(cluster_df["member_tpase_id"] == "t2")["exact_cluster_id"][0]
    assert t1_c == t2_c


def test_annotation_conflict_detection():
    seq = "MKKITLALSSAA"
    rec1 = CanonicalTpaseRecord(
        tpase_id="t1",
        is_name="IS1A",
        family="IS1",
        protein_sequence=seq,
        protein_length=len(seq),
        protein_sha256=compute_sha256(seq),
    )
    rec2 = CanonicalTpaseRecord(
        tpase_id="t2",
        is_name="IS3A",
        family="IS3",  # Conflicting family
        protein_sequence=seq,
        protein_length=len(seq),
        protein_sha256=compute_sha256(seq),
    )

    reps, cluster_df = deduplicate_tpases([rec1, rec2])
    assert len(reps) == 1
    assert reps[0].annotation_conflict is True
    assert cluster_df["annotation_conflict"][0] is True
