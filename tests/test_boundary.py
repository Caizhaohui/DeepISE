"""Unit tests for Phase-2 boundary detection engines (Plan A and Plan B)."""

import pytest
from deepise_ml.boundary.tir import align_semiglobal, find_candidate_tirs, reverse_complement
from deepise_ml.boundary.tsd import find_candidate_tsd
from deepise_ml.boundary.plan_a import PlanAAdaptiveEngine
from deepise_ml.boundary.plan_b import PlanBCanonicalEngine
from deepise_ml.boundary.schemas import BoundaryPrediction


def test_semiglobal_alignment_exact():
    s1 = "ATGCGATCGATCGATC"
    s2 = "ATGCGATCGATCGATC"
    score, aln_len, mismatches, gaps, identity = align_semiglobal(s1, s2)
    assert mismatches == 0
    assert gaps == 0
    assert identity == 1.0
    assert score == 32


def test_semiglobal_alignment_mismatch():
    s1 = "ATGCGATCGATCGATC"
    s2 = "ATGCGATCTATCGATC"  # 1 mismatch
    score, aln_len, mismatches, gaps, identity = align_semiglobal(s1, s2)
    assert mismatches == 1
    assert gaps == 0
    assert identity == 15 / 16


def test_find_candidate_tsd():
    flank_left = "A" * 50 + "TTTT"
    flank_right = "CCCC" + "G" * 50
    tsd = "ATGC"
    is_seq = "C" * 200
    contig = flank_left + tsd + is_seq + tsd + flank_right

    true_start = len(flank_left) + len(tsd)
    true_end = true_start + len(is_seq)

    matched_tsd, ref_l, ref_r = find_candidate_tsd(contig, true_start, true_end, min_tsd_len=2, max_tsd_len=8, micro_shift=0)
    assert matched_tsd is not None
    assert matched_tsd.length == 4
    assert matched_tsd.sequence == "ATGC"
    assert ref_l == true_start
    assert ref_r == true_end


def test_plan_b_canonical_prediction():
    from pathlib import Path
    import polars as pl
    benchmark_path = Path("data/benchmark/phase2_ground_truth.parquet")
    assert benchmark_path.exists(), "Phase 2 benchmark parquet must exist"
    df = pl.read_parquet(benchmark_path)
    r = df.filter(pl.col("is_name") == "IS231F").to_dicts()[0]

    engine = PlanBCanonicalEngine()
    pred = engine.predict_boundary(
        contig=r["contig_sequence"],
        tpase_start=r["tpase_start"],
        tpase_end=r["tpase_end"],
        contig_id=r["contig_id"],
        is_name=r["is_name"],
        family=r["family"],
    )
    assert isinstance(pred, BoundaryPrediction)
    assert pred.method == "plan_b"
    assert abs(pred.predicted_start - r["true_start"]) <= 3
    assert abs(pred.predicted_end - r["true_end"]) <= 3
    assert pred.tir is not None
    assert pred.confidence_score >= 0.5


def test_plan_a_special_families():
    engine = PlanAAdaptiveEngine()
    # 1. IS110 test contig
    flank_up = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATC" * 6
    flank_down = "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA" * 6
    # IS110 terminal motifs: 5' starts with AATGAAAGGGA, 3' ends with GGAGTCCATATA
    motif_5p = "AATGAAAGGGA"
    motif_3p = "GGAGTCCATATA"
    is_core = "GAATTCGCGGCCGCACTAGTGAGCTCGTCGACCCGGGAATT" * 25
    is_seq = motif_5p + is_core + motif_3p
    contig = flank_up + is_seq + flank_down

    true_start = len(flank_up)
    true_end = true_start + len(is_seq)

    tpase_start = true_start + 85
    tpase_end = true_end - 280

    pred_110 = engine.predict_boundary(
        contig=contig,
        tpase_start=tpase_start,
        tpase_end=tpase_end,
        contig_id="test_110_01",
        is_name="IS110Test",
        family="IS110",
    )
    assert pred_110.method == "plan_a"
    assert pred_110.structure is not None
    assert pred_110.structure.feature_type == "is110_recombination"
    assert abs(pred_110.predicted_start - true_start) <= 5
    assert abs(pred_110.predicted_end - true_end) <= 5
