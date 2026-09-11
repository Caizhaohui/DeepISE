"""Unit tests for full-length IS composite scorer and genome scanning pipeline."""

import pytest
import pyrodigal
from deepise_ml.boundary.schemas import BoundaryPrediction, TIRMatch, TSDMatch, StructureEvidence
from deepise_ml.composite.scorer import ISCompositeScorer, ISCompositeResult


def test_composite_scorer_complete_canonical():
    scorer = ISCompositeScorer()
    tir = TIRMatch(
        left_start=100,
        left_end=125,
        right_start=1475,
        right_end=1500,
        left_seq="A" * 25,
        right_seq="T" * 25,
        length=25,
        mismatches=1,
        gaps=0,
        identity=0.96,
        score=45.0,
    )
    tsd = TSDMatch(
        left_start=96,
        left_end=100,
        right_start=1500,
        right_end=1504,
        sequence="CTAG",
        length=4,
        mismatches=0,
        score=10.0,
    )
    pred = BoundaryPrediction(
        contig_id="test_contig",
        is_name="IS3_test",
        family="IS3",
        method="plan_a",
        predicted_start=100,
        predicted_end=1500,
        predicted_length=1400,
        is_complete=True,
        confidence_score=0.88,
        tir=tir,
        tsd=tsd,
        structure=None,
        latency_ms=1.2,
    )
    res = scorer.score_element(pred, tpase_score=0.95, tpase_length_bp=1050)
    assert isinstance(res, ISCompositeResult)
    assert res.status == "complete"
    assert res.composite_score >= 0.70
    assert "TIR:25bp" in res.notes
    assert "TSD:CTAG" in res.notes


def test_composite_scorer_special_is200():
    scorer = ISCompositeScorer()
    struct = StructureEvidence(
        feature_type="hairpin_stem_loop",
        motif_left="TTAA",
        motif_right=None,
        hairpin_left=None,
        hairpin_right="CCCGC-TTTA-GCGGC",
        stability_score=18.5,
        notes="IS200 hairpin detected",
    )
    pred = BoundaryPrediction(
        contig_id="test_contig",
        is_name="IS200_test",
        family="IS200/IS605",
        method="plan_a",
        predicted_start=50,
        predicted_end=850,
        predicted_length=800,
        is_complete=True,
        confidence_score=0.85,
        tir=None,
        tsd=None,
        structure=struct,
        latency_ms=0.8,
    )
    res = scorer.score_element(pred, tpase_score=0.92, tpase_length_bp=550)
    assert res.status == "complete"
    assert res.composite_score >= 0.65
    assert "hairpin_stem_loop" in res.notes


def test_pyrodigal_basic():
    seq = b"ATGCGATCGATCGATCGAATCGATCGATCGAATCGATCGATAG" * 500
    gf = pyrodigal.GeneFinder(meta=False)
    gf.train(seq)
    genes = gf.find_genes(seq)
    assert len(genes) >= 1
