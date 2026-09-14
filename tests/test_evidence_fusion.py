import pytest
from deepise_ml.screening.evidence import (
    CandidateRoute,
    FinalCategory,
    HMMEREvidence,
    HomologyEvidence,
    MMseqsEvidence,
    ProstT5Evidence,
    RouteReasonCode,
    RoutingDecision,
    ScreeningWithHomologyRecord,
    StructureEvidence,
)
from deepise_ml.screening.fusion import EvidenceFusionEngine
from deepise_ml.screening.runtime import ProteinScreeningRecord


def make_stage1(score: float, pid: str = "prot_1") -> ProteinScreeningRecord:
    return ProteinScreeningRecord(
        protein_id=pid,
        length=150,
        sequence_sha256="c" * 64,
        qc_status="passed",
        profile="standard",
        model_key="esm2_35m",
        classifier_id="plm_family_classifier",
        tpase_score=score,
        top_family="IS1",
        family_score=0.9,
        final_category="Uncertain",
        routing_stage="stage1_sequence",
        notes="stage1",
    )


def test_fusion_engine_classifies_known_like() -> None:
    engine = EvidenceFusionEngine()
    stage1 = make_stage1(0.85)
    homology = HomologyEvidence(
        protein_id="prot_1",
        mmseqs=MMseqsEvidence(
            hit=True,
            target_id="is1_known",
            identity_percent=85.0,
            alignment_length=140,
            query_coverage_percent=95.0,
            target_coverage_percent=95.0,
            evalue=1e-35,
            bitscore=250.0,
        ),
        hmmer=HMMEREvidence.no_hit(),
    )
    decision = RoutingDecision(
        route=CandidateRoute.ACCEPT_KNOWN,
        reason_code=RouteReasonCode.ACCEPT_KNOWN_HIGH_SCORE_STRONG_HOMOLOGY,
    )
    rec = ScreeningWithHomologyRecord.from_stage1(stage1, homology, decision)

    fusion = engine.fuse(
        homology_record=rec,
        prostt5=ProstT5Evidence.not_evaluated(),
        structure=StructureEvidence.not_evaluated(),
    )

    assert fusion.final_category == FinalCategory.KNOWN_LIKE
    assert fusion.transposase_score >= 0.70
    assert fusion.novelty_score < 0.40  # Low novelty due to high known identity


def test_fusion_engine_classifies_novel_candidate() -> None:
    engine = EvidenceFusionEngine()
    stage1 = make_stage1(0.82)
    # No conventional sequence homology
    homology = HomologyEvidence(
        protein_id="prot_1",
        mmseqs=MMseqsEvidence.no_hit(),
        hmmer=HMMEREvidence.no_hit(),
    )
    decision = RoutingDecision(
        route=CandidateRoute.STAGE2A,
        reason_code=RouteReasonCode.STAGE2A_HIGH_SCORE_WEAK_OR_NO_HOMOLOGY,
    )
    rec = ScreeningWithHomologyRecord.from_stage1(stage1, homology, decision)

    # Strong structure-aware support from ProstT5 and SaProt
    prostt5 = ProstT5Evidence(
        evaluated=True,
        predicted_3di="dpdc" * 10,
        score=0.88,
        structure_similarity=0.85,
    )
    structure = StructureEvidence(
        evaluated=True,
        pdb_path="/tmp/test.pdb",
        mean_plddt=88.0,
        foldseek_hit=True,
        foldseek_target_id="IS_core",
        foldseek_score=135.0,
        foldseek_evalue=1e-12,
        saprot_score=0.90,
    )

    fusion = engine.fuse(
        homology_record=rec,
        prostt5=prostt5,
        structure=structure,
    )

    assert fusion.final_category == FinalCategory.NOVEL_CANDIDATE
    assert fusion.transposase_score >= 0.75
    assert fusion.novelty_score >= 0.70  # High novelty: high confidence + zero homology


def test_fusion_engine_classifies_remote() -> None:
    engine = EvidenceFusionEngine()
    stage1 = make_stage1(0.78)
    # Weak sequence homology (22% identity)
    homology = HomologyEvidence(
        protein_id="prot_1",
        mmseqs=MMseqsEvidence(
            hit=True,
            target_id="distant_is",
            identity_percent=22.0,
            alignment_length=60,
            query_coverage_percent=45.0,
            target_coverage_percent=50.0,
            evalue=1e-3,
            bitscore=35.0,
        ),
        hmmer=HMMEREvidence.no_hit(),
    )
    decision = RoutingDecision(
        route=CandidateRoute.STAGE2A,
        reason_code=RouteReasonCode.STAGE2A_HIGH_SCORE_WEAK_OR_NO_HOMOLOGY,
    )
    rec = ScreeningWithHomologyRecord.from_stage1(stage1, homology, decision)

    prostt5 = ProstT5Evidence(
        evaluated=True,
        predicted_3di="dpdc" * 10,
        score=0.74,
        structure_similarity=0.72,
    )

    fusion = engine.fuse(
        homology_record=rec,
        prostt5=prostt5,
        structure=StructureEvidence.not_evaluated(),
    )

    assert fusion.final_category == FinalCategory.REMOTE
    assert fusion.transposase_score >= 0.70


def test_fusion_engine_classifies_negative() -> None:
    engine = EvidenceFusionEngine()
    stage1 = make_stage1(0.15)
    homology = HomologyEvidence(
        protein_id="prot_1",
        mmseqs=MMseqsEvidence.no_hit(),
        hmmer=HMMEREvidence.no_hit(),
    )
    decision = RoutingDecision(
        route=CandidateRoute.REJECT,
        reason_code=RouteReasonCode.REJECT_LOW_SCORE_NO_STRONG_HOMOLOGY,
    )
    rec = ScreeningWithHomologyRecord.from_stage1(stage1, homology, decision)

    fusion = engine.fuse(
        homology_record=rec,
        prostt5=ProstT5Evidence.not_evaluated(),
        structure=StructureEvidence.not_evaluated(),
    )

    assert fusion.final_category == FinalCategory.NEGATIVE
    assert fusion.transposase_score < 0.35
