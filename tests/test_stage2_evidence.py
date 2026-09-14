import pytest
from deepise_ml.screening.evidence import (
    CandidateRoute,
    FinalCategory,
    FusionEvidence,
    HMMEREvidence,
    HomologyEvidence,
    MMseqsEvidence,
    ProstT5Evidence,
    RESEARCH_OUTPUT_COLUMNS,
    ResearchScreeningRecord,
    RouteReasonCode,
    RoutingDecision,
    ScreeningWithHomologyRecord,
    StructureEvidence,
)
from deepise_ml.screening.runtime import ProteinScreeningRecord


def make_stage1_record(score: float = 0.75, protein_id: str = "prot_1") -> ProteinScreeningRecord:
    return ProteinScreeningRecord(
        protein_id=protein_id,
        length=120,
        sequence_sha256="b" * 64,
        qc_status="passed",
        profile="standard",
        model_key="esm2_35m",
        classifier_id="plm_family_classifier",
        tpase_score=score,
        top_family="IS1",
        family_score=0.88,
        final_category="Uncertain",
        routing_stage="stage1_sequence",
        notes="stage1_candidate",
    )


def test_prostt5_evidence_defaults_and_validation() -> None:
    empty_ev = ProstT5Evidence.not_evaluated()
    assert not empty_ev.evaluated
    assert empty_ev.predicted_3di is None
    assert empty_ev.score is None
    assert empty_ev.structure_similarity is None

    valid_ev = ProstT5Evidence(
        evaluated=True,
        predicted_3di="dpdc" * 10,
        score=0.82,
        structure_similarity=0.79,
    )
    assert valid_ev.evaluated
    assert valid_ev.score == pytest.approx(0.82)
    assert valid_ev.structure_similarity == pytest.approx(0.79)


def test_structure_evidence_defaults_and_validation() -> None:
    empty_struct = StructureEvidence.not_evaluated()
    assert not empty_struct.evaluated
    assert empty_struct.mean_plddt is None
    assert empty_struct.saprot_score is None
    assert empty_struct.foldseek_hit is False

    valid_struct = StructureEvidence(
        evaluated=True,
        pdb_path="/tmp/test.pdb",
        mean_plddt=88.5,
        foldseek_hit=True,
        foldseek_target_id="IS1_fold",
        foldseek_score=150.0,
        foldseek_evalue=1e-8,
        saprot_score=0.91,
    )
    assert valid_struct.evaluated
    assert valid_struct.mean_plddt == pytest.approx(88.5)
    assert valid_struct.saprot_score == pytest.approx(0.91)
    assert valid_struct.foldseek_score == pytest.approx(150.0)


def test_research_screening_record_tsv_row_has_full_research_columns() -> None:
    stage1 = make_stage1_record(0.75, "prot_1")
    homology_ev = HomologyEvidence(
        protein_id="prot_1",
        mmseqs=MMseqsEvidence.no_hit(),
        hmmer=HMMEREvidence.no_hit(),
    )
    decision = RoutingDecision(
        route=CandidateRoute.STAGE2A,
        reason_code=RouteReasonCode.STAGE2A_HIGH_SCORE_WEAK_OR_NO_HOMOLOGY,
    )
    homology_record = ScreeningWithHomologyRecord.from_stage1(
        stage1=stage1,
        evidence=homology_ev,
        decision=decision,
    )

    prostt5_ev = ProstT5Evidence(
        evaluated=True,
        predicted_3di="dpdc" * 10,
        score=0.85,
        structure_similarity=0.80,
    )
    struct_ev = StructureEvidence(
        evaluated=True,
        pdb_path="/tmp/prot_1.pdb",
        mean_plddt=85.0,
        foldseek_hit=True,
        foldseek_target_id="IS_ref_1",
        foldseek_score=120.0,
        foldseek_evalue=1e-6,
        saprot_score=0.88,
    )
    fusion_ev = FusionEvidence(
        transposase_score=0.89,
        novelty_score=0.82,
        final_category=FinalCategory.NOVEL_CANDIDATE,
        routing_stage="stage2_fusion",
        notes="high_structure_novelty_candidate",
    )

    record = ResearchScreeningRecord(
        homology_record=homology_record,
        prostt5=prostt5_ev,
        structure=struct_ev,
        fusion=fusion_ev,
    )

    row = record.tsv_row()
    assert tuple(row.keys()) == RESEARCH_OUTPUT_COLUMNS
    assert row["protein_id"] == "prot_1"
    assert row["tpase_score"] == 0.75
    assert row["mmseqs_hit"] is False
    assert row["prostt5_score"] == pytest.approx(0.85)
    assert row["saprot_score"] == pytest.approx(0.88)
    assert row["transposase_score"] == pytest.approx(0.89)
    assert row["novelty_score"] == pytest.approx(0.82)
    assert row["final_category"] == "Novel_candidate"
