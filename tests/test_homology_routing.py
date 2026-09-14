from typing import Literal

import pytest
from deepise_ml.screening.evidence import (
    COMBINED_OUTPUT_COLUMNS,
    HMMEREvidence,
    HomologyContractError,
    HomologyEvidence,
    HomologyEvidenceSet,
    MMseqsEvidence,
    ScreeningWithHomologyRecord,
    parse_mmseqs_evidence,
)
from deepise_ml.screening.routing import (
    CandidateRoute,
    CandidateRouter,
    RouteReasonCode,
)
from deepise_ml.screening.runtime import ProteinScreeningRecord


def stage1_record(score: float, protein_id: str = "protein_a") -> ProteinScreeningRecord:
    return ProteinScreeningRecord(
        protein_id=protein_id,
        length=120,
        sequence_sha256="a" * 64,
        qc_status="passed",
        profile="fast",
        model_key="esm2_8m",
        classifier_id="esm2_8m_linear_classifier",
        tpase_score=score,
        top_family=None,
        family_score=None,
        final_category="Uncertain",
        routing_stage="stage1_sequence",
        notes="sequence_candidate_pending_homology",
    )


def mmseqs_evidence(strength: Literal["none", "weak", "strong"]) -> MMseqsEvidence:
    match strength:
        case "none":
            return MMseqsEvidence.no_hit()
        case "weak":
            return MMseqsEvidence(
                hit=True,
                target_id="weak_hit",
                identity_percent=29.9,
                alignment_length=70,
                query_coverage_percent=80.0,
                target_coverage_percent=80.0,
                evalue=1e-20,
                bitscore=80.0,
            )
        case "strong":
            return MMseqsEvidence(
                hit=True,
                target_id="strong_hit",
                identity_percent=30.0,
                alignment_length=80,
                query_coverage_percent=80.0,
                target_coverage_percent=85.0,
                evalue=1e-5,
                bitscore=90.0,
            )


def evidence_for(
    strength: Literal["none", "weak", "strong"], protein_id: str = "protein_a"
) -> HomologyEvidence:
    return HomologyEvidence(
        protein_id=protein_id,
        mmseqs=mmseqs_evidence(strength),
        hmmer=HMMEREvidence.no_hit(),
    )


@pytest.mark.parametrize(
    ("score", "strength", "expected_route", "expected_reason"),
    [
        (0.34, "none", CandidateRoute.REJECT, RouteReasonCode.REJECT_LOW_SCORE_NO_STRONG_HOMOLOGY),
        (0.34, "strong", CandidateRoute.UNCERTAIN, RouteReasonCode.UNCERTAIN_PLM_HOMOLOGY_CONFLICT),
        (0.50, "none", CandidateRoute.UNCERTAIN, RouteReasonCode.UNCERTAIN_INTERMEDIATE_SCORE),
        (0.70, "strong", CandidateRoute.ACCEPT_KNOWN, RouteReasonCode.ACCEPT_KNOWN_HIGH_SCORE_STRONG_HOMOLOGY),
        (0.70, "weak", CandidateRoute.STAGE2A, RouteReasonCode.STAGE2A_HIGH_SCORE_WEAK_OR_NO_HOMOLOGY),
    ],
)
def test_router_returns_the_deterministic_policy_matrix(
    score: float,
    strength: Literal["none", "weak", "strong"],
    expected_route: CandidateRoute,
    expected_reason: RouteReasonCode,
) -> None:
    decision = CandidateRouter().route(stage1_record(score), evidence_for(strength))

    assert decision.route is expected_route
    assert decision.reason_code is expected_reason
    assert decision.route is not CandidateRoute.STAGE2B


def test_hmmer_full_evalue_can_supply_strong_homology() -> None:
    evidence = HomologyEvidence(
        protein_id="protein_a",
        mmseqs=MMseqsEvidence.no_hit(),
        hmmer=HMMEREvidence(
            hit=True,
            profile_id="IS_hmm",
            full_evalue=1e-5,
            full_bitscore=42.0,
            domain_evalue=1e-6,
            domain_bitscore=35.0,
        ),
    )

    decision = CandidateRouter().route(stage1_record(0.70), evidence)

    assert decision.route is CandidateRoute.ACCEPT_KNOWN
    assert decision.reason_code is RouteReasonCode.ACCEPT_KNOWN_HIGH_SCORE_STRONG_HOMOLOGY


def test_combined_record_preserves_stage1_traceability_and_uses_null_no_hit_fields() -> None:
    stage1 = stage1_record(0.34)
    evidence = evidence_for("none")
    decision = CandidateRouter().route(stage1, evidence)

    combined = ScreeningWithHomologyRecord.from_stage1(stage1, evidence, decision)
    row = combined.tsv_row()

    assert tuple(row) == COMBINED_OUTPUT_COLUMNS
    assert row["protein_id"] == stage1.protein_id
    assert row["sequence_sha256"] == stage1.sequence_sha256
    assert row["qc_status"] == stage1.qc_status
    assert row["tpase_score"] == stage1.tpase_score
    assert row["mmseqs_hit"] is False
    assert row["mmseqs_target_id"] is None
    assert row["mmseqs_evalue"] is None
    assert row["hmmer_hit"] is False
    assert row["hmmer_profile_id"] is None
    assert row["route"] == CandidateRoute.REJECT.value


def test_malformed_or_duplicate_evidence_is_rejected_by_contract_error() -> None:
    with pytest.raises(HomologyContractError, match="no-hit MMseqs evidence"):
        parse_mmseqs_evidence(
            hit=False,
            target_id="misleading_hit",
            identity_percent=None,
            alignment_length=None,
            query_coverage_percent=None,
            target_coverage_percent=None,
            evalue=None,
            bitscore=None,
        )

    first = evidence_for("none")
    duplicate = evidence_for("strong")
    with pytest.raises(HomologyContractError, match="duplicate homology evidence"):
        HomologyEvidenceSet(records=(first, duplicate))


def test_composition_rejects_mismatched_stage1_and_homology_ids() -> None:
    stage1 = stage1_record(0.70, protein_id="protein_a")
    evidence = evidence_for("strong", protein_id="protein_b")
    decision = CandidateRouter().route(stage1_record(0.70, protein_id="protein_b"), evidence)

    with pytest.raises(HomologyContractError, match="protein IDs do not match"):
        ScreeningWithHomologyRecord.from_stage1(stage1, evidence, decision)
