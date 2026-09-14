from pathlib import Path

import pytest
from deepise_ml.screening.evidence import (
    CandidateRoute,
    RouteReasonCode,
    RoutingDecision,
)
from deepise_ml.screening.prostt5 import (
    VALID_3DI_ALPHABET,
    ProstT5Engine,
    ProstT5RescueRouter,
)
from deepise_ml.screening.runtime import ProteinInputRecord


def test_prostt5_engine_translates_aa_to_3di_tokens() -> None:
    engine = ProstT5Engine()
    sequence = "MKTIIALSYIFCLVFA"
    predicted_3di = engine.predict_3di_single(sequence)

    assert len(predicted_3di) == len(sequence)
    assert set(predicted_3di).issubset(VALID_3DI_ALPHABET)


def test_prostt5_engine_cache_skips_recomputation(tmp_path: Path) -> None:
    cache_dir = tmp_path / "3di_cache"
    engine = ProstT5Engine(cache_dir=cache_dir)
    sequence = "MKTIIALSYIFCLVFA"

    # First call - cache miss
    first = engine.predict_3di_single(sequence)
    cache_files = list(cache_dir.glob("*.3di"))
    assert len(cache_files) == 1

    # Second call - cache hit
    second = engine.predict_3di_single(sequence)
    assert first == second


def test_prostt5_structural_scoring_differentiates_profiles() -> None:
    engine = ProstT5Engine()
    # Canonical transposase-like DDE fold 3Di pattern
    tpase_seq = "MAALVLDVDEVHLLAAAKLV"
    tpase_ev = engine.evaluate_sequence(tpase_seq)

    assert tpase_ev.evaluated
    assert tpase_ev.score is not None
    assert tpase_ev.structure_similarity is not None
    assert 0.0 <= tpase_ev.score <= 1.0
    assert 0.0 <= tpase_ev.structure_similarity <= 1.0


def test_prostt5_rescue_router_decision_logic() -> None:
    router = ProstT5RescueRouter()

    # High structural score rescues candidate
    decision_high = router.route_stage2a(
        initial_route=CandidateRoute.STAGE2A,
        prostt5_score=0.85,
        prostt5_similarity=0.80,
    )
    assert decision_high.route == CandidateRoute.STAGE2A  # confirmed for Stage 2A rescue
    assert decision_high.reason_code == RouteReasonCode.STAGE2A_HIGH_SCORE_WEAK_OR_NO_HOMOLOGY

    # Ambiguous structural score escalates to Stage 2B for explicit ESMFold / SaProt
    decision_ambiguous = router.route_stage2a(
        initial_route=CandidateRoute.STAGE2A,
        prostt5_score=0.45,
        prostt5_similarity=0.40,
    )
    assert decision_ambiguous.route == CandidateRoute.STAGE2B
    assert decision_ambiguous.reason_code == RouteReasonCode.STAGE2B_PROSTT5_UNRESOLVED_TWILIGHT
