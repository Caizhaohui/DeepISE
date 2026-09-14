from deepise_ml.screening.evidence import (
    CandidateRoute,
    HomologyContractError,
    HomologyEvidence,
    RouteReasonCode,
    RoutingDecision,
)
from deepise_ml.screening.runtime import ProteinScreeningRecord

REJECT_SCORE_THRESHOLD = 0.35
HIGH_SCORE_THRESHOLD = 0.70


class CandidateRouter:
    def route(
        self,
        prediction: ProteinScreeningRecord,
        evidence: HomologyEvidence,
    ) -> RoutingDecision:
        if prediction.protein_id != evidence.protein_id:
            raise HomologyContractError("Stage-1 and homology evidence protein IDs do not match")
        strong_homology = evidence.has_strong_homology
        if prediction.tpase_score < REJECT_SCORE_THRESHOLD:
            if strong_homology:
                return RoutingDecision(
                    route=CandidateRoute.UNCERTAIN,
                    reason_code=RouteReasonCode.UNCERTAIN_PLM_HOMOLOGY_CONFLICT,
                )
            return RoutingDecision(
                route=CandidateRoute.REJECT,
                reason_code=RouteReasonCode.REJECT_LOW_SCORE_NO_STRONG_HOMOLOGY,
            )
        if prediction.tpase_score >= HIGH_SCORE_THRESHOLD:
            if strong_homology:
                return RoutingDecision(
                    route=CandidateRoute.ACCEPT_KNOWN,
                    reason_code=RouteReasonCode.ACCEPT_KNOWN_HIGH_SCORE_STRONG_HOMOLOGY,
                )
            return RoutingDecision(
                route=CandidateRoute.STAGE2A,
                reason_code=RouteReasonCode.STAGE2A_HIGH_SCORE_WEAK_OR_NO_HOMOLOGY,
            )
        return RoutingDecision(
            route=CandidateRoute.UNCERTAIN,
            reason_code=RouteReasonCode.UNCERTAIN_INTERMEDIATE_SCORE,
        )


__all__ = [
    "HIGH_SCORE_THRESHOLD",
    "REJECT_SCORE_THRESHOLD",
    "CandidateRoute",
    "CandidateRouter",
    "RouteReasonCode",
    "RoutingDecision",
]
