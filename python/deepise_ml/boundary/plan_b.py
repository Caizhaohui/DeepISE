"""Plan B: Canonical TIR/TSD Prototype Engine for DeepISE Phase-2."""

import time
from typing import Optional
from deepise_ml.boundary.schemas import BoundaryPrediction, TIRMatch, TSDMatch
from deepise_ml.boundary.tir import find_candidate_tirs
from deepise_ml.boundary.tsd import find_candidate_tsd


class PlanBCanonicalEngine:
    """
    Plan B: Standard Canonical IS Element Boundary Detector.
    
    Assumes standard DDE transposition paradigm:
    - Flanking Terminal Inverted Repeats (TIR)
    - Target Site Duplication (TSD)
    """

    def __init__(
        self,
        upstream_search_len: int = 500,
        downstream_search_len: int = 500,
        min_tir_len: int = 8,
        max_tir_len: int = 45,
        min_identity: float = 0.65,
    ):
        self.upstream_search_len = upstream_search_len
        self.downstream_search_len = downstream_search_len
        self.min_tir_len = min_tir_len
        self.max_tir_len = max_tir_len
        self.min_identity = min_identity

    def predict_boundary(
        self,
        contig: str,
        tpase_start: int,
        tpase_end: int,
        contig_id: str = "contig",
        is_name: str = "unknown",
        family: str = "unknown",
    ) -> BoundaryPrediction:
        """Predict IS boundary using canonical TIR + TSD engine."""
        t0 = time.perf_counter()

        # Step 1: Scan for candidate TIR pairs
        tirs = find_candidate_tirs(
            contig=contig,
            tpase_start=tpase_start,
            tpase_end=tpase_end,
            upstream_search_len=self.upstream_search_len,
            downstream_search_len=self.downstream_search_len,
            min_tir_len=self.min_tir_len,
            max_tir_len=self.max_tir_len,
            min_identity=self.min_identity,
            seed_k=7,
            top_k=6,
        )

        best_score = -1.0
        best_pred: Optional[BoundaryPrediction] = None

        for tir in tirs:
            cur_l = tir.left_start
            cur_r = tir.right_end
            is_len = cur_r - cur_l

            # Biological plausibility: IS element must encompass the Tpase ORF
            if cur_l > tpase_start or cur_r < tpase_end:
                continue

            # Length plausibility score (typical IS: 700 - 3000 bp)
            if 600 <= is_len <= 3500:
                len_score = 1.0
            elif 400 <= is_len <= 6000:
                len_score = 0.6
            else:
                len_score = 0.2

            # Step 2: Scan for flanking TSD
            tsd, refined_l, refined_r = find_candidate_tsd(
                contig=contig,
                left_boundary=cur_l,
                right_boundary=cur_r,
                min_tsd_len=2,
                max_tsd_len=14,
                allow_mismatch=True,
                micro_shift=3,
            )

            # Scoring components
            s_tir = min(1.0, tir.score / 45.0)
            s_tsd = min(1.0, tsd.score / 20.0) if tsd else 0.0

            # Canonical composite score
            # TIR (50%) + TSD (35%) + Length (15%)
            conf = 0.50 * s_tir + 0.35 * s_tsd + 0.15 * len_score

            if conf > best_score:
                best_score = conf
                best_pred = BoundaryPrediction(
                    contig_id=contig_id,
                    is_name=is_name,
                    family=family,
                    method="plan_b",
                    predicted_start=refined_l,
                    predicted_end=refined_r,
                    predicted_length=refined_r - refined_l,
                    is_complete=True if (tir and conf >= 0.5) else False,
                    confidence_score=round(conf, 4),
                    tir=tir,
                    tsd=tsd,
                    structure=None,
                    latency_ms=0.0,
                )

        # Fallback if no valid TIR found: naive heuristic around Tpase
        if best_pred is None:
            # Flank heuristic: 100 bp upstream, 80 bp downstream
            fallback_start = max(0, tpase_start - 100)
            fallback_end = min(len(contig), tpase_end + 80)
            best_pred = BoundaryPrediction(
                contig_id=contig_id,
                is_name=is_name,
                family=family,
                method="plan_b",
                predicted_start=fallback_start,
                predicted_end=fallback_end,
                predicted_length=fallback_end - fallback_start,
                is_complete=False,
                confidence_score=0.15,
                tir=None,
                tsd=None,
                structure=None,
                latency_ms=0.0,
            )

        t1 = time.perf_counter()
        best_pred.latency_ms = round((t1 - t0) * 1000, 2)
        return best_pred
