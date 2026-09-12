"""Plan A: Full-Family Adaptive IS Element Boundary Engine for DeepISE Phase-2."""

import time
from typing import Optional
from deepise_ml.boundary.schemas import BoundaryPrediction, TIRMatch, TSDMatch
from deepise_ml.boundary.tir import find_candidate_tirs
from deepise_ml.boundary.tsd import find_candidate_tsd
from deepise_ml.boundary.special_families import (
    detect_is200_is605_boundary,
    detect_is91_boundary,
    detect_is110_boundary,
)

# Canonical family geometric priors (expected 5' flank, expected 3' flank)
FAMILY_DISTANCE_PRIORS = {
    "IS1": (40, 250, 20, 200),
    "IS3": (40, 300, 20, 250),
    "IS4": (50, 350, 30, 300),
    "IS5": (40, 250, 20, 200),
    "IS6": (30, 200, 20, 200),
    "IS21": (80, 500, 40, 400),
    "IS30": (40, 250, 20, 200),
    "IS66": (80, 500, 40, 400),
    "IS256": (50, 300, 30, 250),
    "IS481": (40, 250, 20, 200),
    "IS630": (30, 250, 20, 200),
    "IS701": (40, 300, 20, 250),
    "IS982": (40, 250, 20, 200),
    "IS1182": (40, 300, 20, 250),
    "IS1380": (50, 350, 30, 300),
    "IS1595": (40, 250, 20, 200),
    "ISKra4": (40, 300, 20, 250),
    "ISL3": (40, 300, 20, 250),
    "ISNCY": (40, 300, 20, 250),
}


class PlanAAdaptiveEngine:
    """
    Plan A: Full-Family Adaptive IS Element Boundary Engine.
    
    Features:
    - Family-Aware Routing:
      * IS200/IS605: HUH hairpin secondary structure & 5' motif (no TSD requirement)
      * IS91: Rolling-circle oriIS and terIS motif detection (no TIR/TSD requirement)
      * IS110: DEDD recombinase subterminal boundary detection (no TSD requirement)
    - Canonical DDE Families:
      * Joint TIR + TSD coupled alignment with family-informed geometric bounds.
    """

    def __init__(self):
        pass

    def predict_boundary(
        self,
        contig: str,
        tpase_start: int,
        tpase_end: int,
        contig_id: str = "contig",
        is_name: str = "unknown",
        family: str = "unknown",
    ) -> BoundaryPrediction:
        """Predict IS boundary using family-adaptive architecture."""
        t0 = time.perf_counter()

        # Branch 1: IS200/IS605 (HUH transposase, no TIR, no TSD)
        if family == "IS200/IS605":
            p_s, p_e, conf, struct = detect_is200_is605_boundary(contig, tpase_start, tpase_end)
            t1 = time.perf_counter()
            return BoundaryPrediction(
                contig_id=contig_id,
                is_name=is_name,
                family=family,
                method="plan_a",
                predicted_start=p_s,
                predicted_end=p_e,
                predicted_length=p_e - p_s,
                is_complete=True,
                confidence_score=round(conf, 4),
                tir=None,
                tsd=None,
                structure=struct,
                latency_ms=round((t1 - t0) * 1000, 2),
            )

        # Branch 2: IS91 (Rolling-circle transposase, ori/ter, no TIR, no TSD)
        if family == "IS91":
            p_s, p_e, conf, struct = detect_is91_boundary(contig, tpase_start, tpase_end)
            t1 = time.perf_counter()
            return BoundaryPrediction(
                contig_id=contig_id,
                is_name=is_name,
                family=family,
                method="plan_a",
                predicted_start=p_s,
                predicted_end=p_e,
                predicted_length=p_e - p_s,
                is_complete=True,
                confidence_score=round(conf, 4),
                tir=None,
                tsd=None,
                structure=struct,
                latency_ms=round((t1 - t0) * 1000, 2),
            )

        # Branch 3: IS110 (Recombinase, circular intermediate, no TSD)
        if family == "IS110":
            p_s, p_e, conf, struct = detect_is110_boundary(contig, tpase_start, tpase_end)
            t1 = time.perf_counter()
            return BoundaryPrediction(
                contig_id=contig_id,
                is_name=is_name,
                family=family,
                method="plan_a",
                predicted_start=p_s,
                predicted_end=p_e,
                predicted_length=p_e - p_s,
                is_complete=True,
                confidence_score=round(conf, 4),
                tir=None,
                tsd=None,
                structure=struct,
                latency_ms=round((t1 - t0) * 1000, 2),
            )

        # Branch 4: Canonical DDE families (IS1, IS3, IS4, IS5, IS630, etc.)
        # Retrieve family geometric priors
        priors = FAMILY_DISTANCE_PRIORS.get(family, (40, 350, 20, 300))
        min_up_dist, max_up_dist, min_down_dist, max_down_dist = priors

        # Dynamic search window centered on Tpase ORF
        up_search_len = min(600, max_up_dist + 150)
        down_search_len = min(600, max_down_dist + 150)

        tirs = find_candidate_tirs(
            contig=contig,
            tpase_start=tpase_start,
            tpase_end=tpase_end,
            upstream_search_len=up_search_len,
            downstream_search_len=down_search_len,
            min_tir_len=8,
            max_tir_len=45,
            min_identity=0.65,
            seed_k=6,
            top_k=8,
        )

        best_score = -100.0
        best_cand = None

        for tir in tirs:
            cur_l = tir.left_start
            cur_r = tir.right_end

            # Must encompass Tpase
            if cur_l > tpase_start or cur_r < tpase_end:
                continue

            dist_up = tpase_start - cur_l
            dist_down = cur_r - tpase_end

            # Geometric prior penalty
            geo_penalty = 0.0
            if dist_up < min_up_dist:
                geo_penalty += (min_up_dist - dist_up) * 0.1
            elif dist_up > max_up_dist:
                geo_penalty += (dist_up - max_up_dist) * 0.08

            if dist_down < min_down_dist:
                geo_penalty += (min_down_dist - dist_down) * 0.1
            elif dist_down > max_down_dist:
                geo_penalty += (dist_down - max_down_dist) * 0.08

            # Scan for flanking TSD
            tsd, refined_l, refined_r = find_candidate_tsd(
                contig=contig,
                left_boundary=cur_l,
                right_boundary=cur_r,
                min_tsd_len=2,
                max_tsd_len=14,
                allow_mismatch=True,
                micro_shift=3,
            )

            # Joint Coupled Score:
            # TIR alignment score + boosted TSD score - geometric penalty
            s_tir = tir.score
            s_tsd = (tsd.score * 2.2) if tsd is not None else 0.0
            composite_score = s_tir + s_tsd - geo_penalty

            if composite_score > best_score:
                best_score = composite_score
                best_cand = (refined_l, refined_r, tir, tsd, composite_score)

        if best_cand is not None and best_score > 5.0:
            ref_l, ref_r, sel_tir, sel_tsd, raw_sc = best_cand
            conf = min(0.98, max(0.40, raw_sc / 65.0))
            is_comp = True if (sel_tir is not None and (conf >= 0.45 or sel_tir.length >= 10 or sel_tsd is not None)) else False
            pred = BoundaryPrediction(
                contig_id=contig_id,
                is_name=is_name,
                family=family,
                method="plan_a",
                predicted_start=ref_l,
                predicted_end=ref_r,
                predicted_length=ref_r - ref_l,
                is_complete=is_comp,
                confidence_score=round(conf, 4),
                tir=sel_tir,
                tsd=sel_tsd,
                structure=None,
                latency_ms=0.0,
            )
        else:
            # Fallback using family-informed median boundaries
            fallback_s = max(0, tpase_start - (min_up_dist + max_up_dist) // 2)
            fallback_e = min(len(contig), tpase_end + (min_down_dist + max_down_dist) // 2)
            pred = BoundaryPrediction(
                contig_id=contig_id,
                is_name=is_name,
                family=family,
                method="plan_a",
                predicted_start=fallback_s,
                predicted_end=fallback_e,
                predicted_length=fallback_e - fallback_s,
                is_complete=False,
                confidence_score=0.25,
                tir=None,
                tsd=None,
                structure=None,
                latency_ms=0.0,
            )

        t1 = time.perf_counter()
        pred.latency_ms = round((t1 - t0) * 1000, 2)
        return pred
