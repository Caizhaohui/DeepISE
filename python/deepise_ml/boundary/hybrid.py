"""Hybrid Physics x Neural Boundary Engine (Phase-3).

Combines Plan A full-family adaptive physical heuristics with deep learning
local junction refinement (1D Dilated Residual CNN) to resolve fuzzy or
degenerate terminal boundaries.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from deepise_ml.boundary.neural_refiner import NeuralBoundaryRefiner
from deepise_ml.boundary.plan_a import PlanAAdaptiveEngine
from deepise_ml.boundary.schemas import BoundaryPrediction, StructureEvidence


class HybridBoundaryEngine:
    """Hybrid boundary inference coupling physics heuristics with neural refinement."""

    def __init__(
        self,
        neural_model_path: Optional[Path] = Path("benchmark/models/neural_boundary_refiner.pt"),
        high_confidence_thresh: float = 0.82,
        device: Optional[str] = None,
    ):
        self.plan_a = PlanAAdaptiveEngine()
        self.high_confidence_thresh = high_confidence_thresh
        
        # Load neural refiner if weights available
        if neural_model_path and neural_model_path.exists():
            self.refiner = NeuralBoundaryRefiner(model_path=neural_model_path, device=device)
        else:
            self.refiner = None

    def predict_boundary(
        self,
        contig: str,
        tpase_start: int,
        tpase_end: int,
        contig_id: str = "contig",
        is_name: str = "unknown",
        family: str = "IS3",
        max_flank: int = 1500,
    ) -> BoundaryPrediction:
        """Predict IS element boundaries using hybrid physical + deep learning strategy."""
        t0 = time.perf_counter()

        # 1. Step 1: Obtain adaptive physical candidate from Plan A
        phys_pred = self.plan_a.predict_boundary(
            contig=contig,
            tpase_start=tpase_start,
            tpase_end=tpase_end,
            contig_id=contig_id,
            is_name=is_name,
            family=family,
        )

        # 2. Step 2: Determine if neural refinement is required
        # Non-canonical families (IS110, IS200, IS91) use specific biological core motifs
        is_special = family in ["IS110", "IS492", "IS200/IS605", "IS607", "IS91"]
        is_high_conf = (
            phys_pred.confidence_score >= self.high_confidence_thresh
            and phys_pred.tir is not None
            and phys_pred.tsd is not None
            and phys_pred.tir.length >= 15
        )

        # If already highly confident or special family or no neural refiner, retain Plan A
        if is_high_conf or is_special or self.refiner is None:
            return phys_pred

        # 3. Step 3: Neural local context refinement on fuzzy boundaries
        orig_s = phys_pred.predicted_start
        orig_e = phys_pred.predicted_end

        ref_s, off_s, conf_s = self.refiner.refine_boundary(
            contig_seq=contig,
            cand_coord=orig_s,
            is_five_prime=True,
            max_shift=25,
        )
        ref_e, off_e, conf_e = self.refiner.refine_boundary(
            contig_seq=contig,
            cand_coord=orig_e,
            is_five_prime=False,
            max_shift=25,
        )

        # Ensure valid physical length
        if ref_e - ref_s < 300:
            return phys_pred

        # Construct updated hybrid boundary prediction
        hybrid_confidence = float(0.7 * phys_pred.confidence_score + 0.3 * ((conf_s + conf_e) / 2.0))
        t1 = time.perf_counter()
        
        struct_note = (
            f"NeuralRefine(5'={off_s:+.0f}bp, 3'={off_e:+.0f}bp, conf={(conf_s+conf_e)/2.0:.2f})"
        )
        structure_ev = (
            phys_pred.structure
            if phys_pred.structure
            else StructureEvidence(
                feature_type="neural_refined_junction",
                notes=struct_note,
            )
        )

        return BoundaryPrediction(
            contig_id=phys_pred.contig_id,
            is_name=phys_pred.is_name,
            family=phys_pred.family,
            method="plan_a_hybrid_neural",
            predicted_start=ref_s,
            predicted_end=ref_e,
            predicted_length=ref_e - ref_s,
            is_complete=phys_pred.is_complete,
            confidence_score=min(1.0, hybrid_confidence),
            tir=phys_pred.tir,
            tsd=phys_pred.tsd,
            structure=structure_ev,
            latency_ms=phys_pred.latency_ms + (t1 - t0) * 1000.0,
        )
