from __future__ import annotations

from typing import Final

from deepise_ml.screening.evidence import (
    FinalCategory,
    FusionEvidence,
    ProstT5Evidence,
    ScreeningWithHomologyRecord,
    StructureEvidence,
)

HIGH_CONFIDENCE_THRESHOLD: Final = 0.70
LOW_CONFIDENCE_THRESHOLD: Final = 0.35


class EvidenceFusionEngine:
    """Multi-modal evidence fusion engine combining sequence, homology, and structural channels."""

    def fuse(
        self,
        homology_record: ScreeningWithHomologyRecord,
        prostt5: ProstT5Evidence,
        structure: StructureEvidence,
    ) -> FusionEvidence:
        stage1 = homology_record.stage1
        evidence = homology_record.evidence

        s_plm = stage1.tpase_score
        has_strong_homology = evidence.has_strong_homology
        has_any_hit = evidence.mmseqs.hit or evidence.hmmer.hit

        # Homology identity & coverage fraction
        identity = (evidence.mmseqs.identity_percent or 0.0) / 100.0
        qcov = (evidence.mmseqs.query_coverage_percent or 0.0) / 100.0
        homology_strength = identity * qcov

        # Structural evidence channel aggregation
        struct_scores: list[float] = []
        if prostt5.evaluated and prostt5.score is not None:
            struct_scores.append(prostt5.score)
        if structure.evaluated and structure.saprot_score is not None:
            struct_scores.append(structure.saprot_score)

        s_struct = (sum(struct_scores) / len(struct_scores)) if struct_scores else None

        # 1. Compute composite Transposase Score
        if s_struct is not None:
            if has_strong_homology:
                raw_score = 0.40 * s_plm + 0.35 * s_struct + 0.20 * homology_strength + 0.08
            elif has_any_hit:
                # Weak/distant homology + structural support
                raw_score = 0.50 * s_plm + 0.42 * s_struct + 0.08 * homology_strength
            else:
                # Novel/orphan regime: zero homology, supported entirely by PLM + structure
                raw_score = 0.50 * s_plm + 0.50 * s_struct
        else:
            # Sequence + Homology only (no structural evaluation)
            if has_strong_homology:
                raw_score = max(s_plm, 0.70 * s_plm + 0.30)
            else:
                raw_score = s_plm

        transposase_score = round(max(0.0, min(1.0, raw_score)), 4)

        # 2. Compute Novelty Score (high confidence + absent/weak sequence homology)
        conf = max(s_plm, s_struct if s_struct is not None else 0.0, transposase_score)
        homology_penalty = min(1.0, homology_strength * 1.2 if has_any_hit else 0.0)
        if has_strong_homology:
            homology_penalty = max(0.85, homology_penalty)

        if transposase_score < LOW_CONFIDENCE_THRESHOLD:
            novelty_score = 0.0
        else:
            raw_novelty = conf * (1.0 - homology_penalty)
            novelty_score = round(max(0.0, min(1.0, raw_novelty)), 4)

        # 3. Final Category Assignment
        if transposase_score >= HIGH_CONFIDENCE_THRESHOLD:
            if has_strong_homology:
                final_cat = FinalCategory.KNOWN_LIKE
                routing_stage = "fusion_known_like"
                notes = "high_confidence_strong_homology"
            elif has_any_hit and (s_struct is not None and s_struct >= 0.55):
                final_cat = FinalCategory.REMOTE
                routing_stage = "fusion_remote"
                notes = "high_confidence_weak_homology_supported"
            elif (not has_any_hit or homology_strength < 0.05) and (s_struct is not None and s_struct >= 0.65):
                final_cat = FinalCategory.NOVEL_CANDIDATE
                routing_stage = "fusion_novel_candidate"
                notes = "high_confidence_structural_support_zero_homology"
            elif s_struct is not None and s_struct >= 0.55:
                final_cat = FinalCategory.REMOTE
                routing_stage = "fusion_remote"
                notes = "high_confidence_structural_support"
            else:
                final_cat = FinalCategory.UNCERTAIN
                routing_stage = "fusion_uncertain"
                notes = "high_score_unverified_structure"
        elif transposase_score < LOW_CONFIDENCE_THRESHOLD:
            if has_strong_homology:
                final_cat = FinalCategory.UNCERTAIN
                routing_stage = "fusion_conflict"
                notes = "plm_negative_homology_positive_conflict"
            else:
                final_cat = FinalCategory.NEGATIVE
                routing_stage = "fusion_negative"
                notes = "low_transposase_confidence"
        else:
            final_cat = FinalCategory.UNCERTAIN
            routing_stage = "fusion_intermediate"
            notes = "intermediate_multimodal_confidence"

        return FusionEvidence(
            transposase_score=transposase_score,
            novelty_score=novelty_score,
            final_category=final_cat,
            routing_stage=routing_stage,
            notes=notes,
        )


__all__ = [
    "EvidenceFusionEngine",
    "HIGH_CONFIDENCE_THRESHOLD",
    "LOW_CONFIDENCE_THRESHOLD",
]
