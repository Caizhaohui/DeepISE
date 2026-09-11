"""Full-Length IS Element Composite Scorer for DeepISE."""

from typing import Dict, Optional, Tuple
from pydantic import BaseModel, Field
from deepise_ml.boundary.schemas import BoundaryPrediction

NON_CANONICAL_FAMILIES = {"IS200/IS605", "IS91", "IS110", "IS607"}

# Typical element length ranges by family (min_expected, max_expected)
FAMILY_LENGTH_EXPECTATIONS = {
    "IS1": (650, 1200),
    "IS3": (1100, 1600),
    "IS4": (1200, 1800),
    "IS5": (800, 1600),
    "IS6": (750, 1100),
    "IS21": (1800, 3200),
    "IS30": (1000, 1400),
    "IS66": (2200, 3500),
    "IS91": (1500, 2200),
    "IS110": (1100, 1600),
    "IS200/IS605": (600, 2000),
    "IS256": (1200, 1600),
    "IS481": (900, 1300),
    "IS630": (1000, 1400),
    "IS701": (1200, 1700),
    "IS982": (900, 1200),
    "IS1182": (1300, 1900),
    "IS1380": (1400, 2000),
    "IS1595": (650, 1000),
}


class ISCompositeResult(BaseModel):
    """Calibrated composite evaluation result for an IS element candidate."""
    composite_score: float = Field(..., description="Overall calibrated score [0.0 - 1.0]")
    status: str = Field(..., description="'complete', 'partial', or 'pseudo'")
    tpase_score: float
    boundary_score: float
    tsd_score: float
    architecture_score: float
    length_score: float
    coding_density: float
    notes: str = ""


class ISCompositeScorer:
    """Calculates multi-evidence composite score for full-length IS candidates."""

    def __init__(
        self,
        weight_tpase: float = 0.35,
        weight_boundary: float = 0.25,
        weight_tsd: float = 0.15,
        weight_architecture: float = 0.15,
        weight_length: float = 0.10,
    ):
        self.w_tpase = weight_tpase
        self.w_boundary = weight_boundary
        self.w_tsd = weight_tsd
        self.w_arch = weight_architecture
        self.w_len = weight_length

    def score_element(
        self,
        boundary_pred: BoundaryPrediction,
        tpase_score: float,
        tpase_length_bp: int,
    ) -> ISCompositeResult:
        """Score an IS element candidate based on joint biological evidence."""
        elem_len = max(1, boundary_pred.predicted_length)
        family = boundary_pred.family
        is_canonical = family not in NON_CANONICAL_FAMILIES

        # 1. Transposase PLM score
        s_tpase = min(1.0, max(0.0, tpase_score))

        # 2. Boundary score
        if is_canonical:
            if boundary_pred.tir is not None:
                tir_len_factor = min(1.0, boundary_pred.tir.length / 20.0)
                s_boundary = boundary_pred.tir.identity * tir_len_factor
            else:
                s_boundary = 0.15
        else:
            if boundary_pred.structure is not None:
                s_boundary = min(1.0, max(0.2, boundary_pred.structure.stability_score / 22.0))
            else:
                s_boundary = 0.20

        # 3. TSD score
        if is_canonical:
            if boundary_pred.tsd is not None:
                s_tsd = min(1.0, boundary_pred.tsd.score / 18.0)
            else:
                s_tsd = 0.05
        else:
            # Biology: non-canonical families do not generate TSDs, so full neutral score
            s_tsd = 1.0

        # 4. ORF Architecture / Coding density score
        coding_density = min(1.0, tpase_length_bp / elem_len)
        if 0.50 <= coding_density <= 0.95:
            s_arch = 1.0
        elif 0.35 <= coding_density < 0.50:
            s_arch = 0.70
        elif coding_density > 0.95:
            s_arch = 0.85
        else:
            s_arch = 0.40

        # 5. Length expectation score
        min_exp, max_exp = FAMILY_LENGTH_EXPECTATIONS.get(family, (700, 3200))
        if min_exp <= elem_len <= max_exp:
            s_len = 1.0
        elif min_exp * 0.7 <= elem_len <= max_exp * 1.3:
            s_len = 0.75
        elif min_exp * 0.5 <= elem_len <= max_exp * 1.8:
            s_len = 0.50
        else:
            s_len = 0.25

        # Weighted composite score
        comp = (
            self.w_tpase * s_tpase
            + self.w_boundary * s_boundary
            + self.w_tsd * s_tsd
            + self.w_arch * s_arch
            + self.w_len * s_len
        )
        comp = round(min(1.0, max(0.0, comp)), 4)

        # Status determination
        if comp >= 0.65 and boundary_pred.is_complete:
            status = "complete"
        elif comp >= 0.38:
            status = "partial"
        else:
            status = "pseudo"

        notes_parts = []
        if is_canonical and boundary_pred.tir is not None:
            notes_parts.append(f"TIR:{boundary_pred.tir.length}bp")
        if is_canonical and boundary_pred.tsd is not None:
            notes_parts.append(f"TSD:{boundary_pred.tsd.sequence}")
        if not is_canonical and boundary_pred.structure is not None:
            notes_parts.append(f"Structure:{boundary_pred.structure.feature_type}")

        return ISCompositeResult(
            composite_score=comp,
            status=status,
            tpase_score=round(s_tpase, 4),
            boundary_score=round(s_boundary, 4),
            tsd_score=round(s_tsd, 4),
            architecture_score=round(s_arch, 4),
            length_score=round(s_len, 4),
            coding_density=round(coding_density, 3),
            notes="; ".join(notes_parts),
        )
