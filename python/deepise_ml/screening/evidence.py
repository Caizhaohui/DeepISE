from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from deepise_ml.screening.runtime import OUTPUT_COLUMNS, ProteinScreeningRecord

MMSEQS_STRONG_EVALUE: Final = 1e-5
MMSEQS_STRONG_IDENTITY_PERCENT: Final = 30.0
MMSEQS_STRONG_QUERY_COVERAGE_PERCENT: Final = 80.0
HMMER_STRONG_FULL_EVALUE: Final = 1e-5

COMBINED_OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    *OUTPUT_COLUMNS,
    "mmseqs_hit",
    "mmseqs_target_id",
    "mmseqs_identity_percent",
    "mmseqs_alignment_length",
    "mmseqs_query_coverage_percent",
    "mmseqs_target_coverage_percent",
    "mmseqs_evalue",
    "mmseqs_bitscore",
    "hmmer_hit",
    "hmmer_profile_id",
    "hmmer_full_evalue",
    "hmmer_full_bitscore",
    "hmmer_domain_evalue",
    "hmmer_domain_bitscore",
    "route",
    "reason_code",
)

RESEARCH_OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    *COMBINED_OUTPUT_COLUMNS,
    "prostt5_evaluated",
    "prostt5_score",
    "prostt5_structure_similarity",
    "predicted_3di",
    "structure_evaluated",
    "mean_plddt",
    "foldseek_hit",
    "foldseek_target_id",
    "foldseek_score",
    "foldseek_evalue",
    "saprot_score",
    "transposase_score",
    "novelty_score",
)


@dataclass(frozen=True, slots=True)
class HomologyContractError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


class CandidateRoute(StrEnum):
    ACCEPT_KNOWN = "ACCEPT_KNOWN"
    REJECT = "REJECT"
    STAGE2A = "STAGE2A"
    STAGE2B = "STAGE2B"
    UNCERTAIN = "UNCERTAIN"


class FinalCategory(StrEnum):
    KNOWN_LIKE = "Known-like"
    REMOTE = "Remote"
    NOVEL_CANDIDATE = "Novel_candidate"
    UNCERTAIN = "Uncertain"
    NEGATIVE = "Negative"


class RouteReasonCode(StrEnum):
    ACCEPT_KNOWN_HIGH_SCORE_STRONG_HOMOLOGY = "accept_known_high_score_strong_homology"
    REJECT_LOW_SCORE_NO_STRONG_HOMOLOGY = "reject_low_score_no_strong_homology"
    STAGE2A_HIGH_SCORE_WEAK_OR_NO_HOMOLOGY = "stage2a_high_score_weak_or_no_homology"
    STAGE2B_PROSTT5_UNRESOLVED_TWILIGHT = "stage2b_prostt5_unresolved_twilight"
    UNCERTAIN_INTERMEDIATE_SCORE = "uncertain_intermediate_score"
    UNCERTAIN_PLM_HOMOLOGY_CONFLICT = "uncertain_plm_homology_conflict"


class RoutingDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    route: CandidateRoute
    reason_code: RouteReasonCode


class MMseqsEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    hit: bool
    target_id: str | None = Field(default=None, min_length=1)
    identity_percent: float | None = Field(default=None, ge=0.0, le=100.0)
    alignment_length: int | None = Field(default=None, ge=1)
    query_coverage_percent: float | None = Field(default=None, ge=0.0, le=100.0)
    target_coverage_percent: float | None = Field(default=None, ge=0.0, le=100.0)
    evalue: float | None = Field(default=None, ge=0.0)
    bitscore: float | None = None

    @classmethod
    def no_hit(cls) -> Self:
        return cls(hit=False)

    @property
    def is_strong(self) -> bool:
        return (
            self.hit
            and self.evalue is not None
            and self.identity_percent is not None
            and self.query_coverage_percent is not None
            and self.evalue <= MMSEQS_STRONG_EVALUE
            and self.identity_percent >= MMSEQS_STRONG_IDENTITY_PERCENT
            and self.query_coverage_percent >= MMSEQS_STRONG_QUERY_COVERAGE_PERCENT
        )

    @model_validator(mode="after")
    def _validate_hit_shape(self) -> Self:
        fields = (
            self.target_id,
            self.identity_percent,
            self.alignment_length,
            self.query_coverage_percent,
            self.target_coverage_percent,
            self.evalue,
            self.bitscore,
        )
        if self.hit and any(value is None for value in fields):
            raise ValueError("MMseqs hit evidence requires complete best-hit fields")
        if not self.hit and any(value is not None for value in fields):
            raise ValueError("no-hit MMseqs evidence must contain only null evidence fields")
        return self


class HMMEREvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    hit: bool
    profile_id: str | None = Field(default=None, min_length=1)
    full_evalue: float | None = Field(default=None, ge=0.0)
    full_bitscore: float | None = None
    domain_evalue: float | None = Field(default=None, ge=0.0)
    domain_bitscore: float | None = None

    @classmethod
    def no_hit(cls) -> Self:
        return cls(hit=False)

    @property
    def is_strong(self) -> bool:
        return (
            self.hit
            and self.full_evalue is not None
            and self.full_evalue <= HMMER_STRONG_FULL_EVALUE
        )

    @model_validator(mode="after")
    def _validate_hit_shape(self) -> Self:
        no_hit_fields = (
            self.profile_id,
            self.full_evalue,
            self.full_bitscore,
            self.domain_evalue,
            self.domain_bitscore,
        )
        if self.hit and (
            self.profile_id is None
            or self.full_evalue is None
            or self.full_bitscore is None
        ):
            raise ValueError("HMMER hit evidence requires profile and full-sequence fields")
        if not self.hit and any(value is not None for value in no_hit_fields):
            raise ValueError("no-hit HMMER evidence must contain only null evidence fields")
        return self


class HomologyEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    protein_id: str = Field(min_length=1)
    mmseqs: MMseqsEvidence
    hmmer: HMMEREvidence

    @property
    def has_strong_homology(self) -> bool:
        return self.mmseqs.is_strong or self.hmmer.is_strong


@dataclass(frozen=True, slots=True)
class HomologyEvidenceSet:
    records: tuple[HomologyEvidence, ...]

    def __post_init__(self) -> None:
        protein_ids = tuple(record.protein_id for record in self.records)
        if len(protein_ids) != len(set(protein_ids)):
            raise HomologyContractError("duplicate homology evidence protein IDs")


@dataclass(frozen=True, slots=True)
class ScreeningWithHomologyRecord:
    stage1: ProteinScreeningRecord
    evidence: HomologyEvidence
    decision: RoutingDecision

    @classmethod
    def from_stage1(
        cls,
        stage1: ProteinScreeningRecord,
        evidence: HomologyEvidence,
        decision: RoutingDecision,
    ) -> Self:
        if stage1.protein_id != evidence.protein_id:
            raise HomologyContractError("Stage-1 and homology evidence protein IDs do not match")
        return cls(stage1=stage1, evidence=evidence, decision=decision)

    def tsv_row(self) -> Mapping[str, str | int | float | bool | None]:
        return {
            **self.stage1.tsv_row(),
            "mmseqs_hit": self.evidence.mmseqs.hit,
            "mmseqs_target_id": self.evidence.mmseqs.target_id,
            "mmseqs_identity_percent": self.evidence.mmseqs.identity_percent,
            "mmseqs_alignment_length": self.evidence.mmseqs.alignment_length,
            "mmseqs_query_coverage_percent": self.evidence.mmseqs.query_coverage_percent,
            "mmseqs_target_coverage_percent": self.evidence.mmseqs.target_coverage_percent,
            "mmseqs_evalue": self.evidence.mmseqs.evalue,
            "mmseqs_bitscore": self.evidence.mmseqs.bitscore,
            "hmmer_hit": self.evidence.hmmer.hit,
            "hmmer_profile_id": self.evidence.hmmer.profile_id,
            "hmmer_full_evalue": self.evidence.hmmer.full_evalue,
            "hmmer_full_bitscore": self.evidence.hmmer.full_bitscore,
            "hmmer_domain_evalue": self.evidence.hmmer.domain_evalue,
            "hmmer_domain_bitscore": self.evidence.hmmer.domain_bitscore,
            "route": self.decision.route.value,
            "reason_code": self.decision.reason_code.value,
        }


def parse_mmseqs_evidence(
    *,
    hit: bool,
    target_id: str | None,
    identity_percent: float | None,
    alignment_length: int | None,
    query_coverage_percent: float | None,
    target_coverage_percent: float | None,
    evalue: float | None,
    bitscore: float | None,
) -> MMseqsEvidence:
    try:
        return MMseqsEvidence(
            hit=hit,
            target_id=target_id,
            identity_percent=identity_percent,
            alignment_length=alignment_length,
            query_coverage_percent=query_coverage_percent,
            target_coverage_percent=target_coverage_percent,
            evalue=evalue,
            bitscore=bitscore,
        )
    except ValidationError as error:
        raise HomologyContractError(f"malformed MMseqs evidence: {error}") from error


def parse_hmmer_evidence(
    *,
    hit: bool,
    profile_id: str | None,
    full_evalue: float | None,
    full_bitscore: float | None,
    domain_evalue: float | None,
    domain_bitscore: float | None,
) -> HMMEREvidence:
    try:
        return HMMEREvidence(
            hit=hit,
            profile_id=profile_id,
            full_evalue=full_evalue,
            full_bitscore=full_bitscore,
            domain_evalue=domain_evalue,
            domain_bitscore=domain_bitscore,
        )
    except ValidationError as error:
        raise HomologyContractError(f"malformed HMMER evidence: {error}") from error


class ProstT5Evidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    evaluated: bool = False
    predicted_3di: str | None = None
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    structure_similarity: float | None = Field(default=None, ge=0.0, le=1.0)

    @classmethod
    def not_evaluated(cls) -> Self:
        return cls(evaluated=False)


class StructureEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    evaluated: bool = False
    pdb_path: str | None = None
    mean_plddt: float | None = Field(default=None, ge=0.0, le=100.0)
    foldseek_hit: bool = False
    foldseek_target_id: str | None = None
    foldseek_score: float | None = None
    foldseek_evalue: float | None = Field(default=None, ge=0.0)
    saprot_score: float | None = Field(default=None, ge=0.0, le=1.0)

    @classmethod
    def not_evaluated(cls) -> Self:
        return cls(evaluated=False)


class FusionEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    transposase_score: float = Field(ge=0.0, le=1.0)
    novelty_score: float = Field(ge=0.0, le=1.0)
    final_category: FinalCategory
    routing_stage: str
    notes: str


@dataclass(frozen=True, slots=True)
class ResearchScreeningRecord:
    homology_record: ScreeningWithHomologyRecord
    prostt5: ProstT5Evidence
    structure: StructureEvidence
    fusion: FusionEvidence

    @property
    def protein_id(self) -> str:
        return self.homology_record.stage1.protein_id

    def tsv_row(self) -> Mapping[str, str | int | float | bool | None]:
        homology_dict = dict(self.homology_record.tsv_row())
        homology_dict["final_category"] = self.fusion.final_category.value
        homology_dict["routing_stage"] = self.fusion.routing_stage
        homology_dict["notes"] = self.fusion.notes
        return {
            **homology_dict,
            "prostt5_evaluated": self.prostt5.evaluated,
            "prostt5_score": self.prostt5.score,
            "prostt5_structure_similarity": self.prostt5.structure_similarity,
            "predicted_3di": self.prostt5.predicted_3di,
            "structure_evaluated": self.structure.evaluated,
            "mean_plddt": self.structure.mean_plddt,
            "foldseek_hit": self.structure.foldseek_hit,
            "foldseek_target_id": self.structure.foldseek_target_id,
            "foldseek_score": self.structure.foldseek_score,
            "foldseek_evalue": self.structure.foldseek_evalue,
            "saprot_score": self.structure.saprot_score,
            "transposase_score": self.fusion.transposase_score,
            "novelty_score": self.fusion.novelty_score,
        }

