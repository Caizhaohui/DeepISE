"""Pydantic data models and schemas for Phase-2 boundary detection."""

from typing import Optional
from pydantic import BaseModel, Field


class TIRMatch(BaseModel):
    """Terminal Inverted Repeat (TIR) match details."""
    left_start: int = Field(..., description="0-indexed start coordinate of left TIR in contig")
    left_end: int = Field(..., description="0-indexed end coordinate of left TIR in contig (exclusive)")
    right_start: int = Field(..., description="0-indexed start coordinate of right TIR in contig")
    right_end: int = Field(..., description="0-indexed end coordinate of right TIR in contig (exclusive)")
    left_seq: str = Field(..., description="Nucleotide sequence of left TIR")
    right_seq: str = Field(..., description="Nucleotide sequence of right TIR (forward orientation in contig)")
    length: int = Field(..., description="Aligned length of TIR")
    mismatches: int = Field(0, description="Number of mismatches in TIR alignment")
    gaps: int = Field(0, description="Number of indels in TIR alignment")
    identity: float = Field(..., description="Identity ratio (matches / length)")
    score: float = Field(..., description="Alignment score")


class TSDMatch(BaseModel):
    """Target Site Duplication (TSD) direct repeat match details."""
    left_start: int = Field(..., description="0-indexed start of 5' TSD in contig")
    left_end: int = Field(..., description="0-indexed end of 5' TSD in contig (exclusive)")
    right_start: int = Field(..., description="0-indexed start of 3' TSD in contig")
    right_end: int = Field(..., description="0-indexed end of 3' TSD in contig (exclusive)")
    sequence: str = Field(..., description="Consensus sequence of the duplicated target site")
    length: int = Field(..., description="Length of TSD (typically 2 to 14 bp)")
    mismatches: int = Field(0, description="Mismatches between left and right copy (0 or 1)")
    score: float = Field(..., description="TSD confidence score")


class StructureEvidence(BaseModel):
    """Structural/motif evidence for non-TIR special families (Plan A)."""
    feature_type: str = Field(..., description="Feature type: hairpin_stem_loop, is91_ori_ter, is110_recombination")
    motif_left: Optional[str] = Field(None, description="Conserved 5' motif or ori sequence")
    motif_right: Optional[str] = Field(None, description="Conserved 3' motif or ter sequence")
    hairpin_left: Optional[str] = Field(None, description="Upstream stem-loop secondary structure")
    hairpin_right: Optional[str] = Field(None, description="Downstream stem-loop secondary structure")
    stability_score: float = Field(0.0, description="Stem-loop pairing stability score / motif match score")
    notes: str = Field("", description="Diagnostic notes")


class BoundaryPrediction(BaseModel):
    """Comprehensive IS element boundary prediction output."""
    contig_id: str
    is_name: str
    family: str
    method: str = Field(..., description="'plan_a' or 'plan_b'")
    predicted_start: int = Field(..., description="0-indexed predicted start of IS element in contig")
    predicted_end: int = Field(..., description="0-indexed predicted end of IS element in contig (exclusive)")
    predicted_length: int = Field(..., description="Length of predicted IS element")
    is_complete: bool = Field(True, description="Whether predicted IS element is structurally complete")
    confidence_score: float = Field(..., description="Overall confidence score [0.0 - 1.0]")
    tir: Optional[TIRMatch] = None
    tsd: Optional[TSDMatch] = None
    structure: Optional[StructureEvidence] = None
    latency_ms: float = Field(0.0, description="Elapsed prediction time in milliseconds")


class ContigGroundTruth(BaseModel):
    """Ground-truth benchmark contig containing an embedded IS element."""
    contig_id: str
    is_name: str
    family: str
    is_canonical: bool = Field(..., description="True if canonical DDE family, False if non-canonical")
    contig_length: int
    contig_sequence: str
    true_start: int = Field(..., description="0-indexed start of true IS element in contig")
    true_end: int = Field(..., description="0-indexed end of true IS element in contig (exclusive)")
    true_length: int
    tpase_start: int = Field(..., description="0-indexed start of Tpase ORF in contig")
    tpase_end: int = Field(..., description="0-indexed end of Tpase ORF in contig (exclusive)")
    tpase_strand: str = Field("+", description="'+' or '-'")
    annotated_tir_len: Optional[int] = None
    annotated_tsd_len: Optional[int] = None
    true_tir_seq: Optional[str] = None
    true_tsd_seq: Optional[str] = None
