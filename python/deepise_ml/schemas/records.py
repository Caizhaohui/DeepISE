"""Canonical data schemas for DeepISE dataset components."""

import hashlib
from typing import Optional, Union
from pydantic import BaseModel, Field


def compute_sha256(sequence: str) -> str:
    """Compute normalized SHA256 hex digest for sequence string."""
    clean_seq = "".join(sequence.strip().upper().split())
    return hashlib.sha256(clean_seq.encode("utf-8")).hexdigest()


class CanonicalTpaseRecord(BaseModel):
    """Canonical record representing a transposase protein."""
    tpase_id: str = Field(description="Unique deterministic ID, e.g., tpase_000001")
    is_name: str = Field(description="Source IS element name, e.g., IS1A")
    raw_header: str = Field(default="", description="Original FASTA header")
    family: str = Field(description="IS family annotation, e.g., IS1")
    group: Optional[str] = Field(default=None, description="IS group annotation")
    subgroup: Optional[str] = Field(default=None, description="IS subgroup annotation")
    protein_sequence: str = Field(description="Amino acid sequence")
    protein_length: int = Field(description="Length in amino acids")
    protein_sha256: str = Field(description="SHA-256 hash of protein sequence")
    host_organism: Optional[str] = Field(default=None, description="Host organism name")
    accession: Optional[str] = Field(default=None, description="GenBank / EMBL accession")
    source: str = Field(default="ISfinder", description="Data source name")
    is_transposase: bool = Field(default=True, description="Whether confirmed as transposase")
    exact_cluster_id: Optional[str] = Field(default=None, description="100% identity deduplication cluster ID")
    cluster30_id: Optional[str] = Field(default=None, description="30% identity MMseqs2 cluster ID")
    split: Optional[str] = Field(default=None, description="Dataset split (train/validation/test)")
    annotation_conflict: bool = Field(default=False, description="True if identical sequence has conflicting families")


class CanonicalISRecord(BaseModel):
    """Canonical record representing a full nucleotide IS element."""
    is_id: str = Field(description="Unique deterministic ID, e.g., is_000001")
    is_name: str = Field(description="IS element name, e.g., IS1A")
    family: str = Field(description="IS family annotation")
    group: Optional[str] = Field(default=None, description="IS group annotation")
    dna_sequence: str = Field(description="Nucleotide sequence")
    dna_length: int = Field(description="Element length in base pairs")
    dna_sha256: str = Field(description="SHA-256 hash of DNA sequence")
    tir_left: Optional[str] = Field(default=None, description="Left terminal inverted repeat annotation")
    tir_right: Optional[str] = Field(default=None, description="Right terminal inverted repeat annotation")
    dr_length: Optional[str] = Field(default=None, description="Direct repeat / target site duplication annotation")
    orf_annotation: Optional[str] = Field(default=None, description="Raw ORF description from metadata")
    host_organism: Optional[str] = Field(default=None, description="Host organism name")
    accession: Optional[str] = Field(default=None, description="GenBank accession")
    source: str = Field(default="ISfinder", description="Data source name")
    exact_cluster_id: Optional[str] = Field(default=None, description="100% identity deduplication cluster ID")
    split: Optional[str] = Field(default=None, description="Dataset split")


class NegativeRecord(BaseModel):
    """Record representing a negative (non-transposase) protein."""
    negative_id: str = Field(description="Unique deterministic ID, e.g., neg_000001")
    accession: str = Field(description="UniProt / Swiss-Prot accession, e.g., P0A8T7")
    entry_name: Optional[str] = Field(default=None, description="UniProt entry name")
    protein_sequence: str = Field(description="Amino acid sequence")
    protein_length: int = Field(description="Length in amino acids")
    protein_sha256: str = Field(description="SHA-256 hash of protein sequence")
    organism: Optional[str] = Field(default=None, description="Organism scientific name")
    taxonomy_id: Optional[Union[str, int]] = Field(default=None, description="NCBI Taxonomy ID")
    taxonomy_lineage: Optional[str] = Field(default=None, description="Taxonomic lineage")
    negative_type: str = Field(description="easy | length_matched | taxonomy_matched | hard | ambiguous")
    description: Optional[str] = Field(default=None, description="Protein functional description")
    exact_cluster_id: Optional[str] = Field(default=None, description="100% identity deduplication cluster ID")
    cluster30_id: Optional[str] = Field(default=None, description="30% identity MMseqs2 cluster ID")
    split: Optional[str] = Field(default=None, description="Dataset split")
