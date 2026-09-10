"""Exact sequence deduplication module for DeepISE datasets."""

from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict
import polars as pl

from deepise_ml.schemas.records import CanonicalTpaseRecord, CanonicalISRecord


def deduplicate_tpases(
    records: List[CanonicalTpaseRecord],
) -> Tuple[List[CanonicalTpaseRecord], pl.DataFrame]:
    """Perform 100% exact sequence deduplication on transposase records.
    
    Returns:
        representatives: list of unique canonical records (one per unique sequence)
        cluster_df: DataFrame mapping representative_id -> member_ids, cluster_size, annotation_conflict
    """
    groups_by_seq = defaultdict(list)
    for rec in records:
        groups_by_seq[rec.protein_sha256].append(rec)

    representatives = []
    cluster_rows = []

    # Deterministic order
    sorted_hashes = sorted(groups_by_seq.keys())
    for cluster_idx, sha in enumerate(sorted_hashes, start=1):
        members = groups_by_seq[sha]
        rep_id = f"tpase_exact_{cluster_idx:06d}"

        # Check for family conflicts
        families = {m.family for m in members if m.family}
        has_conflict = len(families) > 1

        # Select primary representative: prioritize member with known family and accession
        def rep_score(m: CanonicalTpaseRecord):
            has_fam = 1 if m.family and m.family != "Unknown" else 0
            has_acc = 1 if m.accession else 0
            return (has_fam, has_acc, -len(m.is_name), m.is_name)

        sorted_members = sorted(members, key=rep_score, reverse=True)
        primary = sorted_members[0]

        # Update representative record
        primary.exact_cluster_id = rep_id
        primary.annotation_conflict = has_conflict
        representatives.append(primary)

        for m in members:
            cluster_rows.append({
                "exact_cluster_id": rep_id,
                "representative_tpase_id": primary.tpase_id,
                "member_tpase_id": m.tpase_id,
                "is_name": m.is_name,
                "family": m.family,
                "cluster_size": len(members),
                "annotation_conflict": has_conflict,
                "protein_sha256": sha,
            })

    cluster_df = pl.DataFrame(cluster_rows)
    return representatives, cluster_df


def deduplicate_is_elements(
    records: List[CanonicalISRecord],
) -> Tuple[List[CanonicalISRecord], pl.DataFrame]:
    """Perform 100% exact sequence deduplication on IS element DNA sequences."""
    groups_by_seq = defaultdict(list)
    for rec in records:
        groups_by_seq[rec.dna_sha256].append(rec)

    representatives = []
    cluster_rows = []

    sorted_hashes = sorted(groups_by_seq.keys())
    for cluster_idx, sha in enumerate(sorted_hashes, start=1):
        members = groups_by_seq[sha]
        rep_id = f"is_exact_{cluster_idx:06d}"

        # Choose primary representative
        def rep_score(m: CanonicalISRecord):
            has_fam = 1 if m.family and m.family != "Unknown" else 0
            has_ir = 1 if m.tir_left else 0
            return (has_fam, has_ir, m.is_name)

        sorted_members = sorted(members, key=rep_score, reverse=True)
        primary = sorted_members[0]
        primary.exact_cluster_id = rep_id
        representatives.append(primary)

        for m in members:
            cluster_rows.append({
                "exact_cluster_id": rep_id,
                "representative_is_id": primary.is_id,
                "member_is_id": m.is_id,
                "is_name": m.is_name,
                "family": m.family,
                "cluster_size": len(members),
                "dna_sha256": sha,
            })

    cluster_df = pl.DataFrame(cluster_rows)
    return representatives, cluster_df


def export_fasta(records: List[CanonicalTpaseRecord], output_path: Path):
    """Export clean FASTA with simple IDs according to FASTA ID rules."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            # Rule: >tpase_000001 (no spaces, no pipes)
            f.write(f">{rec.tpase_id}\n{rec.protein_sequence}\n")


def export_dna_fasta(records: List[CanonicalISRecord], output_path: Path):
    """Export clean DNA FASTA for IS elements."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(f">{rec.is_id}\n{rec.dna_sequence}\n")
