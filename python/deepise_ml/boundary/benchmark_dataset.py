"""Ground-truth benchmark contig generator for IS element boundary evaluation."""

import hashlib
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl
from deepise_ml.boundary.schemas import ContigGroundTruth

# Non-canonical families without standard TIR / TSD
NON_CANONICAL_FAMILIES = {"IS200/IS605", "IS91", "IS110", "IS607"}

# Default TSD lengths by family when missing in annotation
DEFAULT_FAMILY_TSD: Dict[str, int] = {
    "IS1": 9,
    "IS3": 3,
    "IS4": 9,
    "IS5": 4,
    "IS6": 8,
    "IS21": 4,
    "IS30": 2,
    "IS66": 8,
    "IS256": 8,
    "IS481": 6,
    "IS630": 2,
    "IS701": 5,
    "IS982": 8,
    "IS1182": 8,
    "IS1380": 4,
    "IS1595": 8,
}


def parse_primary_orf(orf_annotation: Optional[str], element_length: int) -> Tuple[int, int, str]:
    """Parse the primary (longest) transposase ORF within the IS element (0-indexed)."""
    if not orf_annotation:
        # Fallback: assume central 70% is coding
        s = int(element_length * 0.15)
        e = int(element_length * 0.85)
        return s, e, "+"

    matches = re.findall(r"(\d+)\s*\(\s*(\d+)\s*-\s*(\d+)\s*\)", orf_annotation)
    if not matches:
        s = int(element_length * 0.15)
        e = int(element_length * 0.85)
        return s, e, "+"

    best_len = -1
    best_orf = (0, element_length, "+")
    for m in matches:
        aa_len = int(m[0])
        s = int(m[1])
        e = int(m[2])
        start0 = max(0, min(s, e) - 1)
        end0 = min(element_length, max(s, e))
        strand = "+" if s <= e else "-"
        if aa_len > best_len:
            best_len = aa_len
            best_orf = (start0, end0, strand)
    return best_orf


def parse_tir_annotation(tir_str: Optional[str]) -> Optional[int]:
    """Extract annotated TIR length from metadata (e.g. '24/27' -> 27, '16' -> 16)."""
    if not tir_str or tir_str.lower() in ("none", "na", "null", "0"):
        return None
    m = re.search(r"(\d+)\s*/\s*(\d+)", tir_str)
    if m:
        return int(m.group(2))
    m = re.search(r"(\d+)", tir_str)
    if m:
        val = int(m.group(1))
        return val if val > 3 else None
    return None


def parse_dr_annotation(dr_str: Optional[str], family: str) -> Optional[int]:
    """Extract annotated DR/TSD length from metadata."""
    if family in NON_CANONICAL_FAMILIES:
        return None
    if dr_str and dr_str.lower() not in ("none", "na", "null", "0"):
        m = re.search(r"(\d+)", dr_str)
        if m:
            val = int(m.group(1))
            if 2 <= val <= 20:
                return val
    return DEFAULT_FAMILY_TSD.get(family, 4)


def generate_flank_sequence(length: int, gc_content: float, rng: random.Random) -> str:
    """Generate deterministic flanking sequence matching specific GC content."""
    g_or_c = gc_content / 2.0
    a_or_t = (1.0 - gc_content) / 2.0
    choices = ["A", "C", "G", "T"]
    weights = [a_or_t, g_or_c, g_or_c, a_or_t]
    return "".join(rng.choices(choices, weights=weights, k=length))


def compute_gc(seq: str) -> float:
    """Calculate GC content of a nucleotide string."""
    seq_u = seq.upper()
    gc = seq_u.count("G") + seq_u.count("C")
    total = len(seq_u)
    return gc / total if total > 0 else 0.5


def build_benchmark_contigs(
    elements_parquet: Path,
    test_split_parquet: Path,
    val_split_parquet: Path,
    tpases_parquet: Path,
    output_parquet: Path,
    max_elements: Optional[int] = 400,
    flank_length: int = 600,
    seed: int = 42,
) -> pl.DataFrame:
    """Construct benchmark contigs with verified ground-truth boundaries."""
    elems = pl.read_parquet(elements_parquet)
    tpases = pl.read_parquet(tpases_parquet)
    test_df = pl.read_parquet(test_split_parquet)
    val_df = pl.read_parquet(val_split_parquet)

    # 1. Identify test elements (zero homology leakage to train)
    test_pos = test_df.filter(pl.col("label") == 1)
    joined_test = test_pos.join(tpases, left_on="seq_id", right_on="tpase_id", how="inner")
    test_names = set(joined_test["is_name"].unique())

    # 2. Identify validation non-canonical elements (e.g. IS200/IS605)
    val_pos_special = val_df.filter((pl.col("label") == 1) & (pl.col("family") == "IS200/IS605"))
    joined_val = val_pos_special.join(tpases, left_on="seq_id", right_on="tpase_id", how="inner")
    val_names = set(joined_val["is_name"].unique())

    # Combine names
    all_names = list(test_names.union(val_names))
    target_elems = elems.filter(pl.col("is_name").is_in(all_names)).sort("is_id")

    # Sample balanced across families if max_elements is set
    if max_elements and len(target_elems) > max_elements:
        # Group by family and sample proportionally
        sampled_rows = []
        families = target_elems["family"].unique().to_list()
        # Ensure special non-canonical families get solid representation
        per_fam_limit = max(10, max_elements // len(families))
        rng_sample = random.Random(seed)
        for fam in families:
            fam_df = target_elems.filter(pl.col("family") == fam)
            n_sample = min(len(fam_df), per_fam_limit if fam not in NON_CANONICAL_FAMILIES else 60)
            chosen_indices = rng_sample.sample(range(len(fam_df)), n_sample)
            sampled_rows.append(fam_df[chosen_indices])
        target_elems = pl.concat(sampled_rows)

    records: List[Dict] = []
    contig_idx = 0

    for row in target_elems.iter_rows(named=True):
        contig_idx += 1
        is_name = row["is_name"]
        family = row["family"]
        is_seq = row["dna_sequence"].upper().strip()
        is_len = len(is_seq)
        if is_len < 200 or is_len > 10000:
            continue

        is_canonical = family not in NON_CANONICAL_FAMILIES

        # Deterministic RNG for this element
        elem_seed = int(hashlib.md5(f"{is_name}_{seed}".encode()).hexdigest()[:8], 16)
        rng = random.Random(elem_seed)

        # ORF coordinates
        orf_s, orf_e, orf_strand = parse_primary_orf(row.get("orf_annotation"), is_len)

        # Flanking DNA
        gc = compute_gc(is_seq)
        flank_up = generate_flank_sequence(flank_length, gc, rng)
        flank_down = generate_flank_sequence(flank_length, gc, rng)

        # TSD logic
        tsd_len = parse_dr_annotation(row.get("dr_length"), family) if is_canonical else None
        tir_len = parse_tir_annotation(row.get("tir_left")) if is_canonical else None

        true_tsd_seq = None
        if is_canonical and tsd_len and tsd_len > 0:
            # Generate a realistic TSD sequence
            true_tsd_seq = generate_flank_sequence(tsd_len, gc, rng)
            contig_seq = flank_up + true_tsd_seq + is_seq + true_tsd_seq + flank_down
            true_start = len(flank_up) + tsd_len
            true_end = true_start + is_len
        else:
            # Non-canonical or no TSD
            contig_seq = flank_up + is_seq + flank_down
            true_start = len(flank_up)
            true_end = true_start + is_len
            tsd_len = None

        tpase_start = true_start + orf_s
        tpase_end = true_start + orf_e

        true_tir_seq = None
        if is_canonical and tir_len and tir_len > 0:
            true_tir_seq = is_seq[:tir_len]

        gt = ContigGroundTruth(
            contig_id=f"contig_{contig_idx:05d}_{is_name}",
            is_name=is_name,
            family=family,
            is_canonical=is_canonical,
            contig_length=len(contig_seq),
            contig_sequence=contig_seq,
            true_start=true_start,
            true_end=true_end,
            true_length=is_len,
            tpase_start=tpase_start,
            tpase_end=tpase_end,
            tpase_strand=orf_strand,
            annotated_tir_len=tir_len,
            annotated_tsd_len=tsd_len,
            true_tir_seq=true_tir_seq,
            true_tsd_seq=true_tsd_seq,
        )
        records.append(gt.model_dump())

    out_df = pl.DataFrame(records)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(output_parquet)
    return out_df
