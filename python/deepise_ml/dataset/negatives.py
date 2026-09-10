"""Streaming negative dataset generator from curated Swiss-Prot database."""

import gzip
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple
import polars as pl
from Bio import SeqIO

from deepise_ml.schemas.records import NegativeRecord, compute_sha256


HEADER_REGEX = re.compile(
    r"^>?sp\|(?P<acc>[A-Z0-9]+)\|(?P<entry>[A-Za-z0-9_]+)\s+(?P<desc>.+?)(?:\s+OS=(?P<org>.+?))?(?:\s+OX=(?P<taxid>\d+))?(?:\s+GN=(?P<gn>.+?))?(?:\s+PE=\d+)?(?:\s+SV=\d+)?$"
)


def is_prokaryote_candidate(desc: str, org: str) -> bool:
    """Heuristic check or filter to prioritize prokaryotic organisms."""
    # Common bacterial / archaeal indicators or exclude known plants, animals, fungi
    euk_terms = [
        "human", "homo sapiens", "mus musculus", "mouse", "rat", "rattus",
        "drosophila", "arabidopsis", "caenorhabditis", "saccharomyces", "yeast",
        "danio rerio", "zebrafish", "xenopus", "gallus gallus", "chicken"
    ]
    org_lower = org.lower()
    for t in euk_terms:
        if t in org_lower:
            return False
    return True


def stream_sprot_candidates(
    sprot_fasta_gz: Path,
    exclude_keywords: List[str],
    hard_keywords: List[str],
    min_length: int = 80,
    max_length: int = 1500,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Stream Swiss-Prot FASTA and classify into hard, easy, and general candidate pools."""
    hard_pool = []
    easy_pool = []
    general_pool = []

    exclude_lower = [k.lower() for k in exclude_keywords]
    hard_lower = [k.lower() for k in hard_keywords]
    easy_keywords = [
        "ribosomal protein", "atp synthase", "dna-directed rna polymerase",
        "glyceraldehyde-3-phosphate dehydrogenase", "enolase", "elongation factor",
        "tryptophan synthase", "pyruvate kinase", "phosphoglycerate kinase"
    ]

    with gzip.open(sprot_fasta_gz, "rt", encoding="utf-8", errors="replace") as f:
        for record in SeqIO.parse(f, "fasta"):
            desc_full = record.description
            seq_str = str(record.seq).strip().upper()
            length = len(seq_str)
            if length < min_length or length > max_length:
                continue

            m = HEADER_REGEX.match(desc_full)
            if m:
                acc = m.group("acc")
                entry = m.group("entry")
                desc = m.group("desc") or ""
                org = m.group("org") or ""
                taxid = str(m.group("taxid")) if m.group("taxid") else None
            else:
                acc = record.id.split("|")[1] if "|" in record.id else record.id
                entry = record.id.split("|")[2] if record.id.count("|") >= 2 else record.id
                desc = record.description
                org = ""
                taxid = None

            if not is_prokaryote_candidate(desc, org):
                continue

            desc_lower = desc.lower()

            # Exclude transposase / mobile element keywords
            if any(k in desc_lower for k in exclude_lower):
                continue

            entry_dict = {
                "accession": acc,
                "entry_name": entry,
                "description": desc,
                "organism": org,
                "taxonomy_id": taxid,
                "protein_sequence": seq_str,
                "protein_length": length,
            }

            # Classify
            if any(k in desc_lower for k in hard_lower):
                entry_dict["negative_type"] = "hard"
                hard_pool.append(entry_dict)
            elif any(k in desc_lower for k in easy_keywords):
                entry_dict["negative_type"] = "easy"
                easy_pool.append(entry_dict)
            else:
                entry_dict["negative_type"] = "general"
                general_pool.append(entry_dict)

    return hard_pool, easy_pool, general_pool


def sample_negatives(
    hard_pool: List[dict],
    easy_pool: List[dict],
    general_pool: List[dict],
    positive_lengths: List[int],
    target_count: int,
    hard_ratio: float = 0.50,
    matched_ratio: float = 0.30,
    easy_ratio: float = 0.20,
    seed: int = 42,
) -> List[NegativeRecord]:
    """Sample stratified negatives matching target count and positive length distribution."""
    import random
    rng = random.Random(seed)

    target_hard = int(target_count * hard_ratio)
    target_matched = int(target_count * matched_ratio)
    target_easy = target_count - target_hard - target_matched

    # 1. Hard negatives
    rng.shuffle(hard_pool)
    selected_hard = hard_pool[:target_hard]

    # 2. Easy negatives
    rng.shuffle(easy_pool)
    selected_easy = easy_pool[:target_easy]

    # 3. Length-matched negatives from general pool
    rng.shuffle(general_pool)
    selected_matched = []
    used_accs = {x["accession"] for x in selected_hard} | {x["accession"] for x in selected_easy}

    # Group general pool by length bin (bins of 50 aa)
    general_by_bin = defaultdict(list)
    for g in general_pool:
        if g["accession"] not in used_accs:
            b = g["protein_length"] // 50
            general_by_bin[b].append(g)

    # Sample matched to positive length distribution
    shuffled_pos_lengths = list(positive_lengths)
    rng.shuffle(shuffled_pos_lengths)

    pos_idx = 0
    while len(selected_matched) < target_matched and pos_idx < len(shuffled_pos_lengths) * 3:
        target_len = shuffled_pos_lengths[pos_idx % len(shuffled_pos_lengths)]
        pos_idx += 1
        bin_idx = target_len // 50
        candidate = None
        for offset in [0, -1, 1, -2, 2]:
            b = bin_idx + offset
            if general_by_bin[b]:
                candidate = general_by_bin[b].pop()
                break
        if candidate and candidate["accession"] not in used_accs:
            candidate["negative_type"] = "length_matched"
            used_accs.add(candidate["accession"])
            selected_matched.append(candidate)

    # If general pool didn't have enough, fill from remaining general
    if len(selected_matched) < target_matched:
        for g in general_pool:
            if g["accession"] not in used_accs:
                g["negative_type"] = "length_matched"
                used_accs.add(g["accession"])
                selected_matched.append(g)
                if len(selected_matched) >= target_matched:
                    break

    all_selected = selected_hard + selected_easy + selected_matched
    # Sort deterministically
    all_selected.sort(key=lambda x: (x["negative_type"], x["accession"]))

    records = []
    for idx, item in enumerate(all_selected, start=1):
        neg_id = f"neg_{idx:06d}"
        seq = item["protein_sequence"]
        rec = NegativeRecord(
            negative_id=neg_id,
            accession=item["accession"],
            entry_name=item["entry_name"],
            protein_sequence=seq,
            protein_length=item["protein_length"],
            protein_sha256=compute_sha256(seq),
            organism=item["organism"],
            taxonomy_id=item["taxonomy_id"],
            taxonomy_lineage=None,
            negative_type=item["negative_type"],
            description=item["description"],
        )
        records.append(rec)

    return records


def export_negatives(records: List[NegativeRecord], parquet_path: Path, faa_path: Path):
    """Export negative records to Parquet and clean FASTA."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    faa_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [r.model_dump() for r in records]
    df = pl.DataFrame(rows)
    df.write_parquet(parquet_path)

    with open(faa_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(f">{r.negative_id}\n{r.protein_sequence}\n")
