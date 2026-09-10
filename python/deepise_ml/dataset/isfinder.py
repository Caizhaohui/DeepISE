"""Parser and normalizer for raw ISfinder snapshot data."""

import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple
from Bio import SeqIO

from deepise_ml.schemas.records import (
    CanonicalTpaseRecord,
    CanonicalISRecord,
    compute_sha256,
)


def load_isfinder_metadata(csv_path: Path) -> Dict[str, dict]:
    """Parse IS.csv metadata table into mapping by IS name."""
    metadata = {}
    with open(csv_path, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Name", "").strip()
            if not name:
                continue
            family = row.get("Family", "").strip() or "Unknown"
            group = row.get("Group", "").strip() or None
            if group in ("NA", "", "None", "unknown"):
                group = None
            origin = row.get("Origin", "").strip() or None
            length_str = row.get("Length", "").strip()
            length = int(length_str) if length_str.isdigit() else None
            ir = row.get("IR", "").strip() or None
            dr = row.get("DR", "").strip() or None
            orf = row.get("ORF", "").strip() or None
            acc = row.get("Accession Number", "").strip() or None

            metadata[name] = {
                "family": family,
                "group": group,
                "origin": origin,
                "length": length,
                "ir": ir,
                "dr": dr,
                "orf": orf,
                "accession": acc,
            }
    return metadata


def parse_protein_header(header: str) -> Tuple[str, str, str]:
    """Parse IS.faa header to extract is_name, orf_id, and role.
    
    Example header:
    html.2020//IS-LL6 ~~~html.2020//IS-LL6_unknown_unknown_ORF~~~Transposase~~~
    """
    parts = header.split("~~~")
    raw_name = parts[0].strip()
    is_name = raw_name.split("//")[-1].strip()
    orf_id = parts[1].strip() if len(parts) > 1 else ""
    role = parts[2].strip() if len(parts) > 2 else ""
    return is_name, orf_id, role


def parse_dna_header(header: str) -> Tuple[str, str, str]:
    """Parse IS.fna header to extract is_name, family, and group.
    
    Example header:
    IS-LL6_IS3_IS3 or IS100_IS21_unknown
    """
    parts = header.strip().split("_")
    is_name = parts[0]
    family = parts[1] if len(parts) > 1 else "Unknown"
    group = parts[2] if len(parts) > 2 and parts[2] != "unknown" else None
    return is_name, family, group


def extract_tpases(
    faa_path: Path, metadata: Dict[str, dict], confirmed_only: bool = True
) -> List[CanonicalTpaseRecord]:
    """Extract transposase protein records from IS.faa with metadata enrichment."""
    records = []
    # Read all records first to sort deterministically
    raw_entries = []
    for record in SeqIO.parse(str(faa_path), "fasta"):
        is_name, orf_id, role = parse_protein_header(record.description)
        is_tpase = (role.lower() == "transposase")
        if confirmed_only and not is_tpase:
            continue
        seq = str(record.seq).strip().upper()
        if not seq:
            continue
        raw_entries.append({
            "is_name": is_name,
            "orf_id": orf_id,
            "role": role,
            "raw_header": record.description,
            "sequence": seq,
            "is_tpase": is_tpase,
        })

    # Sort deterministically
    raw_entries.sort(key=lambda x: (x["is_name"], x["orf_id"], x["sequence"]))

    for idx, entry in enumerate(raw_entries, start=1):
        is_name = entry["is_name"]
        meta = metadata.get(is_name, {})
        family = meta.get("family") or "Unknown"
        group = meta.get("group")
        origin = meta.get("origin")
        acc = meta.get("accession")

        tpase_id = f"tpase_{idx:06d}"
        seq = entry["sequence"]
        seq_hash = compute_sha256(seq)

        record = CanonicalTpaseRecord(
            tpase_id=tpase_id,
            is_name=is_name,
            raw_header=entry["raw_header"],
            family=family,
            group=group,
            subgroup=None,
            protein_sequence=seq,
            protein_length=len(seq),
            protein_sha256=seq_hash,
            host_organism=origin,
            accession=acc,
            source="ISfinder",
            is_transposase=entry["is_tpase"],
        )
        records.append(record)

    return records


def extract_is_elements(
    fna_path: Path, metadata: Dict[str, dict]
) -> List[CanonicalISRecord]:
    """Extract full IS nucleotide element records from IS.fna with metadata enrichment."""
    raw_entries = []
    for record in SeqIO.parse(str(fna_path), "fasta"):
        is_name, header_family, header_group = parse_dna_header(record.id)
        seq = str(record.seq).strip().upper()
        if not seq:
            continue
        raw_entries.append({
            "is_name": is_name,
            "header_family": header_family,
            "header_group": header_group,
            "sequence": seq,
        })

    # Sort deterministically
    raw_entries.sort(key=lambda x: (x["is_name"], x["sequence"]))

    records = []
    for idx, entry in enumerate(raw_entries, start=1):
        is_name = entry["is_name"]
        meta = metadata.get(is_name, {})
        family = meta.get("family") or entry["header_family"]
        group = meta.get("group") or entry["header_group"]
        origin = meta.get("origin")
        acc = meta.get("accession")
        ir = meta.get("ir")
        dr = meta.get("dr")
        orf_ann = meta.get("orf")

        is_id = f"is_{idx:06d}"
        seq = entry["sequence"]
        seq_hash = compute_sha256(seq)

        record = CanonicalISRecord(
            is_id=is_id,
            is_name=is_name,
            family=family,
            group=group,
            dna_sequence=seq,
            dna_length=len(seq),
            dna_sha256=seq_hash,
            tir_left=ir,
            tir_right=None,
            dr_length=dr,
            orf_annotation=orf_ann,
            host_organism=origin,
            accession=acc,
            source="ISfinder",
        )
        records.append(record)

    return records
