"""HMMER profile HMM baseline implementation for remote transposase discovery."""

import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
import polars as pl
from Bio import SeqIO


def build_family_hmms(
    train_parquet: Path,
    output_hmm: Path,
    tmp_dir: Path,
    min_family_size: int = 3,
    threads: int = 16,
) -> Path:
    """Construct multi-family profile HMM database from training transposases.
    
    Families with >= min_family_size sequences are aligned via MAFFT and built into HMMs.
    """
    output_hmm.parent.mkdir(parents=True, exist_ok=True)
    if output_hmm.exists() and output_hmm.stat().st_size > 0:
        return output_hmm
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    train_df = pl.read_parquet(train_parquet)
    fams = defaultdict(list)
    for row in train_df.iter_rows(named=True):
        fams[row["family"]].append(row)

    built_hmms = []

    for fam_name, members in fams.items():
        if len(members) < min_family_size:
            continue

        clean_fam = "".join(c if c.isalnum() else "_" for c in fam_name)
        fam_faa = tmp_dir / f"{clean_fam}.faa"
        fam_aln = tmp_dir / f"{clean_fam}.aln"
        fam_hmm = tmp_dir / f"{clean_fam}.hmm"

        # Write FASTA
        with open(fam_faa, "w", encoding="utf-8") as f:
            for m in members:
                f.write(f">{m['tpase_id']}\n{m['protein_sequence']}\n")

        # Run MAFFT alignment
        cmd_mafft = f"mafft --auto --quiet --thread {min(threads, 4)} {fam_faa} > {fam_aln}"
        subprocess.run(cmd_mafft, shell=True, check=True)

        # Run hmmbuild
        cmd_hmmbuild = [
            "hmmbuild",
            "-n", f"Tpase_{clean_fam}",
            str(fam_hmm),
            str(fam_aln),
        ]
        subprocess.run(cmd_hmmbuild, check=True, capture_output=True, text=True)
        built_hmms.append(fam_hmm)

    # Concatenate all HMMs into single profile database
    with open(output_hmm, "w", encoding="utf-8") as outfile:
        for hmm_file in built_hmms:
            with open(hmm_file, "r", encoding="utf-8") as infile:
                outfile.write(infile.read())

    # Press database
    subprocess.run(["hmmpress", "-f", str(output_hmm)], check=True, capture_output=True, text=True)
    return output_hmm


def run_hmmsearch(
    query_faa: Path,
    hmm_db: Path,
    output_tbl: Path,
    threads: int = 16,
    evalue: float = 10.0,
) -> Path:
    """Execute hmmsearch against multi-family HMM database."""
    output_tbl.parent.mkdir(parents=True, exist_ok=True)
    if output_tbl.exists() and output_tbl.stat().st_size > 0:
        return output_tbl

    cmd = [
        "hmmsearch",
        "--cpu", str(threads),
        "-E", str(evalue),
        "--tblout", str(output_tbl),
        "--noali",
        str(hmm_db),
        str(query_faa),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return output_tbl


def parse_hmmsearch_predictions(
    tbl_path: Path,
    query_faa: Path,
) -> pl.DataFrame:
    """Parse HMMER tblout into standardized prediction DataFrame.
    
    Assigns top bitscore per query (0.0 for sequences without hits).
    """
    all_query_ids = []
    for rec in SeqIO.parse(query_faa, "fasta"):
        all_query_ids.append(rec.id)

    best_hits: Dict[str, dict] = {}
    if tbl_path.exists() and tbl_path.stat().st_size > 0:
        with open(tbl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) >= 6:
                    seq_name = parts[0]
                    profile_name = parts[2]
                    evalue = float(parts[4])
                    bitscore = float(parts[5])

                    if seq_name not in best_hits or bitscore > best_hits[seq_name]["score"]:
                        best_hits[seq_name] = {
                            "target_id": profile_name,
                            "score": bitscore,
                            "evalue": evalue,
                        }

    rows = []
    for q_id in all_query_ids:
        if q_id in best_hits:
            h = best_hits[q_id]
            rows.append({
                "seq_id": q_id,
                "score": h["score"],
                "target_id": h["target_id"],
                "pident": 0.0,
                "coverage": 0.0,
                "evalue": h["evalue"],
            })
        else:
            rows.append({
                "seq_id": q_id,
                "score": 0.0,
                "target_id": "None",
                "pident": 0.0,
                "coverage": 0.0,
                "evalue": 999.0,
            })

    return pl.DataFrame(rows)
