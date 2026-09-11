"""BLASTP baseline implementation for remote transposase discovery benchmark."""

import subprocess
from pathlib import Path
from typing import Dict, Optional
import polars as pl
from Bio import SeqIO


def build_blast_database(
    train_faa: Path,
    db_dir: Path,
    db_name: str = "blast_tpase",
) -> Path:
    """Build BLAST protein database from training transposase FASTA."""
    db_dir.mkdir(parents=True, exist_ok=True)
    db_prefix = db_dir / db_name

    cmd = [
        "makeblastdb",
        "-in", str(train_faa),
        "-dbtype", "prot",
        "-out", str(db_prefix),
        "-title", db_name,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return db_prefix


def run_blastp(
    query_faa: Path,
    db_prefix: Path,
    output_tsv: Path,
    threads: int = 16,
    evalue: float = 10.0,
) -> Path:
    """Execute BLASTP search against the reference database."""
    output_tsv.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "blastp",
        "-query", str(query_faa),
        "-db", str(db_prefix),
        "-out", str(output_tsv),
        "-outfmt", "6 qseqid sseqid pident length qcovs evalue bitscore",
        "-evalue", str(evalue),
        "-max_target_seqs", "1",
        "-num_threads", str(threads),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return output_tsv


def parse_blast_predictions(
    blast_tsv: Path,
    query_faa: Path,
) -> pl.DataFrame:
    """Parse BLAST results into standardized prediction DataFrame.
    
    Assigns top bitscore per query (0.0 for sequences without hits).
    """
    all_query_ids = []
    for rec in SeqIO.parse(query_faa, "fasta"):
        all_query_ids.append(rec.id)

    best_hits: Dict[str, dict] = {}
    if blast_tsv.exists() and blast_tsv.stat().st_size > 0:
        with open(blast_tsv, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 7:
                    q, s, pid, aln, qcov, ev, bit = parts[:7]
                    bit_val = float(bit)
                    if q not in best_hits or bit_val > best_hits[q]["score"]:
                        best_hits[q] = {
                            "target_id": s,
                            "score": bit_val,
                            "pident": float(pid),
                            "coverage": float(qcov),
                            "evalue": float(ev),
                        }

    rows = []
    for q_id in all_query_ids:
        if q_id in best_hits:
            h = best_hits[q_id]
            rows.append({
                "seq_id": q_id,
                "score": h["score"],
                "target_id": h["target_id"],
                "pident": h["pident"],
                "coverage": h["coverage"],
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
