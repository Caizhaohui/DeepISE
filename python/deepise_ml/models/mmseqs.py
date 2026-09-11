"""MMseqs2 baseline implementation for remote transposase discovery benchmark."""

import shutil
import subprocess
from pathlib import Path
from typing import Dict
import polars as pl
from Bio import SeqIO


def run_mmseqs_search(
    query_faa: Path,
    target_faa: Path,
    output_tsv: Path,
    tmp_dir: Path,
    sensitivity: float = 7.5,
    threads: int = 16,
) -> Path:
    """Execute MMseqs2 easy-search against reference training proteins."""
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "mmseqs", "easy-search",
        str(query_faa),
        str(target_faa),
        str(output_tsv),
        str(tmp_dir),
        "-s", str(sensitivity),
        "--threads", str(threads),
        "--format-output", "query,target,fident,alnlen,qcov,tcov,evalue,bits",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return output_tsv


def parse_mmseqs_predictions(
    search_tsv: Path,
    query_faa: Path,
) -> pl.DataFrame:
    """Parse MMseqs search results into standardized prediction DataFrame.
    
    Assigns top bitscore per query (0.0 for sequences without hits).
    """
    all_query_ids = []
    for rec in SeqIO.parse(query_faa, "fasta"):
        all_query_ids.append(rec.id)

    best_hits: Dict[str, dict] = {}
    if search_tsv.exists() and search_tsv.stat().st_size > 0:
        with open(search_tsv, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 8:
                    q, t, fid, aln, qcov, tcov, ev, bit = parts[:8]
                    bit_val = float(bit)
                    if q not in best_hits or bit_val > best_hits[q]["score"]:
                        best_hits[q] = {
                            "target_id": t,
                            "score": bit_val,
                            "pident": float(fid) * 100.0,
                            "coverage": max(float(qcov), float(tcov)) * 100.0,
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
