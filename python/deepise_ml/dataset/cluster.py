"""MMseqs2 sequence clustering and cluster statistics analysis."""

import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import polars as pl


def run_mmseqs_clustering(
    fasta_path: Path,
    output_tsv: Path,
    tmp_dir: Path,
    min_seq_id: float = 0.30,
    coverage: float = 0.80,
    cov_mode: int = 0,
    cluster_mode: int = 1,
    sensitivity: float = 7.5,
    threads: int = 8,
    leakage_free_graph: bool = True,
) -> Path:
    """Execute MMseqs2 cluster pipeline on FASTA file.
    
    If leakage_free_graph is True, runs all-vs-all MMseqs2 search followed by
    exact connected-component partitioning to guarantee zero cross-cluster leakage.
    """
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    db_path = tmp_dir / "seq_db"
    clu_path = tmp_dir / "cluster_db"
    run_tmp = tmp_dir / "mmseqs_tmp"
    run_tmp.mkdir(parents=True, exist_ok=True)

    # 1. createdb
    cmd_createdb = [
        "mmseqs", "createdb",
        str(fasta_path),
        str(db_path),
    ]
    subprocess.run(cmd_createdb, check=True, capture_output=True, text=True)

    if leakage_free_graph:
        search_tmp = tmp_dir / "search_tmp"
        search_tmp.mkdir(parents=True, exist_ok=True)
        aln_db = tmp_dir / "aln_db"
        search_tsv = tmp_dir / "all_vs_all.tsv"

        subprocess.run([
            "mmseqs", "search",
            str(db_path), str(db_path),
            str(aln_db), str(search_tmp),
            "-s", str(sensitivity),
            "--threads", str(threads),
        ], check=True, capture_output=True, text=True)

        subprocess.run([
            "mmseqs", "convertalis",
            str(db_path), str(db_path),
            str(aln_db), str(search_tsv),
            "--format-output", "query,target,fident,alnlen,qcov,tcov,evalue,bits",
            "--threads", str(threads),
        ], check=True, capture_output=True, text=True)

        from Bio import SeqIO
        seq_lengths = {}
        for rec in SeqIO.parse(fasta_path, "fasta"):
            seq_lengths[rec.id] = len(rec.seq)

        parent = {sid: sid for sid in seq_lengths}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        with open(search_tsv, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 6:
                    q, t, fid, aln, qcov, tcov = parts[:6]
                    if q == t:
                        continue
                    fid, qcov, tcov = float(fid), float(qcov), float(tcov)
                    cov = max(qcov, tcov)
                    if fid > min_seq_id and cov >= coverage:
                        union(q, t)

        comp_members = defaultdict(list)
        for sid in seq_lengths:
            comp_members[find(sid)].append(sid)

        with open(output_tsv, "w", encoding="utf-8") as out:
            for root, members in comp_members.items():
                rep = max(members, key=lambda m: (seq_lengths.get(m, 0), m))
                for mem in members:
                    out.write(f"{rep}\t{mem}\n")

        return output_tsv

    # Fallback to standard mmseqs cluster
    cmd_cluster = [
        "mmseqs", "cluster",
        str(db_path),
        str(clu_path),
        str(run_tmp),
        "--min-seq-id", str(min_seq_id),
        "-c", str(coverage),
        "--cov-mode", str(cov_mode),
        "--cluster-mode", str(cluster_mode),
        "-s", str(sensitivity),
        "--threads", str(threads),
    ]
    subprocess.run(cmd_cluster, check=True, capture_output=True, text=True)

    cmd_tsv = [
        "mmseqs", "createtsv",
        str(db_path),
        str(db_path),
        str(clu_path),
        str(output_tsv),
        "--threads", str(threads),
    ]
    subprocess.run(cmd_tsv, check=True, capture_output=True, text=True)

    return output_tsv


def parse_cluster_tsv(
    tsv_path: Path, prefix: str = "cluster30_"
) -> Tuple[Dict[str, str], pl.DataFrame]:
    """Parse MMseqs2 cluster TSV file into member->cluster_id mapping.
    
    TSV columns: rep_id, member_id
    """
    clusters = defaultdict(list)
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                rep, mem = parts[0].strip(), parts[1].strip()
                clusters[rep].append(mem)

    # Sort clusters deterministically by size (descending), then representative ID
    sorted_reps = sorted(clusters.keys(), key=lambda r: (-len(clusters[r]), r))

    member_to_cluster = {}
    rows = []
    for idx, rep in enumerate(sorted_reps, start=1):
        cluster_id = f"{prefix}{idx:06d}"
        members = clusters[rep]
        for mem in members:
            member_to_cluster[mem] = cluster_id
            rows.append({
                "cluster_id": cluster_id,
                "representative_id": rep,
                "member_id": mem,
                "cluster_size": len(members),
                "is_representative": (mem == rep),
            })

    cluster_df = pl.DataFrame(rows)
    return member_to_cluster, cluster_df


def compute_cluster_statistics(
    cluster_df: pl.DataFrame,
    tpase_df: pl.DataFrame,
) -> dict:
    """Compute comprehensive cluster metrics and family distribution statistics."""
    # Join with tpase metadata
    merged = cluster_df.join(
        tpase_df.select(["tpase_id", "family", "protein_length"]),
        left_on="member_id",
        right_on="tpase_id",
        how="inner",
    )

    total_sequences = len(merged)
    cluster_sizes = (
        merged.group_by("cluster_id")
        .agg([
            pl.len().alias("size"),
            pl.col("family").n_unique().alias("num_families"),
            pl.col("family").unique().alias("families"),
        ])
    )

    num_clusters = len(cluster_sizes)
    singletons = cluster_sizes.filter(pl.col("size") == 1)
    num_singletons = len(singletons)
    singleton_fraction = num_singletons / num_clusters if num_clusters > 0 else 0.0

    sizes = cluster_sizes["size"].to_list()
    sizes.sort()
    median_size = sizes[len(sizes) // 2] if sizes else 0
    max_size = max(sizes) if sizes else 0

    # Multi-family clusters
    multi_family_clusters = cluster_sizes.filter(pl.col("num_families") > 1)
    num_multi_family = len(multi_family_clusters)

    # Per-family cluster counts
    family_stats = (
        merged.group_by("family")
        .agg([
            pl.len().alias("sequence_count"),
            pl.col("cluster_id").n_unique().alias("cluster_count"),
        ])
        .sort("sequence_count", descending=True)
    )

    return {
        "total_sequences": total_sequences,
        "num_clusters": num_clusters,
        "num_singletons": num_singletons,
        "singleton_fraction": singleton_fraction,
        "median_cluster_size": median_size,
        "max_cluster_size": max_size,
        "num_multi_family_clusters": num_multi_family,
        "family_stats": family_stats.to_dicts(),
        "cluster_sizes": cluster_sizes,
    }
