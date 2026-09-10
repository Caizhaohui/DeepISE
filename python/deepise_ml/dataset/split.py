"""Cluster-aware homology-disjoint dataset splitting algorithm."""

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import polars as pl
import yaml

from deepise_ml.schemas.records import CanonicalTpaseRecord


def cluster_aware_split(
    tpase_df: pl.DataFrame,
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict]:
    """Split dataset by cluster30_id ensuring zero cluster overlap across splits.
    
    Priority:
    1. Cluster integrity (clusters are atomic)
    2. Approximate sequence ratio (70/15/15)
    3. Balanced family distribution across splits
    """
    assert abs(sum(ratios) - 1.0) < 1e-4, "Ratios must sum to 1.0"
    rng = random.Random(seed)

    # Group sequences by cluster30_id
    cluster_groups = (
        tpase_df.group_by("cluster30_id")
        .agg([
            pl.len().alias("size"),
            pl.col("family").alias("families"),
            pl.col("tpase_id").alias("tpase_ids"),
        ])
    ).to_dicts()

    total_seqs = len(tpase_df)
    target_counts = {
        "train": total_seqs * ratios[0],
        "validation": total_seqs * ratios[1],
        "test": total_seqs * ratios[2],
    }

    # Count global family frequencies
    global_family_counts = Counter(tpase_df["family"].to_list())

    # Sort clusters: multi-sequence clusters first (descending size), with seed tie-breaker
    # Shuffle first with seed, then sort by size descending to get deterministic behavior with randomized tie-breaks
    rng.shuffle(cluster_groups)
    cluster_groups.sort(key=lambda c: c["size"], reverse=True)

    split_clusters = {"train": [], "validation": [], "test": []}
    split_counts = {"train": 0, "validation": 0, "test": 0}
    split_family_counts = {
        "train": Counter(),
        "validation": Counter(),
        "test": Counter(),
    }

    for cluster in cluster_groups:
        c_id = cluster["cluster30_id"]
        c_size = cluster["size"]
        c_fams = cluster["families"]

        best_split = None
        best_score = float("inf")

        for s_name in ["train", "validation", "test"]:
            current_c = split_counts[s_name]
            target_c = target_counts[s_name]

            # Capacity penalty: deviation from target
            capacity_ratio = (current_c + c_size) / (target_c + 1e-5)

            # Family representation score: evaluate if adding cluster helps balance
            fam_penalty = 0.0
            for f in c_fams:
                current_f = split_family_counts[s_name][f]
                target_f = global_family_counts[f] * (target_counts[s_name] / total_seqs)
                # Penalize over-representation of this family in this split
                if current_f > target_f:
                    fam_penalty += (current_f - target_f) / (target_f + 1.0)

            # Combined score (minimize)
            score = capacity_ratio + 0.5 * fam_penalty

            if score < best_score:
                best_score = score
                best_split = s_name

        # Assign cluster
        split_clusters[best_split].append(c_id)
        split_counts[best_split] += c_size
        for f in c_fams:
            split_family_counts[best_split][f] += 1

    # Map back to DataFrames
    train_clusters = set(split_clusters["train"])
    val_clusters = set(split_clusters["validation"])
    test_clusters = set(split_clusters["test"])

    # Double check no overlap
    assert len(train_clusters & val_clusters) == 0, "Cluster leakage train-val!"
    assert len(train_clusters & test_clusters) == 0, "Cluster leakage train-test!"
    assert len(val_clusters & test_clusters) == 0, "Cluster leakage val-test!"

    train_df = tpase_df.filter(pl.col("cluster30_id").is_in(train_clusters)).with_columns(pl.lit("train").alias("split"))
    val_df = tpase_df.filter(pl.col("cluster30_id").is_in(val_clusters)).with_columns(pl.lit("validation").alias("split"))
    test_df = tpase_df.filter(pl.col("cluster30_id").is_in(test_clusters)).with_columns(pl.lit("test").alias("split"))

    manifest = {
        "strategy": "cluster30",
        "seed": seed,
        "protein_identity_constraint": {
            "threshold": 0.30,
            "coverage": 0.80,
        },
        "counts": {
            "total": total_seqs,
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
        },
        "clusters": {
            "total": len(cluster_groups),
            "train": len(train_clusters),
            "validation": len(val_clusters),
            "test": len(test_clusters),
        },
    }

    return train_df, val_df, test_df, manifest


def save_splits(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    manifest: dict,
    output_dir: Path,
):
    """Save split Parquet files, fasta sequences, and manifest.yaml."""
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.write_parquet(output_dir / "train.parquet")
    val_df.write_parquet(output_dir / "validation.parquet")
    test_df.write_parquet(output_dir / "test.parquet")

    # Fastas
    for name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        faa_path = output_dir / f"{name}.faa"
        with open(faa_path, "w", encoding="utf-8") as f:
            for row in df.iter_rows(named=True):
                f.write(f">{row['tpase_id']}\n{row['protein_sequence']}\n")

    with open(output_dir / "manifest.yaml", "w", encoding="utf-8") as f:
        yaml.dump(manifest, f, default_flow_style=False)
