"""Cluster-aware, stratified negative dataset splitting and evaluation dataset assembly."""

import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import polars as pl
from Bio import SeqIO


def split_negatives(
    negatives_parquet: Path,
    split_dir: Path,
    seed: int = 42,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split negatives into train, validation, and test to achieve exact 1:3 positive-to-negative ratio.
    
    Preserves negative_type stratification (50% hard, 30% length_matched, 20% easy)
    across all splits.
    """
    neg_df = pl.read_parquet(negatives_parquet)
    train_pos = pl.read_parquet(split_dir / "train.parquet")
    val_pos = pl.read_parquet(split_dir / "validation.parquet")
    test_pos = pl.read_parquet(split_dir / "test.parquet")

    target_train_neg = len(train_pos) * 3
    target_val_neg = len(val_pos) * 3
    target_test_neg = len(test_pos) * 3
    total_needed = target_train_neg + target_val_neg + target_test_neg

    assert len(neg_df) >= total_needed, f"Not enough negatives: {len(neg_df)} < {total_needed}"

    rng = random.Random(seed)

    # Group negatives by negative_type
    by_type = defaultdict(list)
    for row in neg_df.iter_rows(named=True):
        by_type[row["negative_type"]].append(row)

    train_rows, val_rows, test_rows = [], [], []

    # For each negative type, split proportionally 70/15/15
    for neg_type, items in by_type.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(round(n * (target_train_neg / total_needed)))
        n_val = int(round(n * (target_val_neg / total_needed)))
        # Remainder to test
        train_slice = items[:n_train]
        val_slice = items[n_train:n_train + n_val]
        test_slice = items[n_train + n_val:]

        train_rows.extend(train_slice)
        val_rows.extend(val_slice)
        test_rows.extend(test_slice)

    # Adjust exact counts if rounding difference
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)

    train_neg_df = pl.DataFrame(train_rows).with_columns(pl.lit("train").alias("split"))
    val_neg_df = pl.DataFrame(val_rows).with_columns(pl.lit("validation").alias("split"))
    test_neg_df = pl.DataFrame(test_rows).with_columns(pl.lit("test").alias("split"))

    return train_neg_df, val_neg_df, test_neg_df


def build_combined_splits(
    split_dir: Path,
    negatives_parquet: Path,
    output_dir: Path,
    seed: int = 42,
):
    """Create combined positive+negative datasets for train, validation, and test splits."""
    output_dir.mkdir(parents=True, exist_ok=True)
    train_neg_df, val_neg_df, test_neg_df = split_negatives(negatives_parquet, split_dir, seed=seed)

    # Save negative splits
    train_neg_df.write_parquet(output_dir / "train_negatives.parquet")
    val_neg_df.write_parquet(output_dir / "validation_negatives.parquet")
    test_neg_df.write_parquet(output_dir / "test_negatives.parquet")

    splits = [
        ("train", "train.parquet", train_neg_df),
        ("validation", "validation.parquet", val_neg_df),
        ("test", "test.parquet", test_neg_df),
    ]

    for split_name, pos_file, neg_df in splits:
        pos_df = pl.read_parquet(split_dir / pos_file)

        # Standardize columns
        pos_standard = pos_df.select([
            pl.col("tpase_id").alias("seq_id"),
            pl.col("protein_sequence"),
            pl.col("protein_length"),
            pl.col("family").alias("family"),
            pl.lit(1).alias("label"),
            pl.lit("positive").alias("class_type"),
            pl.col("cluster30_id"),
            pl.col("split"),
        ])

        neg_standard = neg_df.select([
            pl.col("negative_id").alias("seq_id"),
            pl.col("protein_sequence"),
            pl.col("protein_length"),
            pl.lit("None").alias("family"),
            pl.lit(0).alias("label"),
            pl.col("negative_type").alias("class_type"),
            pl.col("cluster30_id").fill_null(pl.col("negative_id")),
            pl.col("split"),
        ])

        combined_df = pl.concat([pos_standard, neg_standard])
        combined_df.write_parquet(output_dir / f"{split_name}_combined.parquet")

        # Export combined FASTA
        faa_path = output_dir / f"{split_name}_combined.faa"
        with open(faa_path, "w", encoding="utf-8") as f:
            for row in combined_df.iter_rows(named=True):
                f.write(f">{row['seq_id']}\n{row['protein_sequence']}\n")
