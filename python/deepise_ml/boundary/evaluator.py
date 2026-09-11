"""Unified comparative evaluator for Plan A vs Plan B boundary engines."""

import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import polars as pl
from deepise_ml.boundary.schemas import BoundaryPrediction, ContigGroundTruth
from deepise_ml.boundary.plan_a import PlanAAdaptiveEngine
from deepise_ml.boundary.plan_b import PlanBCanonicalEngine

NON_CANONICAL_FAMILIES = {"IS200/IS605", "IS91", "IS110", "IS607"}


def evaluate_engine_on_benchmark(
    engine,
    benchmark_df: pl.DataFrame,
    method_name: str,
) -> Tuple[List[BoundaryPrediction], Dict[str, float], pl.DataFrame]:
    """
    Run an engine (Plan A or Plan B) on the benchmark contig dataset.
    Returns:
        (predictions, summary_metrics, per_family_df)
    """
    predictions: List[BoundaryPrediction] = []
    eval_records: List[Dict] = []
    
    t_start = time.perf_counter()

    for row in benchmark_df.iter_rows(named=True):
        pred = engine.predict_boundary(
            contig=row["contig_sequence"],
            tpase_start=row["tpase_start"],
            tpase_end=row["tpase_end"],
            contig_id=row["contig_id"],
            is_name=row["is_name"],
            family=row["family"],
        )
        predictions.append(pred)

        true_s = row["true_start"]
        true_e = row["true_end"]
        pred_s = pred.predicted_start
        pred_e = pred.predicted_end

        err_s = abs(pred_s - true_s)
        err_e = abs(pred_e - true_e)
        max_err = max(err_s, err_e)

        is_canonical = row["family"] not in NON_CANONICAL_FAMILIES

        # TIR evaluation
        has_true_tir = row.get("annotated_tir_len") is not None and row.get("annotated_tir_len", 0) > 0
        has_pred_tir = pred.tir is not None
        tir_match_correct = False
        if has_pred_tir and has_true_tir:
            # Check if predicted TIR starts within 15 bp of true start
            if abs(pred.tir.left_start - true_s) <= 15:
                tir_match_correct = True

        # TSD evaluation
        has_true_tsd = row.get("annotated_tsd_len") is not None and row.get("annotated_tsd_len", 0) > 0
        has_pred_tsd = pred.tsd is not None
        tsd_match_correct = False
        if has_pred_tsd and has_true_tsd:
            true_tsd = row.get("true_tsd_seq", "")
            if true_tsd and (pred.tsd.sequence == true_tsd or true_tsd.startswith(pred.tsd.sequence) or pred.tsd.sequence.startswith(true_tsd)):
                tsd_match_correct = True
            elif abs(err_s) <= 2 and abs(err_e) <= 2:
                tsd_match_correct = True

        # False hallucination check on non-canonical
        false_tir_hallucinated = (not is_canonical) and has_pred_tir
        false_tsd_hallucinated = (not is_canonical) and has_pred_tsd

        eval_records.append({
            "contig_id": row["contig_id"],
            "is_name": row["is_name"],
            "family": row["family"],
            "is_canonical": is_canonical,
            "true_start": true_s,
            "true_end": true_e,
            "pred_start": pred_s,
            "pred_end": pred_e,
            "err_start": err_s,
            "err_end": err_e,
            "max_err": max_err,
            "exact_0bp": max_err == 0,
            "near_3bp": max_err <= 3,
            "coarse_10bp": max_err <= 10,
            "has_true_tir": has_true_tir,
            "has_pred_tir": has_pred_tir,
            "tir_correct": tir_match_correct,
            "has_true_tsd": has_true_tsd,
            "has_pred_tsd": has_pred_tsd,
            "tsd_correct": tsd_match_correct,
            "false_tir": false_tir_hallucinated,
            "false_tsd": false_tsd_hallucinated,
            "confidence": pred.confidence_score,
            "latency_ms": pred.latency_ms,
        })

    t_total = time.perf_counter() - t_start
    n = len(eval_records)
    rec_df = pl.DataFrame(eval_records)

    # Calculate overall metrics
    exact_rate = rec_df["exact_0bp"].mean()
    near_rate_3bp = rec_df["near_3bp"].mean()
    coarse_rate_10bp = rec_df["coarse_10bp"].mean()
    mean_err_s = rec_df["err_start"].mean()
    mean_err_e = rec_df["err_end"].mean()
    median_max_err = rec_df["max_err"].median()

    # Canonical sub-analysis
    can_df = rec_df.filter(pl.col("is_canonical"))
    non_can_df = rec_df.filter(~pl.col("is_canonical"))

    can_near_3bp = can_df["near_3bp"].mean() if len(can_df) > 0 else 0.0
    non_can_near_3bp = non_can_df["near_3bp"].mean() if len(non_can_df) > 0 else 0.0

    # TIR metrics on canonical
    can_pred_tirs = can_df["has_pred_tir"].sum()
    can_correct_tirs = can_df["tir_correct"].sum()
    can_true_tirs = can_df["has_true_tir"].sum()
    tir_precision = can_correct_tirs / can_pred_tirs if can_pred_tirs > 0 else 0.0
    tir_recall = can_correct_tirs / can_true_tirs if can_true_tirs > 0 else 0.0
    tir_f1 = (2 * tir_precision * tir_recall / (tir_precision + tir_recall)) if (tir_precision + tir_recall) > 0 else 0.0

    # TSD metrics on canonical
    can_pred_tsds = can_df["has_pred_tsd"].sum()
    can_correct_tsds = can_df["tsd_correct"].sum()
    can_true_tsds = can_df["has_true_tsd"].sum()
    tsd_precision = can_correct_tsds / can_pred_tsds if can_pred_tsds > 0 else 0.0
    tsd_recall = can_correct_tsds / can_true_tsds if can_true_tsds > 0 else 0.0
    tsd_f1 = (2 * tsd_precision * tsd_recall / (tsd_precision + tsd_recall)) if (tsd_precision + tsd_recall) > 0 else 0.0

    # Hallucination on non-canonical
    false_tir_rate = non_can_df["false_tir"].mean() if len(non_can_df) > 0 else 0.0
    false_tsd_rate = non_can_df["false_tsd"].mean() if len(non_can_df) > 0 else 0.0

    throughput = n / t_total if t_total > 0 else 0.0
    avg_latency = rec_df["latency_ms"].mean()

    summary_metrics = {
        "method": method_name,
        "n_total": float(n),
        "exact_match_rate_0bp": float(exact_rate),
        "near_match_rate_3bp": float(near_rate_3bp),
        "coarse_match_rate_10bp": float(coarse_rate_10bp),
        "canonical_near_match_3bp": float(can_near_3bp),
        "non_canonical_near_match_3bp": float(non_can_near_3bp),
        "mean_err_start_bp": float(mean_err_s),
        "mean_err_end_bp": float(mean_err_e),
        "median_max_err_bp": float(median_max_err),
        "tir_precision": float(tir_precision),
        "tir_recall": float(tir_recall),
        "tir_f1": float(tir_f1),
        "tsd_precision": float(tsd_precision),
        "tsd_recall": float(tsd_recall),
        "tsd_f1": float(tsd_f1),
        "non_can_false_tir_rate": float(false_tir_rate),
        "non_can_false_tsd_rate": float(false_tsd_rate),
        "total_runtime_sec": float(t_total),
        "avg_latency_ms": float(avg_latency),
        "throughput_elem_per_sec": float(throughput),
    }

    # Per-family breakdown
    family_summary = (
        rec_df.group_by("family")
        .agg([
            pl.len().alias("count"),
            pl.col("exact_0bp").mean().alias("exact_0bp_rate"),
            pl.col("near_3bp").mean().alias("near_3bp_rate"),
            pl.col("coarse_10bp").mean().alias("coarse_10bp_rate"),
            pl.col("err_start").mean().alias("mean_err_start"),
            pl.col("err_end").mean().alias("mean_err_end"),
            pl.col("confidence").mean().alias("mean_confidence"),
        ])
        .sort("count", descending=True)
    )

    return predictions, summary_metrics, family_summary
