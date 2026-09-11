"""Comprehensive evaluation suite: validation-only FDR threshold selection, test metrics, stratification, and bootstrapping."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import polars as pl
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)


def find_fdr_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    target_fdr: float = 0.05,
) -> Tuple[float, float, float]:
    """Find score threshold on validation set that maximizes recall subject to FDR <= target_fdr.
    
    Returns:
        (best_threshold, achieved_fdr, achieved_recall)
    """
    assert len(y_true) == len(y_scores), "y_true and y_scores must have same length"
    
    # Sort distinct descending thresholds
    unique_scores = np.sort(np.unique(y_scores))[::-1]
    
    best_threshold = float("inf")
    best_recall = 0.0
    best_fdr = 0.0
    
    total_pos = np.sum(y_true == 1)
    if total_pos == 0:
        return best_threshold, 0.0, 0.0

    for thresh in unique_scores:
        pred = (y_scores >= thresh).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        
        if tp + fp == 0:
            continue
            
        fdr = fp / (tp + fp)
        recall = tp / total_pos
        
        if fdr <= target_fdr:
            if recall > best_recall:
                best_recall = recall
                best_threshold = float(thresh)
                best_fdr = float(fdr)

    return best_threshold, best_fdr, best_recall


def compute_binary_metrics_at_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Evaluate performance metrics on a dataset at a frozen score threshold."""
    y_pred = (y_scores >= threshold).astype(int)
    
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fdr = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    try:
        mcc = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        mcc = 0.0
        
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "specificity": specificity,
        "fdr": fdr,
    }


def compute_overall_metrics(
    val_true: np.ndarray,
    val_scores: np.ndarray,
    test_true: np.ndarray,
    test_scores: np.ndarray,
) -> Dict[str, float]:
    """Tune thresholds on validation, then report comprehensive performance on test set."""
    # 1. Global ranking metrics on test set
    try:
        auroc = float(roc_auc_score(test_true, test_scores))
    except Exception:
        auroc = 0.5
        
    try:
        auprc = float(average_precision_score(test_true, test_scores))
    except Exception:
        auprc = 0.0

    # 2. Tune thresholds on validation set
    th_1, val_fdr_1, _ = find_fdr_threshold(val_true, val_scores, target_fdr=0.01)
    th_5, val_fdr_5, _ = find_fdr_threshold(val_true, val_scores, target_fdr=0.05)
    th_10, val_fdr_10, _ = find_fdr_threshold(val_true, val_scores, target_fdr=0.10)

    # 3. Apply to test set
    m_1 = compute_binary_metrics_at_threshold(test_true, test_scores, th_1)
    m_5 = compute_binary_metrics_at_threshold(test_true, test_scores, th_5)
    m_10 = compute_binary_metrics_at_threshold(test_true, test_scores, th_10)

    return {
        "auroc": auroc,
        "auprc": auprc,
        "threshold_1pct_fdr": th_1,
        "threshold_5pct_fdr": th_5,
        "threshold_10pct_fdr": th_10,
        "val_fdr_at_5pct": val_fdr_5,
        "test_recall_at_1pct_fdr": m_1["recall"],
        "test_recall_at_5pct_fdr": m_5["recall"],
        "test_recall_at_10pct_fdr": m_10["recall"],
        "test_precision_at_5pct_fdr": m_5["precision"],
        "test_fdr_at_5pct": m_5["fdr"],
        "test_f1_at_5pct_fdr": m_5["f1"],
        "test_mcc_at_5pct_fdr": m_5["mcc"],
        "test_specificity_at_5pct_fdr": m_5["specificity"],
    }


def compute_identity_stratified_recall(
    test_df: pl.DataFrame,
    predictions_df: pl.DataFrame,
    threshold: float,
    bins: Optional[List[Tuple[float, float, str]]] = None,
) -> pl.DataFrame:
    """Compute test recall within predefined sequence identity bins to the training reference."""
    if bins is None:
        bins = [
            (0.0, 0.20, "<20%"),
            (0.20, 0.30, "20-30%"),
            (0.30, 0.50, "30-50%"),
            (0.50, 0.70, "50-70%"),
            (0.70, 1.01, ">70%"),
        ]

    # Join predictions with test annotations
    joined = test_df.join(predictions_df.select(["seq_id", "score"]), on="seq_id", how="inner")
    positives = joined.filter(pl.col("label") == 1)

    rows = []
    for low, high, label_bin in bins:
        sub = positives.filter(
            (pl.col("max_train_identity") >= low) & (pl.col("max_train_identity") < high)
        )
        n_pos = len(sub)
        if n_pos > 0:
            n_det = len(sub.filter(pl.col("score") >= threshold))
            rec = n_det / n_pos
        else:
            n_det = 0
            rec = 0.0

        rows.append({
            "identity_bin": label_bin,
            "min_identity": low,
            "max_identity": high,
            "n_positives": n_pos,
            "detected_positives": n_det,
            "recall": rec,
        })

    return pl.DataFrame(rows)


def compute_family_recall(
    test_df: pl.DataFrame,
    predictions_df: pl.DataFrame,
    threshold: float,
) -> Tuple[pl.DataFrame, float]:
    """Compute per-family and unweighted macro recall on test positives."""
    joined = test_df.join(predictions_df.select(["seq_id", "score"]), on="seq_id", how="inner")
    positives = joined.filter(pl.col("label") == 1)

    fam_groups = (
        positives.group_by("family")
        .agg([
            pl.len().alias("n_positives"),
            (pl.col("score") >= threshold).sum().alias("detected_positives"),
        ])
        .with_columns(
            (pl.col("detected_positives") / pl.col("n_positives")).alias("recall")
        )
        .sort("n_positives", descending=True)
    )

    macro_recall = float(fam_groups["recall"].mean()) if len(fam_groups) > 0 else 0.0
    return fam_groups, macro_recall


def bootstrap_cluster_ci(
    test_df: pl.DataFrame,
    test_scores: np.ndarray,
    threshold: float,
    n_replicates: int = 1000,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Compute cluster-level bootstrap 95% confidence intervals for Recall@5%FDR and AUPRC."""
    rng = np.random.RandomState(seed)
    
    test_with_score = test_df.with_columns(pl.Series("score", test_scores))
    clusters = test_with_score["cluster30_id"].unique().to_list()
    
    # Map cluster_id to dataframe rows
    cluster_dict = {}
    for c_id in clusters:
        cluster_dict[c_id] = test_with_score.filter(pl.col("cluster30_id") == c_id)

    recalls = []
    auprcs = []

    for _ in range(n_replicates):
        sample_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        sampled_dfs = [cluster_dict[cid] for cid in sample_clusters]
        boot_df = pl.concat(sampled_dfs)
        
        y_true = boot_df["label"].to_numpy()
        y_score = boot_df["score"].to_numpy()
        
        pos_mask = (y_true == 1)
        if np.sum(pos_mask) > 0:
            rec = np.sum((y_score[pos_mask] >= threshold)) / np.sum(pos_mask)
            recalls.append(rec)
            
        try:
            auprc = average_precision_score(y_true, y_score)
            auprcs.append(auprc)
        except Exception:
            pass

    ci_recall = (float(np.percentile(recalls, 2.5)), float(np.percentile(recalls, 97.5))) if recalls else (0.0, 0.0)
    ci_auprc = (float(np.percentile(auprcs, 2.5)), float(np.percentile(auprcs, 97.5))) if auprcs else (0.0, 0.0)

    return {
        "recall_95ci": ci_recall,
        "auprc_95ci": ci_auprc,
    }
