"""DeepISE Scientific Audit v1.1 - Steps 24 to 34: Comprehensive Phase-1 Benchmark v2.

Evaluates all models on cluster30_v2 under strict Audit v1.1 protocol:
1. Models:
   - Alignment: BLASTP, MMseqs2 (against train.faa)
   - Profile HMMs: HMM-A (Whole-family), HMM-B (Cluster-specific), HMM-C (Domain)
   - PLMs: ESM2-LR (35M), ESM2-MLP (35M), ESM2-LR (8M)
2. Fair 5% FDR threshold calibration strictly on validation split.
3. Stratified evaluation on test split:
   - Overall Test
   - Strict Remote-30
   - Strict Remote-20
   - No Full-Length Homolog
4. Hard-negative challenge evaluation (data/challenge/hard_negative_challenge.parquet, n=196).
5. Generates benchmark/tables/phase1_benchmark_v2.tsv and benchmark/reports/phase1_benchmark_v2.md.
"""

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import polars as pl
from Bio import SeqIO
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_curve, auc, matthews_corrcoef
from sklearn.preprocessing import StandardScaler

from deepise_ml.dataset.leakage import audit_homology_v2

# Ensure conda env binaries
ENV_BIN = Path("/hpcfs/fhome/caizhh/.conda/envs/DeepISE/bin")
if str(ENV_BIN) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{ENV_BIN}:{os.environ.get('PATH', '')}"

ROOT_DIR = Path("/hpcfs/fhome/caizhh/Desktop/03_Tool_Development/05_DeepISE")


def find_fdr_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_fdr: float = 0.05,
) -> Tuple[float, float, float]:
    """Calibrate score threshold achieving closest FDR <= target_fdr on validation data."""
    unique_scores = np.unique(scores)
    unique_scores = np.sort(unique_scores)[::-1]

    best_thresh = float(unique_scores[0]) if len(unique_scores) > 0 else 0.5
    best_fdr = 0.0
    best_recall = 0.0

    total_pos = np.sum(y_true == 1)
    if total_pos == 0:
        return 0.5, 0.0, 0.0

    for thresh in unique_scores:
        pred_pos = scores >= thresh
        tp = np.sum((y_true == 1) & pred_pos)
        fp = np.sum((y_true == 0) & pred_pos)
        call_count = tp + fp
        if call_count == 0:
            continue

        fdr = fp / call_count
        recall = tp / total_pos
        if fdr <= target_fdr:
            best_thresh = float(thresh)
            best_fdr = float(fdr)
            best_recall = float(recall)
        else:
            break

    return best_thresh, best_fdr, best_recall


def compute_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(int)
    tp = np.sum((y_true == 1) & (pred == 1))
    fp = np.sum((y_true == 0) & (pred == 1))
    fn = np.sum((y_true == 1) & (pred == 0))
    tn = np.sum((y_true == 0) & (pred == 0))

    total_pos = tp + fn
    total_neg = tn + fp

    recall = tp / total_pos if total_pos > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    fpr = fp / total_neg if total_neg > 0 else 0.0
    fdr = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    mcc = float(matthews_corrcoef(y_true, pred)) if (total_pos > 0 and total_neg > 0 and len(np.unique(pred)) > 1) else 0.0

    # AUPRC
    if total_pos > 0 and total_neg > 0:
        prec_arr, rec_arr, _ = precision_recall_curve(y_true, scores)
        auprc = auc(rec_arr, prec_arr)
    else:
        auprc = 0.0

    return {
        "recall": float(recall),
        "precision": float(precision),
        "fpr": float(fpr),
        "fdr": float(fdr),
        "mcc": float(mcc),
        "auprc": float(auprc),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def run_mmseqs_against_train(query_faa: Path, target_faa: Path, output_tsv: Path, tmp_dir: Path) -> Dict[str, float]:
    """MMseqs2 search against target training database."""
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    all_query_ids = [r.id for r in SeqIO.parse(query_faa, "fasta")]

    cmd = [
        "mmseqs", "easy-search",
        str(query_faa),
        str(target_faa),
        str(output_tsv),
        str(tmp_dir),
        "-s", "7.5",
        "--threads", "16",
        "--format-output", "query,target,fident,alnlen,qcov,tcov,evalue,bits",
        "-v", "0",
    ]
    subprocess.run(cmd, check=True)

    scores = {q_id: 0.0 for q_id in all_query_ids}
    if output_tsv.exists() and output_tsv.stat().st_size > 0:
        with open(output_tsv, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 8:
                    q, bit = parts[0], float(parts[7])
                    if q in scores and bit > scores[q]:
                        scores[q] = bit

    return scores


def run_blastp_against_train(query_faa: Path, target_faa: Path, output_tsv: Path, db_dir: Path) -> Dict[str, float]:
    """BLASTP search against target training database."""
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    db_dir.mkdir(parents=True, exist_ok=True)
    db_prefix = db_dir / "train_db"

    if not (db_prefix.with_suffix(".pin")).exists():
        cmd_db = [
            "makeblastdb",
            "-in", str(target_faa),
            "-dbtype", "prot",
            "-out", str(db_prefix),
            "-title", "train_db",
        ]
        subprocess.run(cmd_db, check=True, capture_output=True, text=True)

    all_query_ids = [r.id for r in SeqIO.parse(query_faa, "fasta")]
    cmd_blast = [
        "blastp",
        "-query", str(query_faa),
        "-db", str(db_prefix),
        "-out", str(output_tsv),
        "-outfmt", "6 qseqid sseqid pident length qcovs evalue bitscore",
        "-evalue", "10.0",
        "-max_target_seqs", "1",
        "-num_threads", "16",
    ]
    subprocess.run(cmd_blast, check=True, capture_output=True, text=True)

    scores = {q_id: 0.0 for q_id in all_query_ids}
    if output_tsv.exists() and output_tsv.stat().st_size > 0:
        with open(output_tsv, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 7:
                    q, bit = parts[0], float(parts[6])
                    if q in scores and bit > scores[q]:
                        scores[q] = bit

    return scores


def run_hmm_search(query_faa: Path, hmm_db: Path) -> Dict[str, float]:
    """HMMER hmmsearch returning max bitscore per query."""
    all_query_ids = [r.id for r in SeqIO.parse(query_faa, "fasta")]
    tbl_out = query_faa.parent / f"{query_faa.stem}_{hmm_db.stem}.tbl"

    cmd = [
        "hmmsearch",
        "--cpu", "16",
        "-E", "10.0",
        "--tblout", str(tbl_out),
        "--noali",
        str(hmm_db),
        str(query_faa),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    scores = {q_id: 0.0 for q_id in all_query_ids}
    if tbl_out.exists() and tbl_out.stat().st_size > 0:
        with open(tbl_out, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) >= 6:
                    q, bit = parts[0], float(parts[5])
                    if q in scores and bit > scores[q]:
                        scores[q] = bit

    if tbl_out.exists():
        tbl_out.unlink()
    return scores


def main():
    print("=" * 80)
    print("STARTING DEEPISE SCIENTIFIC AUDIT v1.1 - PHASE-1 BENCHMARK v2")
    print("=" * 80)

    splits_dir = ROOT_DIR / "data/splits/cluster30_v2"
    challenge_pq = ROOT_DIR / "data/challenge/hard_negative_challenge.parquet"
    db_dir = ROOT_DIR / "benchmark/db"
    emb_dir = ROOT_DIR / "data/processed/embeddings_v2"

    train_df = pl.read_parquet(splits_dir / "train.parquet")
    val_df = pl.read_parquet(splits_dir / "validation.parquet")
    test_df = pl.read_parquet(splits_dir / "test.parquet")
    ch_df = pl.read_parquet(challenge_pq)

    train_faa = splits_dir / "train.faa"
    val_faa = splits_dir / "validation.faa"
    test_faa = splits_dir / "test.faa"
    ch_faa = challenge_pq.parent / "hard_negative_challenge.faa"

    y_val = val_df["label"].to_numpy()
    val_ids = val_df["seq_id"].to_list()

    y_test = test_df["label"].to_numpy()
    test_ids = test_df["seq_id"].to_list()
    ch_ids = ch_df["seq_id"].to_list()

    # --- Step A: Compute Test Homology Stratification to Train Positives ---
    print("\n--> Auditing Test Set Homology Stratification against Train Positives...")
    tmp_audit = Path(tempfile.mkdtemp(prefix="deepise_test_audit_"))
    try:
        test_pos_faa = tmp_audit / "test_pos.faa"
        train_pos_faa = tmp_audit / "train_pos.faa"
        search_tsv = tmp_audit / "test_vs_train.tsv"

        test_pos_records = test_df.filter(pl.col("label") == 1).to_dicts()
        train_pos_records = train_df.filter(pl.col("label") == 1).to_dicts()

        with open(test_pos_faa, "w", encoding="utf-8") as f:
            for r in test_pos_records:
                f.write(f">{r['seq_id']}\n{r['protein_sequence']}\n")

        with open(train_pos_faa, "w", encoding="utf-8") as f:
            for r in train_pos_records:
                f.write(f">{r['seq_id']}\n{r['protein_sequence']}\n")

        cmd_search = [
            "mmseqs", "easy-search",
            str(test_pos_faa),
            str(train_pos_faa),
            str(search_tsv),
            str(tmp_audit / "mm_tmp"),
            "-s", "7.5",
            "--max-seqs", "20000",
            "--format-output", "query,target,fident,alnlen,qcov,tcov,evalue,bits",
            "-v", "0",
        ]
        subprocess.run(cmd_search, check=True)

        homology_df, hom_summary = audit_homology_v2(
            search_tsv,
            [r["seq_id"] for r in test_pos_records],
            identity_threshold=0.30,
            coverage_threshold=0.80,
        )
        print(f"  Test Homology Summary: {hom_summary}")
    finally:
        shutil.rmtree(tmp_audit, ignore_errors=True)

    # Map each test positive to its homology class & identity
    pos_homology = {r["test_id"]: r for r in homology_df.iter_rows(named=True)}

    # Define Stratified Subsets
    # Negatives are identical across all strata (n=2,795)
    test_neg_mask = (y_test == 0)

    def get_fl_id(s_id):
        val = pos_homology.get(s_id, {}).get("max_full_length_identity")
        return 0.0 if val is None else float(val)

    # 1. Overall Test: all test samples
    mask_overall = np.ones(len(y_test), dtype=bool)

    # 2. Strict Remote-30: identity < 30% or no full-length homolog
    mask_remote30 = np.array([
        (test_neg_mask[i] or get_fl_id(s_id) < 0.30)
        for i, s_id in enumerate(test_ids)
    ])

    # 3. Strict Remote-20: identity < 20% (or no full-length homolog)
    mask_remote20 = np.array([
        (test_neg_mask[i] or get_fl_id(s_id) < 0.20)
        for i, s_id in enumerate(test_ids)
    ])

    # 4. No Full-Length Homolog: sequences without reciprocal >=80% coverage hit
    mask_no_full = np.array([
        (test_neg_mask[i] or pos_homology.get(s_id, {}).get("homology_class") in ["NO_FULL_LENGTH_HOMOLOG", "L3_DOMAIN_ONLY"])
        for i, s_id in enumerate(test_ids)
    ])

    print(f"  Overall Test Positives: {np.sum((y_test == 1) & mask_overall)}")
    print(f"  Strict Remote-30 Positives: {np.sum((y_test == 1) & mask_remote30)}")
    print(f"  Strict Remote-20 Positives: {np.sum((y_test == 1) & mask_remote20)}")
    print(f"  No Full-Length Homolog Positives: {np.sum((y_test == 1) & mask_no_full)}")
    print(f"  Test Negatives (all strata): {np.sum(test_neg_mask)}")

    strata = {
        "Overall": mask_overall,
        "Remote30": mask_remote30,
        "Remote20": mask_remote20,
        "NoFullLength": mask_no_full,
    }

    # --- Step B: Run Alignment Baselines (BLASTP & MMseqs2) ---
    print("\n--> Running BLASTP baseline...")
    tmp_blast = ROOT_DIR / "benchmark/tmp_blast"
    blast_val_scores_dict = run_blastp_against_train(val_faa, train_faa, tmp_blast / "val.tsv", tmp_blast / "db")
    blast_test_scores_dict = run_blastp_against_train(test_faa, train_faa, tmp_blast / "test.tsv", tmp_blast / "db")
    blast_ch_scores_dict = run_blastp_against_train(ch_faa, train_faa, tmp_blast / "ch.tsv", tmp_blast / "db")
    shutil.rmtree(tmp_blast, ignore_errors=True)

    print("--> Running MMseqs2 baseline...")
    tmp_mm = ROOT_DIR / "benchmark/tmp_mmseqs"
    mm_val_scores_dict = run_mmseqs_against_train(val_faa, train_faa, tmp_mm / "val.tsv", tmp_mm / "tmp")
    mm_test_scores_dict = run_mmseqs_against_train(test_faa, train_faa, tmp_mm / "test.tsv", tmp_mm / "tmp")
    mm_ch_scores_dict = run_mmseqs_against_train(ch_faa, train_faa, tmp_mm / "ch.tsv", tmp_mm / "tmp")
    shutil.rmtree(tmp_mm, ignore_errors=True)

    # --- Step C: Run Three HMM Baselines ---
    print("--> Running HMM-A (Whole-family HMM v2)...")
    hmm_a_db = db_dir / "deepise_tpases_v2.hmm"
    hmm_a_val_dict = run_hmm_search(val_faa, hmm_a_db)
    hmm_a_test_dict = run_hmm_search(test_faa, hmm_a_db)
    hmm_a_ch_dict = run_hmm_search(ch_faa, hmm_a_db)

    print("--> Running HMM-B (Cluster-specific HMM)...")
    hmm_b_db = db_dir / "deepise_cluster_tpases_v2.hmm"
    hmm_b_val_dict = run_hmm_search(val_faa, hmm_b_db)
    hmm_b_test_dict = run_hmm_search(test_faa, hmm_b_db)
    hmm_b_ch_dict = run_hmm_search(ch_faa, hmm_b_db)

    print("--> Running HMM-C (Domain-HMM)...")
    hmm_c_db = db_dir / "deepise_domain_tpases_v2.hmm"
    hmm_c_val_dict = run_hmm_search(val_faa, hmm_c_db)
    hmm_c_test_dict = run_hmm_search(test_faa, hmm_c_db)
    hmm_c_ch_dict = run_hmm_search(ch_faa, hmm_c_db)

    # --- Step D: Train and Evaluate ESM-2 Models ---
    print("--> Training and evaluating ESM-2 (35M & 8M) models...")
    X_tr_35m = np.load(emb_dir / "esm2_35m_train.npy")
    X_val_35m = np.load(emb_dir / "esm2_35m_validation.npy")
    X_test_35m = np.load(emb_dir / "esm2_35m_test.npy")
    X_ch_35m = np.load(emb_dir / "esm2_35m_hard_neg.npy")

    X_tr_8m = np.load(emb_dir / "esm2_8m_train.npy")
    X_val_8m = np.load(emb_dir / "esm2_8m_validation.npy")
    X_test_8m = np.load(emb_dir / "esm2_8m_test.npy")
    X_ch_8m = np.load(emb_dir / "esm2_8m_hard_neg.npy")

    y_train = train_df["label"].to_numpy()

    # 1. ESM2-LR (35M)
    scaler_35m = StandardScaler()
    X_tr_35m_sc = scaler_35m.fit_transform(X_tr_35m)
    X_val_35m_sc = scaler_35m.transform(X_val_35m)
    X_test_35m_sc = scaler_35m.transform(X_test_35m)
    X_ch_35m_sc = scaler_35m.transform(X_ch_35m)

    clf_lr_35m = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
    clf_lr_35m.fit(X_tr_35m_sc, y_train)
    esm2_lr_35m_val_scores = clf_lr_35m.predict_proba(X_val_35m_sc)[:, 1]
    esm2_lr_35m_test_scores = clf_lr_35m.predict_proba(X_test_35m_sc)[:, 1]
    esm2_lr_35m_ch_scores = clf_lr_35m.predict_proba(X_ch_35m_sc)[:, 1]

    # 2. ESM2-MLP (35M)
    clf_mlp_35m = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=42)
    clf_mlp_35m.fit(X_tr_35m_sc, y_train)
    esm2_mlp_35m_val_scores = clf_mlp_35m.predict_proba(X_val_35m_sc)[:, 1]
    esm2_mlp_35m_test_scores = clf_mlp_35m.predict_proba(X_test_35m_sc)[:, 1]
    esm2_mlp_35m_ch_scores = clf_mlp_35m.predict_proba(X_ch_35m_sc)[:, 1]

    # 3. ESM2-LR (8M)
    scaler_8m = StandardScaler()
    X_tr_8m_sc = scaler_8m.fit_transform(X_tr_8m)
    X_val_8m_sc = scaler_8m.transform(X_val_8m)
    X_test_8m_sc = scaler_8m.transform(X_test_8m)
    X_ch_8m_sc = scaler_8m.transform(X_ch_8m)

    clf_lr_8m = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
    clf_lr_8m.fit(X_tr_8m_sc, y_train)
    esm2_lr_8m_val_scores = clf_lr_8m.predict_proba(X_val_8m_sc)[:, 1]
    esm2_lr_8m_test_scores = clf_lr_8m.predict_proba(X_test_8m_sc)[:, 1]
    esm2_lr_8m_ch_scores = clf_lr_8m.predict_proba(X_ch_8m_sc)[:, 1]

    # Assemble Models Dictionary
    models = {
        "BLASTP": {
            "val_scores": np.array([blast_val_scores_dict[s] for s in val_ids]),
            "test_scores": np.array([blast_test_scores_dict[s] for s in test_ids]),
            "ch_scores": np.array([blast_ch_scores_dict[s] for s in ch_ids]),
        },
        "MMseqs2": {
            "val_scores": np.array([mm_val_scores_dict[s] for s in val_ids]),
            "test_scores": np.array([mm_test_scores_dict[s] for s in test_ids]),
            "ch_scores": np.array([mm_ch_scores_dict[s] for s in ch_ids]),
        },
        "HMM-A (Whole-Family)": {
            "val_scores": np.array([hmm_a_val_dict[s] for s in val_ids]),
            "test_scores": np.array([hmm_a_test_dict[s] for s in test_ids]),
            "ch_scores": np.array([hmm_a_ch_dict[s] for s in ch_ids]),
        },
        "HMM-B (Cluster-Specific)": {
            "val_scores": np.array([hmm_b_val_dict[s] for s in val_ids]),
            "test_scores": np.array([hmm_b_test_dict[s] for s in test_ids]),
            "ch_scores": np.array([hmm_b_ch_dict[s] for s in ch_ids]),
        },
        "HMM-C (Domain-HMM)": {
            "val_scores": np.array([hmm_c_val_dict[s] for s in val_ids]),
            "test_scores": np.array([hmm_c_test_dict[s] for s in test_ids]),
            "ch_scores": np.array([hmm_c_ch_dict[s] for s in ch_ids]),
        },
        "ESM2-LR (35M)": {
            "val_scores": esm2_lr_35m_val_scores,
            "test_scores": esm2_lr_35m_test_scores,
            "ch_scores": esm2_lr_35m_ch_scores,
        },
        "ESM2-MLP (35M)": {
            "val_scores": esm2_mlp_35m_val_scores,
            "test_scores": esm2_mlp_35m_test_scores,
            "ch_scores": esm2_mlp_35m_ch_scores,
        },
        "ESM2-LR (8M)": {
            "val_scores": esm2_lr_8m_val_scores,
            "test_scores": esm2_lr_8m_test_scores,
            "ch_scores": esm2_lr_8m_ch_scores,
        },
    }

    # --- Step E: Evaluate Each Model Across Validation, All Strata, and Challenge Set ---
    print("\n--> Evaluating all models across strata and challenge set...")
    benchmark_rows = []

    for model_name, m_data in models.items():
        v_scores = m_data["val_scores"]
        t_scores = m_data["test_scores"]
        ch_scores = m_data["ch_scores"]

        # Calibrate 5% FDR threshold on validation
        th, val_achieved_fdr, val_rec = find_fdr_threshold(y_val, v_scores, target_fdr=0.05)

        # Hard-negative FPR (all are negative, label=0)
        ch_pred = (ch_scores >= th).astype(int)
        ch_fp = int(np.sum(ch_pred == 1))
        ch_fpr = (ch_fp / len(ch_scores)) * 100.0

        # Evaluate across strata
        row_dict = {
            "model": model_name,
            "val_threshold": round(float(th), 4),
            "val_fdr": round(float(val_achieved_fdr * 100), 2),
            "val_recall": round(float(val_rec * 100), 2),
            "hard_neg_fp": ch_fp,
            "hard_neg_total": len(ch_scores),
            "hard_neg_fpr": round(float(ch_fpr), 2),
        }

        for s_name, s_mask in strata.items():
            y_sub = y_test[s_mask]
            t_sub = t_scores[s_mask]
            met = compute_metrics(y_sub, t_sub, th)

            row_dict[f"{s_name}_auprc"] = round(met["auprc"], 4)
            row_dict[f"{s_name}_recall"] = round(met["recall"] * 100, 2)
            row_dict[f"{s_name}_precision"] = round(met["precision"] * 100, 2)
            row_dict[f"{s_name}_mcc"] = round(met["mcc"], 4)
            row_dict[f"{s_name}_fpr"] = round(met["fpr"] * 100, 2)

        benchmark_rows.append(row_dict)
        print(f"  [{model_name}] Overall: Rec={row_dict['Overall_recall']}%, AUPRC={row_dict['Overall_auprc']} | Remote-20: Rec={row_dict['Remote20_recall']}% | Hard-Neg FPR: {row_dict['hard_neg_fpr']}%")

    # Save TSV
    out_tsv = ROOT_DIR / "benchmark/tables/phase1_benchmark_v2.tsv"
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(benchmark_rows).write_csv(out_tsv, separator="\t")
    print(f"\nSaved Phase-1 Benchmark v2 TSV: {out_tsv}")

    # Generate Markdown Report
    out_md = ROOT_DIR / "benchmark/reports/phase1_benchmark_v2.md"
    generate_phase1_markdown_report(benchmark_rows, out_md)
    print(f"Saved Phase-1 Benchmark v2 Report: {out_md}")


def generate_phase1_markdown_report(rows: List[dict], out_md: Path):
    md = []
    md.append("# DeepISE Scientific Audit v1.1 - Phase-1 Benchmark v2 Report")
    md.append("")
    md.append("> **Dataset**: `cluster30_v2` (Zero cross-split homology leakage, strictly cleaned negatives).")
    md.append("> **Fair Calibration**: All thresholds calibrated on validation split at target FDR = 5%.")
    md.append("")
    md.append("## 1. Primary Benchmark Table (Section 46)")
    md.append("")
    md.append("| Method | Overall AUPRC | Overall Recall@5%FDR | Remote-30 Recall | Remote-20 Recall | No-Full-Length Recall | Hard-Neg FPR | Val FDR |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for r in rows:
        md.append(
            f"| **{r['model']}** | {r['Overall_auprc']:.4f} | {r['Overall_recall']:.2f}% | "
            f"{r['Remote30_recall']:.2f}% | {r['Remote20_recall']:.2f}% | {r['NoFullLength_recall']:.2f}% | "
            f"**{r['hard_neg_fpr']:.2f}%** | {r['val_fdr']:.2f}% |"
        )

    md.append("")
    md.append("## 2. Detailed Stratified Performance Matrix")
    md.append("")
    md.append("| Method | Overall MCC | Overall Prec | Remote-20 AUPRC | Remote-20 Prec | No-Full AUPRC | Hard-Neg FP / Total |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for r in rows:
        md.append(
            f"| {r['model']} | {r['Overall_mcc']:.4f} | {r['Overall_precision']:.2f}% | "
            f"{r['Remote20_auprc']:.4f} | {r['Remote20_precision']:.2f}% | {r['NoFullLength_auprc']:.4f} | "
            f"{r['hard_neg_fp']} / {r['hard_neg_total']} |"
        )

    md.append("")
    md.append("## 3. Pre-registered Decision Gate Analysis (Section 48 - 50)")
    md.append("")
    # Find ESM2-LR 35M and strongest HMM
    esm2_lr = next(r for r in rows if r['model'] == "ESM2-LR (35M)")
    hmm_models = [r for r in rows if "HMM" in r['model']]
    strongest_hmm = max(hmm_models, key=lambda x: x["Remote20_recall"])

    delta_remote20 = esm2_lr["Remote20_recall"] - strongest_hmm["Remote20_recall"]
    hard_neg_pass = esm2_lr["hard_neg_fpr"] <= 5.0

    md.append(f"- **ESM2-LR (35M) Remote-20 Recall**: {esm2_lr['Remote20_recall']:.2f}%")
    md.append(f"- **Strongest HMM Baseline ({strongest_hmm['model']}) Remote-20 Recall**: {strongest_hmm['Remote20_recall']:.2f}%")
    md.append(f"- **Δ Remote-20 Recall Gain**: **{delta_remote20:+.2f} percentage points** (Requirement: $\\ge +10$ pp for UNAMBIGUOUS GO, $+3\\sim+10$ pp for CONDITIONAL GO)")
    md.append(f"- **Hard-Negative FPR**: **{esm2_lr['hard_neg_fpr']:.2f}%** (Requirement: $\\le 5.0\%$, Pass: **{hard_neg_pass}**)")
    md.append("")

    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
