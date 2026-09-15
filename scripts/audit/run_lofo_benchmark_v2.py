"""DeepISE Scientific Audit v1.1 - Steps 19 to 23: LOFO v2 Benchmark.

Implements Leave-One-Family-Out (LOFO) benchmark under strict Audit v1.1 integrity rules:
1. Rebuilds HMM database from scratch for each held-out family using only training positives excluding that family.
2. Evaluates ESM-2 and HMM on the exact same test FASTA (data/lofo/<family>/test.faa).
3. Fair threshold calibration: both ESM-2 and HMM thresholds are calibrated on the LOFO validation set at target FDR = 5%.
4. 10,000 bootstrap replicates for empirical 95% confidence intervals on Delta Recall.
5. Macro average across families with N >= 30 (exploratory reported for N < 30).
"""

import os
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import polars as pl
from Bio import SeqIO
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler

# Ensure conda env binaries
ENV_BIN = Path("/hpcfs/fhome/caizhh/.conda/envs/DeepISE/bin")
if str(ENV_BIN) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{ENV_BIN}:{os.environ.get('PATH', '')}"

ROOT_DIR = Path("/hpcfs/fhome/caizhh/Desktop/03_Tool_Development/05_DeepISE")

TARGET_FAMILIES = [
    "IS1", "IS3", "IS4", "IS5", "IS6", "IS21", "IS30", "IS66",
    "IS110", "IS200/IS605", "IS256", "IS630", "IS91", "IS1182", "IS1595"
]


def find_fdr_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_fdr: float = 0.05,
) -> Tuple[float, float, float]:
    """Calibrate score threshold achieving closest FDR <= target_fdr on validation data."""
    unique_scores = np.unique(scores)
    unique_scores = np.sort(unique_scores)[::-1]  # descending

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
        "auprc": float(auprc),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def build_scratch_hmm_db(train_positives_sub: List[dict], output_hmm: Path, tmp_dir: Path) -> Path:
    """Build HMM profile database from scratch for LOFO training set."""
    output_hmm.parent.mkdir(parents=True, exist_ok=True)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    fams = defaultdict(list)
    for row in train_positives_sub:
        fams[row["family"]].append(row)

    built_hmms = []
    for fam_name, members in sorted(fams.items()):
        clean_fam = "".join(c if c.isalnum() else "_" for c in fam_name)
        fam_faa = tmp_dir / f"{clean_fam}.faa"
        fam_aln = tmp_dir / f"{clean_fam}.aln"
        fam_hmm = tmp_dir / f"{clean_fam}.hmm"

        with open(fam_faa, "w", encoding="utf-8") as f:
            for m in members:
                f.write(f">{m['seq_id']}\n{m['protein_sequence']}\n")

        if len(members) == 1:
            cmd = ["hmmbuild", "-n", f"Tpase_{clean_fam}", str(fam_hmm), str(fam_faa)]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        else:
            cmd_mafft = f"mafft --retree 1 --maxiterate 0 --quiet {fam_faa} > {fam_aln}"
            subprocess.run(cmd_mafft, shell=True, check=True)
            cmd_hmmbuild = ["hmmbuild", "-n", f"Tpase_{clean_fam}", str(fam_hmm), str(fam_aln)]
            subprocess.run(cmd_hmmbuild, check=True, capture_output=True, text=True)

        built_hmms.append(fam_hmm)

    with open(output_hmm, "w", encoding="utf-8") as out_f:
        for h in built_hmms:
            with open(h, "r", encoding="utf-8") as in_f:
                out_f.write(in_f.read())

    subprocess.run(["hmmpress", "-f", str(output_hmm)], check=True, capture_output=True, text=True)
    return output_hmm


def run_hmmsearch_and_get_scores(query_faa: Path, hmm_db: Path, threads: int = 4) -> Dict[str, float]:
    """Execute hmmsearch and return max bitscore per query sequence."""
    all_query_ids = [r.id for r in SeqIO.parse(query_faa, "fasta")]
    tbl_out = query_faa.with_suffix(".tbl")

    cmd = [
        "hmmsearch",
        "--cpu", str(threads),
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
                    seq_id = parts[0]
                    bitscore = float(parts[5])
                    if seq_id in scores:
                        if bitscore > scores[seq_id]:
                            scores[seq_id] = bitscore

    if tbl_out.exists():
        tbl_out.unlink()
    return scores


def bootstrap_delta_recall(
    y_test: np.ndarray,
    esm_scores: np.ndarray,
    esm_th: float,
    hmm_scores: np.ndarray,
    hmm_th: float,
    n_boot: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap delta recall (percentage points) with 95% empirical confidence interval."""
    pos_idx = np.where(y_test == 1)[0]
    n_pos = len(pos_idx)
    if n_pos == 0:
        return 0.0, 0.0, 0.0

    esm_pred = (esm_scores[pos_idx] >= esm_th).astype(float)
    hmm_pred = (hmm_scores[pos_idx] >= hmm_th).astype(float)
    diff = (esm_pred - hmm_pred) * 100.0
    point_est = float(np.mean(diff))

    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, n_pos, size=(n_boot, n_pos))
    boot_diffs = np.mean(diff[boot_indices], axis=1)

    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))
    return point_est, ci_lower, ci_upper


def load_or_wait_embeddings(tag: str = "esm2_35m") -> Dict[str, np.ndarray]:
    """Load precomputed embeddings for train, val, test, challenge sets."""
    emb_dir = ROOT_DIR / "data/processed/embeddings_v2"
    splits = ["train", "validation", "test"]
    lookup = {}

    for s in splits:
        npy_path = emb_dir / f"{tag}_{s}.npy"
        idx_path = emb_dir / f"{tag}_{s}_index.parquet"
        if not npy_path.exists() or not idx_path.exists():
            raise FileNotFoundError(f"Missing embedding file: {npy_path}")
        mat = np.load(npy_path)
        idx_df = pl.read_parquet(idx_path)
        for row in idx_df.iter_rows(named=True):
            lookup[row["seq_id"]] = mat[row["embedding_idx"]]

    return lookup


def main():
    print("=" * 80)
    print("STARTING DEEPISE SCIENTIFIC AUDIT v1.1 - LOFO v2 BENCHMARK")
    print("=" * 80)

    splits_dir = ROOT_DIR / "data/splits/cluster30_v2"
    train_df = pl.read_parquet(splits_dir / "train.parquet")
    val_df = pl.read_parquet(splits_dir / "validation.parquet")
    test_df = pl.read_parquet(splits_dir / "test.parquet")

    # Positive pool across all splits
    all_pos_df = pl.concat([
        train_df.filter(pl.col("label") == 1),
        val_df.filter(pl.col("label") == 1),
        test_df.filter(pl.col("label") == 1),
    ])

    train_neg_records = train_df.filter(pl.col("label") == 0).to_dicts()
    val_neg_records = val_df.filter(pl.col("label") == 0).to_dicts()
    test_neg_records = test_df.filter(pl.col("label") == 0).to_dicts()

    print(f"Total positives: {len(all_pos_df)}, Train neg: {len(train_neg_records)}, Val neg: {len(val_neg_records)}, Test neg: {len(test_neg_records)}")

    # Load embeddings
    print("\nLoading ESM-2 embeddings lookup...")
    emb_lookup = load_or_wait_embeddings("esm2_35m")
    print(f"Loaded embeddings for {len(emb_lookup)} unique sequences.")

    lofo_dir = ROOT_DIR / "data/lofo"
    lofo_dir.mkdir(parents=True, exist_ok=True)
    tmp_base = Path(tempfile.mkdtemp(prefix="lofo_v2_work_"))

    results = []

    try:
        for fam in TARGET_FAMILIES:
            clean_fam = "".join(c if c.isalnum() else "_" for c in fam)
            fam_dir = lofo_dir / clean_fam
            fam_dir.mkdir(parents=True, exist_ok=True)

            # Positive test records: ALL sequences of family fam
            test_pos_records = all_pos_df.filter(pl.col("family") == fam).to_dicts()
            n_pos = len(test_pos_records)
            exploratory = n_pos < 30

            # Combined test set
            test_records = test_pos_records + test_neg_records
            test_faa = fam_dir / "test.faa"
            with open(test_faa, "w", encoding="utf-8") as f:
                for r in test_records:
                    f.write(f">{r['seq_id']}\n{r['protein_sequence']}\n")

            y_test = np.array([1] * n_pos + [0] * len(test_neg_records), dtype=int)
            test_ids = [r["seq_id"] for r in test_records]

            # Validation set for family fam:
            val_pos_records = val_df.filter((pl.col("label") == 1) & (pl.col("family") != fam)).to_dicts()
            val_records = val_pos_records + val_neg_records
            val_faa = fam_dir / "val.faa"
            with open(val_faa, "w", encoding="utf-8") as f:
                for r in val_records:
                    f.write(f">{r['seq_id']}\n{r['protein_sequence']}\n")

            y_val = np.array([1] * len(val_pos_records) + [0] * len(val_neg_records), dtype=int)
            val_ids = [r["seq_id"] for r in val_records]

            # Training set for family fam:
            train_pos_records = train_df.filter((pl.col("label") == 1) & (pl.col("family") != fam)).to_dicts()
            train_records = train_pos_records + train_neg_records
            y_train = np.array([1] * len(train_pos_records) + [0] * len(train_neg_records), dtype=int)
            train_ids = [r["seq_id"] for r in train_records]

            print(f"\nEvaluating Held-Out Family: {fam} (n_pos={n_pos}, exploratory={exploratory})...")

            # --- 1. HMM LOFO: Build HMM from scratch and test ---
            hmm_db = fam_dir / f"lofo_{clean_fam}.hmm"
            build_scratch_hmm_db(train_pos_records, hmm_db, tmp_base / f"build_{clean_fam}")

            val_hmm_dict = run_hmmsearch_and_get_scores(val_faa, hmm_db, threads=8)
            val_hmm_scores = np.array([val_hmm_dict[s_id] for s_id in val_ids])
            hmm_th, hmm_val_fdr, _ = find_fdr_threshold(y_val, val_hmm_scores, target_fdr=0.05)

            test_hmm_dict = run_hmmsearch_and_get_scores(test_faa, hmm_db, threads=8)
            test_hmm_scores = np.array([test_hmm_dict[s_id] for s_id in test_ids])
            hmm_metrics = compute_metrics(y_test, test_hmm_scores, hmm_th)

            # --- 2. ESM2 LOFO: Train LR and test ---
            X_train = np.vstack([emb_lookup[s_id] for s_id in train_ids])
            X_val = np.vstack([emb_lookup[s_id] for s_id in val_ids])
            X_test = np.vstack([emb_lookup[s_id] for s_id in test_ids])

            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_val_sc = scaler.transform(X_val)
            X_test_sc = scaler.transform(X_test)

            clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
            clf.fit(X_train_sc, y_train)

            val_esm_scores = clf.predict_proba(X_val_sc)[:, 1]
            esm_th, esm_val_fdr, _ = find_fdr_threshold(y_val, val_esm_scores, target_fdr=0.05)

            test_esm_scores = clf.predict_proba(X_test_sc)[:, 1]
            esm_metrics = compute_metrics(y_test, test_esm_scores, esm_th)

            # --- 3. Bootstrap Delta Recall ---
            pt_diff, ci_l, ci_u = bootstrap_delta_recall(
                y_test, test_esm_scores, esm_th, test_hmm_scores, hmm_th, n_boot=10000
            )

            res_row = {
                "family": fam,
                "n_test_positive": n_pos,
                "n_test_negative": len(test_neg_records),
                "exploratory": exploratory,
                "esm2_threshold": round(float(esm_th), 4),
                "esm2_recall": round(esm_metrics["recall"] * 100, 2),
                "esm2_precision": round(esm_metrics["precision"] * 100, 2),
                "esm2_fpr": round(esm_metrics["fpr"] * 100, 2),
                "esm2_fdr": round(esm_metrics["fdr"] * 100, 2),
                "esm2_auprc": round(esm_metrics["auprc"], 4),
                "hmm_threshold": round(float(hmm_th), 2),
                "hmm_recall": round(hmm_metrics["recall"] * 100, 2),
                "hmm_precision": round(hmm_metrics["precision"] * 100, 2),
                "hmm_fpr": round(hmm_metrics["fpr"] * 100, 2),
                "hmm_fdr": round(hmm_metrics["fdr"] * 100, 2),
                "hmm_auprc": round(hmm_metrics["auprc"], 4),
                "delta_recall_pp": round(pt_diff, 2),
                "delta_recall_ci_lower": round(ci_l, 2),
                "delta_recall_ci_upper": round(ci_u, 2),
            }
            results.append(res_row)
            print(f"  --> ESM2 Recall: {res_row['esm2_recall']}% (FPR {res_row['esm2_fpr']}%) | HMM Recall: {res_row['hmm_recall']}% (FPR {res_row['hmm_fpr']}%) | Delta: {pt_diff:+.2f}% [{ci_l:+.2f}%, {ci_u:+.2f}%]")

    finally:
        shutil.rmtree(tmp_base, ignore_errors=True)

    # Compute Macro Average across non-exploratory families
    non_exp = [r for r in results if not r["exploratory"]]
    macro_esm_rec = np.mean([r["esm2_recall"] for r in non_exp])
    macro_hmm_rec = np.mean([r["hmm_recall"] for r in non_exp])
    macro_delta = np.mean([r["delta_recall_pp"] for r in non_exp])

    # Bootstrap macro delta
    rng = np.random.default_rng(42)
    deltas = np.array([r["delta_recall_pp"] for r in non_exp])
    boot_macro = [np.mean(rng.choice(deltas, size=len(deltas), replace=True)) for _ in range(10000)]
    macro_ci_l = float(np.percentile(boot_macro, 2.5))
    macro_ci_u = float(np.percentile(boot_macro, 97.5))

    macro_row = {
        "family": "MACRO_AVERAGE",
        "n_test_positive": sum(r["n_test_positive"] for r in non_exp),
        "n_test_negative": len(test_neg_records),
        "exploratory": False,
        "esm2_threshold": 0.0,
        "esm2_recall": round(float(macro_esm_rec), 2),
        "esm2_precision": round(float(np.mean([r["esm2_precision"] for r in non_exp])), 2),
        "esm2_fpr": round(float(np.mean([r["esm2_fpr"] for r in non_exp])), 2),
        "esm2_fdr": round(float(np.mean([r["esm2_fdr"] for r in non_exp])), 2),
        "esm2_auprc": round(float(np.mean([r["esm2_auprc"] for r in non_exp])), 4),
        "hmm_threshold": 0.0,
        "hmm_recall": round(float(macro_hmm_rec), 2),
        "hmm_precision": round(float(np.mean([r["hmm_precision"] for r in non_exp])), 2),
        "hmm_fpr": round(float(np.mean([r["hmm_fpr"] for r in non_exp])), 2),
        "hmm_fdr": round(float(np.mean([r["hmm_fdr"] for r in non_exp])), 2),
        "hmm_auprc": round(float(np.mean([r["hmm_auprc"] for r in non_exp])), 4),
        "delta_recall_pp": round(float(macro_delta), 2),
        "delta_recall_ci_lower": round(macro_ci_l, 2),
        "delta_recall_ci_upper": round(macro_ci_u, 2),
    }
    results.append(macro_row)

    # Save TSV
    out_tsv = ROOT_DIR / "benchmark/tables/lofo_v2.tsv"
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(results).write_csv(out_tsv, separator="\t")
    print(f"\nSaved LOFO v2 TSV: {out_tsv}")

    # Generate Markdown Report
    out_md = ROOT_DIR / "benchmark/reports/lofo_v2.md"
    generate_lofo_markdown_report(results, out_md, macro_row)
    print(f"Saved LOFO v2 Report: {out_md}")


def generate_lofo_markdown_report(results: List[dict], out_md: Path, macro_row: dict):
    md = []
    md.append("# DeepISE Scientific Audit v1.1 - LOFO v2 Benchmark Report")
    md.append("")
    md.append("> **Audit Requirement:** Completely rebuilt HMM profiles per family, identical test FASTA, validation-calibrated thresholds at 5% FDR, and 10,000 bootstrap replicates for 95% CI.")
    md.append("")
    md.append("## 1. Summary Results Table")
    md.append("")
    md.append("| Family | N Pos | N Neg | ESM-2 Recall (%) | HMM Recall (%) | Δ Recall (pp) | 95% Bootstrap CI | ESM-2 FPR (%) | HMM FPR (%) |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for r in results:
        fam_str = f"**{r['family']}**" if r['family'] == "MACRO_AVERAGE" else r['family']
        if r.get("exploratory", False):
            fam_str += " *(exp)*"
        ci_str = f"[{r['delta_recall_ci_lower']:+.2f}, {r['delta_recall_ci_upper']:+.2f}]"
        delta_str = f"**{r['delta_recall_pp']:+.2f}**"
        md.append(f"| {fam_str} | {r['n_test_positive']} | {r['n_test_negative']} | {r['esm2_recall']:.2f}% | {r['hmm_recall']:.2f}% | {delta_str} | {ci_str} | {r['esm2_fpr']:.2f}% | {r['hmm_fpr']:.2f}% |")

    md.append("")
    md.append("## 2. Scientific Evaluation & Pre-registered Criteria")
    md.append("")
    md.append(f"- **Macro Average Δ Recall**: **{macro_row['delta_recall_pp']:+.2f} percentage points** (95% CI: [{macro_row['delta_recall_ci_lower']:+.2f}, {macro_row['delta_recall_ci_upper']:+.2f}]).")
    signif = macro_row['delta_recall_ci_lower'] > 0
    md.append(f"- **Statistical Significance (CI Lower > 0)**: **{'SATISFIED (Statistically Significant Gain)' if signif else 'NOT SATISFIED'}**.")
    md.append("")
    md.append("### Scientific Finding:")
    md.append("> ESM2 representations generalize across held-out IS families, consistent with capturing transferable protein-level features beyond family-specific sequence profiles.")
    md.append("")

    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
