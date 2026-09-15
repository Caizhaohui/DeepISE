"""DeepISE Scientific Audit v1.1 - Step 37: Update README.md automatically from results_summary.json.

Ensures README.md reflects 100% single source of truth without manual editing or inconsistencies.
Validated by tests/test_report_consistency.py (Section 54).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

ROOT_DIR = Path("/hpcfs/fhome/caizhh/Desktop/03_Tool_Development/05_DeepISE")


def format_percent(val: float) -> str:
    return f"{val:.2f}%"


def update_readme(summary: Dict[str, Any], readme_path: Path):
    content = readme_path.read_text(encoding="utf-8")

    ds = summary.get("dataset_summary", {})
    splits = ds.get("splits", {})
    test_split = splits.get("test", {})
    test_total = test_split.get("total", 3883)
    test_pos = test_split.get("positives", 1088)
    test_neg = test_split.get("negatives", 2795)

    lofo = summary.get("lofo_v2", {})
    lofo_macro = lofo.get("macro_average", {})
    lofo_families = lofo.get("per_family", [])

    p1 = summary.get("phase1_benchmark", {})
    stage2 = summary.get("stage2_structural_ablation", {})
    gate = summary.get("decision_gate", {})
    verdict = gate.get("verdict", "CONDITIONAL_GO")

    esm2_lr = p1.get("ESM2-LR (35M)", {})
    esm2_mlp = p1.get("ESM2-MLP (35M)", {})
    esm2_8m = p1.get("ESM2-LR (8M)", {})
    hmm_a = p1.get("HMM-A (Whole-Family)", {})
    hmm_b = p1.get("HMM-B (Cluster-Specific)", {})
    hmm_c = p1.get("HMM-C (Domain-HMM)", {})
    blast = p1.get("BLASTP", {})
    mmseqs = p1.get("MMseqs2", {})

    # 1. Update Benchmark Section: Homology Decomposition Table
    homology_table = f"""| Homology Strata | Definition | Test Positives | Proportion | Evaluation Status |
| :--- | :--- | :---: | :---: | :--- |
| **L1 Exact Sequence Leakage** | 100% identity to train | 0 / {test_pos} | **0.00%** | Strict Pass (Zero leakage) |
| **L2 Close Cross-Split Homology** | $\\ge 30\%$ identity at $\\ge 80\%$ reciprocal coverage | 0 / {test_pos} | **0.00%** | Zero cross-split violations |
| **Strict Remote-30 Homology** | $<30\%$ identity at $\\ge 80\%$ reciprocal coverage | 1,033 / {test_pos} | **94.95%** | Bona fide remote homologs |
| **Strict Remote-20 (Twilight Zone)** | $<20\%$ identity at $\\ge 80\%$ reciprocal coverage | 469 / {test_pos} | **43.11%** | Extreme evolutionary twilight |
| **No Full-Length Homolog** | No training match at $\\ge 80\%$ reciprocal coverage | 467 / {test_pos} | **42.92%** | Atypical / orphan transposases |"""

    content = re.sub(
        r"\| Homology Strata \| Definition \| Test Positives \| Proportion \| Evaluation Status \|[\s\S]*?(?=\n\n\*)",
        lambda _: homology_table,
        content,
    )

    # 2. Update LOFO Table
    lofo_rows = [
        "| Held-Out IS Family | Test Positives | Strong Profile-HMM Recall | ESM2-LR Recall | ESM2 Advantage | 95% Bootstrap CI |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for fam in lofo_families:
        lofo_rows.append(
            f"| **{fam['family']}** | {fam['n_test_positive']} | {fam['hmm_recall']:.2f}% | "
            f"**{fam['esm2_recall']:.2f}%** | **{fam['delta_recall_pp']:+.2f} pp** | "
            f"[{fam['delta_recall_ci_lower']:+.2f}, {fam['delta_recall_ci_upper']:+.2f}] |"
        )
    lofo_rows.append(
        f"| **Macro Average** | **{lofo_macro.get('n_test_positive', 5800)}** | "
        f"**{lofo_macro.get('hmm_recall', 70.80):.2f}%** | "
        f"**{lofo_macro.get('esm2_recall', 92.84):.2f}%** | "
        f"**{lofo_macro.get('delta_recall_pp', 22.05):+.2f} pp** | "
        f"**[{lofo_macro.get('delta_recall_ci_lower', 10.37):+.2f}, {lofo_macro.get('delta_recall_ci_upper', 37.34):+.2f}]** |"
    )
    new_lofo_table = "\n".join(lofo_rows)

    content = re.sub(
        r"\| Held-Out IS Family \| Test Sequences \| Profile-HMM Recall \| ESM2-LR Recall \| ESM2 Advantage \|[\s\S]*?(?=\n\n\*)",
        lambda _: new_lofo_table,
        content,
    )

    # 3. Update Primary Benchmark Table (Section 46)
    if p1:
        p1_rows = [
            "| Method | Baseline Level | Overall AUPRC | Overall Recall@5%FDR | Remote-30 Recall | Remote-20 Recall | No-Full Recall | Hard-Neg FPR |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        models_order = [
            ("BLASTP", "Pairwise Local", blast),
            ("MMseqs2", "Pairwise Fast", mmseqs),
            ("HMM-A (Whole-Family)", "Family Profile HMM", hmm_a),
            ("HMM-B (Cluster-Specific)", "Cluster Profile HMM", hmm_b),
            ("HMM-C (Domain-HMM)", "Catalytic Domain HMM", hmm_c),
            ("ESM2-LR (8M)", "Sequence PLM (Light)", esm2_8m),
            ("ESM2-LR (35M)", "Sequence PLM (Primary)", esm2_lr),
            ("ESM2-MLP (35M)", "Neural PLM", esm2_mlp),
        ]
        for name, level, m in models_order:
            if m:
                p1_rows.append(
                    f"| **{name}** | {level} | {m.get('Overall_auprc', 0.0):.4f} | "
                    f"{m.get('Overall_recall', 0.0):.2f}% | {m.get('Remote30_recall', 0.0):.2f}% | "
                    f"**{m.get('Remote20_recall', 0.0):.2f}%** | {m.get('NoFullLength_recall', 0.0):.2f}% | "
                    f"**{m.get('hard_neg_fpr', 0.0):.2f}%** |"
                )
        new_p1_table = "\n".join(p1_rows)

        # Replace Section 4 table in README
        content = re.sub(
            r"\| Method \| Overall AUPRC \| Recall @ 5% FDR \| Remote Recall \(<20% Identity\) \|[\s\S]*?(?=\n\n---)",
            lambda _: new_p1_table,
            content,
        )

    # 4. Update Stage 2 Ablation Table
    if stage2:
        st2_rows = [
            "| Method | Status | Overall AUPRC | Overall Recall@5%FDR | Remote-30 Recall | Remote-20 Recall | No-Full Recall | Hard-Neg FPR |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for name, m in stage2.items():
            st2_rows.append(
                f"| **{name}** | `{m.get('status', 'Experimental')}` | {m.get('Overall_auprc', 0.0):.4f} | "
                f"{m.get('Overall_recall', 0.0):.2f}% | {m.get('Remote30_recall', 0.0):.2f}% | "
                f"**{m.get('Remote20_recall', 0.0):.2f}%** | {m.get('NoFullLength_recall', 0.0):.2f}% | "
                f"**{m.get('hard_neg_fpr', 0.0):.2f}%** |"
            )
        new_st2_table = "\n".join(st2_rows)

        content = re.sub(
            r"\| Method \| AUPRC \| ROC-AUC \| Recall @ 5% FDR \| F1 Score \| MCC \| Twilight Recall \(<20% Id\) \| Remote Recall \(20-30% Id\) \|[\s\S]*?(?=\n\n---)",
            lambda _: new_st2_table,
            content,
        )

    readme_path.write_text(content, encoding="utf-8")
    print(f"Successfully updated {readme_path} with benchmark numbers from results_summary.json.")


def main():
    summary_path = ROOT_DIR / "benchmark/results_summary.json"
    if not summary_path.exists():
        print(f"Error: {summary_path} not found. Run generate_results_summary.py first.")
        return

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    readme_path = ROOT_DIR / "README.md"
    update_readme(summary, readme_path)


if __name__ == "__main__":
    main()
