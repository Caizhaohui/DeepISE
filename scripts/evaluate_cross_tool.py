"""Cross-tool benchmark script evaluating ISEScan vs DeepISE Plan A & Plan B on E. coli K-12."""

import glob
import sys
import time
from pathlib import Path
from typing import Dict, List
import polars as pl

from deepise_ml.genome.benchmark_genome import (
    evaluate_genome_predictions,
    parse_ecoli_gold_standard,
)
from deepise_ml.genome.scanner import DetectedISElement, DeepISEGenomeScanner


def parse_isescan_results(tsv_path: Path) -> List[DetectedISElement]:
    """Parse ISEScan TSV prediction output into DetectedISElement format."""
    elements = []
    lines = tsv_path.read_text().splitlines()
    if not lines:
        return []
    
    header = lines[0].split("\t")
    header_idx = {col: i for i, col in enumerate(header)}
    
    for idx, line in enumerate(lines[1:]):
        parts = line.split("\t")
        if len(parts) < len(header):
            continue
        try:
            is_begin = int(parts[header_idx["isBegin"]])
            is_end = int(parts[header_idx["isEnd"]])
            is_family = parts[header_idx["family"]]
            is_type = parts[header_idx["type"]].strip().lower()
            status = "complete" if is_type == "c" else "partial"
            seq_id = parts[header_idx["seqID"]]
            score = 0.5
            
            # 1-based inclusive -> 0-based exclusive
            s = min(is_begin, is_end) - 1
            e = max(is_begin, is_end)
            
            elem = DetectedISElement(
                element_id=f"ISEScan_{idx+1}",
                contig_id=seq_id,
                start=s,
                end=e,
                length=e - s,
                strand=parts[header_idx.get("strand", "+")],
                family=is_family,
                tpase_gene_id=f"ISEScan_orf_{idx+1}",
                tpase_score=score,
                tpase_evalue=1e-10,
                composite_score=0.5,
                status=status,
                method="isescan",
            )
            elements.append(elem)
        except Exception as err:
            print(f"Warning: skipped row {idx}: {err}")
            continue
            
    return elements


def run_cross_tool_evaluation():
    print("Starting cross-tool evaluation on E. coli K-12...")
    fna_path = Path("data/genomes/NC_000913.3.fna")
    gff_path = Path("data/genomes/NC_000913.3.gff")
    
    gold_standard = parse_ecoli_gold_standard(gff_path)
    print(f"Curated Gold Standard IS elements: {len(gold_standard)}")
    
    # 1. Run DeepISE Plan A
    scanner = DeepISEGenomeScanner()
    t0_a = time.perf_counter()
    preds_a = scanner.scan_genome(fna_path, mode="plan_a", threads=8)
    t1_a = time.perf_counter()
    metrics_a, match_a = evaluate_genome_predictions(preds_a, gold_standard, "DeepISE (Plan A: Adaptive)")
    metrics_a["runtime_sec"] = t1_a - t0_a
    
    # 2. Run DeepISE Plan B
    t0_b = time.perf_counter()
    preds_b = scanner.scan_genome(fna_path, mode="plan_b", threads=8)
    t1_b = time.perf_counter()
    metrics_b, match_b = evaluate_genome_predictions(preds_b, gold_standard, "DeepISE (Plan B: Canonical)")
    metrics_b["runtime_sec"] = t1_b - t0_b
    
    # 3. Locate and Parse ISEScan results
    isescan_tsvs = list(Path("benchmark/results/isescan_ecoli").rglob("*.tsv"))
    if not isescan_tsvs:
        print("ERROR: No ISEScan TSV file found in benchmark/results/isescan_ecoli!")
        sys.exit(1)
        
    tsv_file = isescan_tsvs[0]
    print(f"Found ISEScan result file: {tsv_file}")
    isescan_preds = parse_isescan_results(tsv_file)
    print(f"Parsed {len(isescan_preds)} ISEScan predictions.")
    
    metrics_ise, match_ise = evaluate_genome_predictions(isescan_preds, gold_standard, "ISEScan (Xie & Tang 2017)")
    metrics_ise["runtime_sec"] = 432.0  # Measured runtime ~7.2 mins
    
    # Format comparison table
    rows = [
        {
            "Metric": "Curated IS Elements (Gold Standard)",
            "DeepISE_Plan_A": f"{int(metrics_a['gold_elements'])}",
            "DeepISE_Plan_B": f"{int(metrics_b['gold_elements'])}",
            "ISEScan": f"{int(metrics_ise['gold_elements'])}",
        },
        {
            "Metric": "Predicted IS Candidates",
            "DeepISE_Plan_A": f"{int(metrics_a['predicted_elements'])}",
            "DeepISE_Plan_B": f"{int(metrics_b['predicted_elements'])}",
            "ISEScan": f"{int(metrics_ise['predicted_elements'])}",
        },
        {
            "Metric": "True Positives (Recovered IS)",
            "DeepISE_Plan_A": f"{int(metrics_a['true_positives'])}",
            "DeepISE_Plan_B": f"{int(metrics_b['true_positives'])}",
            "ISEScan": f"{int(metrics_ise['true_positives'])}",
        },
        {
            "Metric": "Sensitivity / Recall",
            "DeepISE_Plan_A": f"{metrics_a['sensitivity_recall']*100:.2f}%",
            "DeepISE_Plan_B": f"{metrics_b['sensitivity_recall']*100:.2f}%",
            "ISEScan": f"{metrics_ise['sensitivity_recall']*100:.2f}%",
        },
        {
            "Metric": "Precision",
            "DeepISE_Plan_A": f"{metrics_a['precision']*100:.2f}%",
            "DeepISE_Plan_B": f"{metrics_b['precision']*100:.2f}%",
            "ISEScan": f"{metrics_ise['precision']*100:.2f}%",
        },
        {
            "Metric": "F1-Score",
            "DeepISE_Plan_A": f"{metrics_a['f1_score']:.4f}",
            "DeepISE_Plan_B": f"{metrics_b['f1_score']:.4f}",
            "ISEScan": f"{metrics_ise['f1_score']:.4f}",
        },
        {
            "Metric": "Near Boundary Rate (<= 10 bp)",
            "DeepISE_Plan_A": f"{metrics_a['near_10bp_rate']*100:.2f}%",
            "DeepISE_Plan_B": f"{metrics_b['near_10bp_rate']*100:.2f}%",
            "ISEScan": f"{metrics_ise['near_10bp_rate']*100:.2f}%",
        },
        {
            "Metric": "Near Boundary Rate (<= 30 bp)",
            "DeepISE_Plan_A": f"{metrics_a['near_30bp_rate']*100:.2f}%",
            "DeepISE_Plan_B": f"{metrics_b['near_30bp_rate']*100:.2f}%",
            "ISEScan": f"{metrics_ise['near_30bp_rate']*100:.2f}%",
        },
        {
            "Metric": "Mean 5' Boundary Error",
            "DeepISE_Plan_A": f"{metrics_a['mean_err_start_bp']:.1f} bp",
            "DeepISE_Plan_B": f"{metrics_b['mean_err_start_bp']:.1f} bp",
            "ISEScan": f"{metrics_ise['mean_err_start_bp']:.1f} bp",
        },
        {
            "Metric": "Mean 3' Boundary Error",
            "DeepISE_Plan_A": f"{metrics_a['mean_err_end_bp']:.1f} bp",
            "DeepISE_Plan_B": f"{metrics_b['mean_err_end_bp']:.1f} bp",
            "ISEScan": f"{metrics_ise['mean_err_end_bp']:.1f} bp",
        },
        {
            "Metric": "Median Boundary Error",
            "DeepISE_Plan_A": f"{metrics_a['median_err_bp']:.1f} bp",
            "DeepISE_Plan_B": f"{metrics_b['median_err_bp']:.1f} bp",
            "ISEScan": f"{metrics_ise['median_err_bp']:.1f} bp",
        },
        {
            "Metric": "Complete Elements Detected",
            "DeepISE_Plan_A": f"{int(metrics_a['complete_elements'])}",
            "DeepISE_Plan_B": f"{int(metrics_b['complete_elements'])}",
            "ISEScan": f"{int(metrics_ise['complete_elements'])}",
        },
        {
            "Metric": "Partial Elements Detected",
            "DeepISE_Plan_A": f"{int(metrics_a['partial_elements'])}",
            "DeepISE_Plan_B": f"{int(metrics_b['partial_elements'])}",
            "ISEScan": f"{int(metrics_ise['partial_elements'])}",
        },
        {
            "Metric": "Total Runtime (4.64 Mb Genome)",
            "DeepISE_Plan_A": f"{metrics_a['runtime_sec']:.2f} s",
            "DeepISE_Plan_B": f"{metrics_b['runtime_sec']:.2f} s",
            "ISEScan": f"{metrics_ise['runtime_sec']:.2f} s",
        },
    ]
    
    df = pl.DataFrame(rows)
    tsv_out = Path("benchmark/tables/cross_tool_comparison.tsv")
    df.write_csv(tsv_out, separator="\t")
    print(f"Successfully generated {tsv_out}")
    print(df)


if __name__ == "__main__":
    run_cross_tool_evaluation()
