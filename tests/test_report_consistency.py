"""Test Report Consistency (DeepISE Scientific Audit v1.1 - Section 54).

Verifies that:
1. benchmark/results_summary.json is the valid Single Source of Truth.
2. README.md numbers match results_summary.json exactly.
3. TSV benchmark tables match results_summary.json.
4. Pre-registered decision gate verdict matches pre-registered logic.
"""

import json
import re
from pathlib import Path

import polars as pl
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_results_summary_exists_and_valid():
    summary_path = ROOT_DIR / "benchmark/results_summary.json"
    assert summary_path.exists(), "results_summary.json does not exist"
    
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    assert "version" in data
    assert "provenance" in data
    assert "dataset_summary" in data
    assert "lofo_v2" in data
    assert "decision_gate" in data


def test_lofo_numbers_match_tsv_and_readme():
    summary_path = ROOT_DIR / "benchmark/results_summary.json"
    tsv_path = ROOT_DIR / "benchmark/tables/lofo_v2.tsv"
    readme_path = ROOT_DIR / "README.md"
    
    assert summary_path.exists() and tsv_path.exists() and readme_path.exists()
    
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
        
    tsv_df = pl.read_csv(tsv_path, separator="\t")
    macro_tsv = tsv_df.filter(pl.col("family") == "MACRO_AVERAGE").to_dicts()[0]
    
    macro_json = summary["lofo_v2"]["macro_average"]
    
    # Check JSON matches TSV
    assert abs(macro_json["esm2_recall"] - macro_tsv["esm2_recall"]) < 1e-4
    assert abs(macro_json["hmm_recall"] - macro_tsv["hmm_recall"]) < 1e-4
    assert abs(macro_json["delta_recall_pp"] - macro_tsv["delta_recall_pp"]) < 1e-4
    
    # Check README matches JSON
    readme_text = readme_path.read_text(encoding="utf-8")
    
    esm2_str = f"{macro_json['esm2_recall']:.2f}%"
    hmm_str = f"{macro_json['hmm_recall']:.2f}%"
    delta_str = f"{macro_json['delta_recall_pp']:+.2f} pp"
    
    assert esm2_str in readme_text, f"README does not contain expected ESM2 recall {esm2_str}"
    assert hmm_str in readme_text, f"README does not contain expected HMM recall {hmm_str}"
    assert delta_str in readme_text, f"README does not contain expected Delta recall {delta_str}"


def test_phase1_numbers_match_tsv_and_readme():
    summary_path = ROOT_DIR / "benchmark/results_summary.json"
    p1_tsv = ROOT_DIR / "benchmark/tables/phase1_benchmark_v2.tsv"
    readme_path = ROOT_DIR / "README.md"
    
    if not p1_tsv.exists() or not summary_path.exists():
        pytest.skip("Phase-1 benchmark not completed yet")
        
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
        
    p1_json = summary.get("phase1_benchmark", {})
    if not p1_json:
        pytest.skip("Phase-1 benchmark empty in results_summary.json")
        
    p1_df = pl.read_csv(p1_tsv, separator="\t")
    for row in p1_df.iter_rows(named=True):
        m_name = row["model"]
        json_row = p1_json[m_name]
        assert abs(json_row["Overall_auprc"] - row["Overall_auprc"]) < 1e-4
        assert abs(json_row["Remote20_recall"] - row["Remote20_recall"]) < 1e-4
        assert abs(json_row["hard_neg_fpr"] - row["hard_neg_fpr"]) < 1e-4
        
    # Check ESM2-LR (35M) Remote-20 recall in README
    readme_text = readme_path.read_text(encoding="utf-8")
    esm2_remote20_str = f"{p1_json['ESM2-LR (35M)']['Remote20_recall']:.2f}%"
    assert esm2_remote20_str in readme_text, f"README missing ESM2-LR Remote-20 recall {esm2_remote20_str}"


def test_decision_gate_logic():
    summary_path = ROOT_DIR / "benchmark/results_summary.json"
    if not summary_path.exists():
        pytest.skip("results_summary.json does not exist")
        
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
        
    gate = summary.get("decision_gate", {})
    verdict = gate.get("verdict")
    assert verdict in ["GO", "CONDITIONAL_GO", "NO_GO"], f"Unexpected verdict {verdict}"
    
    # Verify logic adherence
    metrics = gate.get("metrics", {})
    r20_gain = metrics.get("remote20_gain_pp", 0.0)
    ch_fpr = metrics.get("hard_neg_fpr_percent", 100.0)
    lofo_ci_lower = metrics.get("lofo_gain_ci_lower_pp", -100.0)
    no_full_gain = metrics.get("no_full_length_gain_pp", 0.0)
    
    if r20_gain >= 10.0 and ch_fpr <= 5.0 and lofo_ci_lower > 0.0:
        assert verdict == "GO"
    elif r20_gain >= 3.0 or no_full_gain >= 10.0 or lofo_ci_lower > 0.0:
        assert verdict == "CONDITIONAL_GO"
    else:
        assert verdict == "NO_GO"
