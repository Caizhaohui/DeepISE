"""Tests for closed-loop bidirectional integration between genome scanning and protein structure research."""

import json
from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from deepise_ml.cli import app
from deepise_ml.screening.closed_loop import (
    DualValidationCategory,
    perform_closed_loop_research_validation,
)


def test_scan_cli_help_includes_research_options() -> None:
    runner = CliRunner(env={"COLUMNS": "200"})
    res = runner.invoke(app, ["scan", "--help"])
    assert res.exit_code == 0
    assert "--research" in res.stdout
    assert "--research-profile" in res.stdout
    assert "--enable-stage2a" in res.stdout
    assert "--enable-stage2b" in res.stdout


def test_scan_metagenome_cli_help_includes_research_options() -> None:
    runner = CliRunner(env={"COLUMNS": "200"})
    res = runner.invoke(app, ["scan-metagenome", "--help"])
    assert res.exit_code == 0
    assert "--research" in res.stdout
    assert "--research-profile" in res.stdout


def test_closed_loop_validation_empty_tpases(tmp_path: Path) -> None:
    elements_tsv = tmp_path / "deepise_is_elements.tsv"
    elements_tsv.write_text("element_id\tcontig_id\tstart\tend\tlength\tfamily\tstatus\tcomposite_score\ttpase_gene_id\ttpase_score\n")
    tpases_faa = tmp_path / "deepise_tpases.faa"
    tpases_faa.write_text("")

    res = perform_closed_loop_research_validation(
        elements_tsv=elements_tsv,
        tpases_faa=tpases_faa,
        output_dir=tmp_path,
    )

    assert res.total_tpases_evaluated == 0
    assert res.novel_discoveries_path.exists()
    df = pl.read_csv(res.novel_discoveries_path, separator="\t")
    assert len(df) == 0


def test_closed_loop_validation_end_to_end(tmp_path: Path) -> None:
    # 1. Create simulated detected elements
    elements_tsv = tmp_path / "deepise_is_elements.tsv"
    elem_data = [
        {
            "element_id": "DeepISE_contig1_0001",
            "contig_id": "contig1",
            "start": 100,
            "end": 1200,
            "length": 1100,
            "family": "IS1",
            "status": "complete",
            "composite_score": 0.95,
            "tpase_gene_id": "contig1_1",
            "tpase_score": 98.5,
        },
        {
            "element_id": "DeepISE_contig1_0002",
            "contig_id": "contig1",
            "start": 3000,
            "end": 4500,
            "length": 1500,
            "family": "IS3",
            "status": "partial",
            "composite_score": 0.45,
            "tpase_gene_id": "contig1_2",
            "tpase_score": 42.0,
        },
    ]
    pl.DataFrame(elem_data).write_csv(elements_tsv, separator="\t")

    # 2. Create simulated tpases FASTA (valid protein sequences >= 50 aa)
    tpases_faa = tmp_path / "deepise_tpases.faa"
    tpases_faa.write_text(
        ">DeepISE_contig1_0001_tpase gene=contig1_1 contig=contig1 family=IS1 score=98.5\n"
        + "M" + "ARNDCQEGHILKMFPSTWYV" * 4 + "\n"
        ">DeepISE_contig1_0002_tpase gene=contig1_2 contig=contig1 family=IS3 score=42.0\n"
        + "M" + "VKWYTSPFWGQEDNRAHILK" * 4 + "\n"
    )

    # 3. Create dummy summary JSON
    summary_json = tmp_path / "deepise_summary.json"
    summary_json.write_text(json.dumps({"runtime_seconds": 1.2, "elements_count": 2}))

    # 4. Use bundled reference and HMM
    ref_faa = Path("benchmark/tmp_hmmer/IS1.faa")
    hmm_db = Path("benchmark/db/deepise_tpases.hmm")

    res = perform_closed_loop_research_validation(
        elements_tsv=elements_tsv,
        tpases_faa=tpases_faa,
        output_dir=tmp_path,
        reference_fasta=ref_faa,
        hmm_database=hmm_db,
        profile="fast",
        summary_json=summary_json,
    )

    assert res.total_tpases_evaluated == 2
    assert res.research_predictions_path.exists()
    assert res.novel_discoveries_path.exists()

    disc_df = pl.read_csv(res.novel_discoveries_path, separator="\t")
    assert len(disc_df) == 2
    assert "dual_category" in disc_df.columns
    assert "dual_confidence" in disc_df.columns

    # Verify summary was updated with research stats
    updated_summary = json.loads(summary_json.read_text())
    assert "research_validation" in updated_summary
    assert updated_summary["research_validation"]["research_enabled"] is True
    assert updated_summary["research_validation"]["total_tpases_evaluated"] == 2
