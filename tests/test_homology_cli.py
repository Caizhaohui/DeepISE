from pathlib import Path

from deepise_ml.cli import app
from typer.testing import CliRunner


def test_screen_proteins_homology_help_exposes_explicit_resource_options() -> None:
    result = CliRunner().invoke(app, ["screen-proteins-homology", "--help"])

    assert result.exit_code == 0
    assert "--reference-fasta" in result.stdout
    assert "--hmm-database" in result.stdout
    assert "--mmseqs-bin" in result.stdout
    assert "--hmmsearch-bin" in result.stdout
    assert "--profile" in result.stdout
    assert "--output" in result.stdout


def test_scan_help_regression_preserves_boundary_modes() -> None:
    result = CliRunner().invoke(app, ["scan", "--help"])

    assert result.exit_code == 0
    assert "hybrid" in result.stdout
    assert "plan_a" in result.stdout
    assert "plan_b" in result.stdout


def test_screen_proteins_homology_missing_reference_fasta_fails_with_actionable_error(
    tmp_path: Path,
) -> None:
    fasta_path = tmp_path / "query.faa"
    fasta_path.write_text(">protein_a\n" + "M" + "A" * 60 + "\n")

    result = CliRunner().invoke(
        app,
        [
            "screen-proteins-homology",
            "--fasta",
            str(fasta_path),
            "--reference-fasta",
            str(tmp_path / "missing_ref.faa"),
            "--hmm-database",
            str(tmp_path / "missing.hmm"),
            "--output",
            str(tmp_path / "predictions.tsv"),
        ],
    )

    assert result.exit_code != 0
    assert "reference FASTA is missing or empty" in result.stdout


def test_screen_proteins_homology_missing_binary_fails_with_actionable_error(
    tmp_path: Path,
) -> None:
    query_fasta = tmp_path / "query.faa"
    query_fasta.write_text(">protein_a\n" + "M" + "A" * 60 + "\n")
    ref_fasta = tmp_path / "ref.faa"
    ref_fasta.write_text(">target\n" + "M" + "A" * 60 + "\n")
    hmm_db = tmp_path / "test.hmm"
    hmm_db.write_text("HMMER3/f\n")
    for sfx in (".h3f", ".h3i", ".h3m", ".h3p"):
        (tmp_path / f"test.hmm{sfx}").write_text("pressed\n")

    result = CliRunner().invoke(
        app,
        [
            "screen-proteins-homology",
            "--fasta",
            str(query_fasta),
            "--reference-fasta",
            str(ref_fasta),
            "--hmm-database",
            str(hmm_db),
            "--mmseqs-bin",
            "/missing/mmseqs",
            "--output",
            str(tmp_path / "predictions.tsv"),
        ],
    )

    assert result.exit_code != 0
    assert "mmseqs executable is unavailable" in result.stdout


def test_screen_proteins_research_help_and_options() -> None:
    result = CliRunner(env={"COLUMNS": "200"}).invoke(app, ["screen-proteins-research", "--help"])

    assert result.exit_code == 0
    assert "--reference-fasta" in result.stdout
    assert "--hmm-database" in result.stdout
    assert "--enable-stage2a" in result.stdout
    assert "--enable-stage2b" in result.stdout
    assert "--output" in result.stdout


def test_screen_proteins_research_missing_reference_fasta_fails(
    tmp_path: Path,
) -> None:
    fasta_path = tmp_path / "query.faa"
    fasta_path.write_text(">protein_a\n" + "M" + "A" * 60 + "\n")

    result = CliRunner().invoke(
        app,
        [
            "screen-proteins-research",
            "--fasta",
            str(fasta_path),
            "--reference-fasta",
            str(tmp_path / "missing_ref.faa"),
            "--hmm-database",
            str(tmp_path / "missing.hmm"),
            "--output",
            str(tmp_path / "predictions.tsv"),
        ],
    )

    assert result.exit_code != 0
    assert "reference FASTA is missing or empty" in result.stdout

