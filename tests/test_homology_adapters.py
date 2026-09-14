import stat
from pathlib import Path

import pytest
from deepise_ml.screening import homology
from deepise_ml.screening.homology import (
    HomologyAdapterError,
    parse_hmmer_tblout,
    parse_mmseqs_tsv,
    resolve_executable,
    run_hmmsearch,
    run_mmseqs_search,
)


def test_parse_mmseqs_selects_highest_bitscore_and_retains_all_evidence(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "mmseqs.tsv"
    raw_path.write_text(
        "query_a\ttarget_low\t0.31\t90\t0.72\t0.64\t1e-08\t55\n"
        "query_a\ttarget_best\t0.42\t120\t0.85\t0.73\t1e-12\t72\n"
    )

    hits = parse_mmseqs_tsv(raw_path)

    assert hits["query_a"].target_id == "target_best"
    assert hits["query_a"].identity == pytest.approx(0.42)
    assert hits["query_a"].alignment_length == 120
    assert hits["query_a"].query_coverage == pytest.approx(0.85)
    assert hits["query_a"].target_coverage == pytest.approx(0.73)
    assert hits["query_a"].evalue == pytest.approx(1e-12)
    assert hits["query_a"].bitscore == pytest.approx(72.0)


def test_parse_hmmer_real_19_column_tblout_retains_full_and_best_domain_evidence() -> None:
    fixture_path = Path("benchmark/results/hmmer_test_raw.tbl")

    hits = parse_hmmer_tblout(fixture_path)

    hit = hits["tpase_004917"]
    assert hit.profile_id == "Tpase_IS1"
    assert hit.full_evalue == pytest.approx(7.3e-48)
    assert hit.full_score == pytest.approx(161.1)
    assert hit.best_domain_evalue == pytest.approx(9.5e-48)
    assert hit.best_domain_score == pytest.approx(160.8)


def test_resolve_executable_rejects_missing_explicit_binary() -> None:
    with pytest.raises(HomologyAdapterError, match="mmseqs executable"):
        resolve_executable(Path("/missing/mmseqs"), "mmseqs", "--mmseqs-bin")


def test_mmseqs_runner_preserves_caller_work_parent_and_cleans_owned_directory(
    tmp_path: Path,
) -> None:
    query_fasta = tmp_path / "query.faa"
    reference_fasta = tmp_path / "reference.faa"
    query_fasta.write_text(">query_a\nMAAA\n")
    reference_fasta.write_text(">target_a\nMAAA\n")
    work_parent = tmp_path / "caller_workspace"
    work_parent.mkdir()
    sentinel = work_parent / "keep.txt"
    sentinel.write_text("caller-owned")
    executable = _write_executable(
        tmp_path / "fake_mmseqs",
        "from pathlib import Path\n"
        "import sys\n"
        "Path(sys.argv[4]).write_text('query_a\\ttarget_a\\t0.5\\t4\\t1.0\\t1.0\\t1e-9\\t99\\n')\n",
    )

    run = run_mmseqs_search(
        query_fasta,
        reference_fasta,
        tmp_path / "raw" / "mmseqs.tsv",
        executable=executable,
        work_parent=work_parent,
    )

    assert sentinel.read_text() == "caller-owned"
    assert not list(work_parent.glob("deepise-mmseqs-*"))
    assert run.best_hits["query_a"].bitscore == pytest.approx(99.0)
    assert run.raw_output.is_file()


def test_hmmer_runner_requires_pressed_database_and_preserves_best_domain_fields(
    tmp_path: Path,
) -> None:
    query_fasta = tmp_path / "query.faa"
    hmm_database = tmp_path / "profiles.hmm"
    query_fasta.write_text(">query_a\nMAAA\n")
    hmm_database.write_text("HMMER3/f")
    for suffix in (".h3f", ".h3i", ".h3m", ".h3p"):
        Path(f"{hmm_database}{suffix}").write_text("pressed")
    executable = _write_executable(
        tmp_path / "fake_hmmsearch",
        "from pathlib import Path\n"
        "import sys\n"
        "output = Path(sys.argv[sys.argv.index('--tblout') + 1])\n"
        "output.write_text('query_a - Profile_A - 2e-8 45.0 0.0 3e-8 44.0 0.0 1.0 1 0 0 1 1 1 1 -\\n')\n",
    )

    run = run_hmmsearch(
        query_fasta,
        hmm_database,
        tmp_path / "raw" / "hmmer.tbl",
        executable=executable,
        work_parent=tmp_path / "workspace",
    )

    assert run.best_hits["query_a"].profile_id == "Profile_A"
    assert run.best_hits["query_a"].best_domain_evalue == pytest.approx(3e-8)
    assert run.best_hits["query_a"].best_domain_score == pytest.approx(44.0)


def test_hmmer_runner_rejects_empty_pressed_sidecar(tmp_path: Path) -> None:
    query_fasta = tmp_path / "query.faa"
    hmm_database = tmp_path / "profiles.hmm"
    query_fasta.write_text(">query_a\nMAAA\n")
    hmm_database.write_text("HMMER3/f")
    for suffix in (".h3f", ".h3i", ".h3m", ".h3p"):
        Path(f"{hmm_database}{suffix}").write_text("pressed")
    Path(f"{hmm_database}.h3p").write_text("")
    executable = _write_executable(tmp_path / "fake_hmmsearch", "")

    with pytest.raises(HomologyAdapterError, match="not pressed"):
        run_hmmsearch(
            query_fasta,
            hmm_database,
            tmp_path / "raw" / "hmmer.tbl",
            executable=executable,
        )


def test_mmseqs_runner_rejects_malformed_success_output_and_cleans_workspace(
    tmp_path: Path,
) -> None:
    query_fasta = tmp_path / "query.faa"
    reference_fasta = tmp_path / "reference.faa"
    query_fasta.write_text(">query_a\nMAAA\n")
    reference_fasta.write_text(">target_a\nMAAA\n")
    work_parent = tmp_path / "workspace"
    executable = _write_executable(
        tmp_path / "malformed_mmseqs",
        "from pathlib import Path\n"
        "import sys\n"
        "Path(sys.argv[4]).write_text('not\\tenough\\tfields\\n')\n",
    )

    with pytest.raises(HomologyAdapterError, match="expected 8 fields"):
        run_mmseqs_search(
            query_fasta,
            reference_fasta,
            tmp_path / "raw" / "mmseqs.tsv",
            executable=executable,
            work_parent=work_parent,
        )

    assert not list(work_parent.glob("deepise-mmseqs-*"))


def test_mmseqs_runner_rejects_file_as_work_parent_with_actionable_error(tmp_path: Path) -> None:
    query_fasta = tmp_path / "query.faa"
    reference_fasta = tmp_path / "reference.faa"
    work_parent = tmp_path / "not_a_directory"
    query_fasta.write_text(">query_a\nMAAA\n")
    reference_fasta.write_text(">target_a\nMAAA\n")
    work_parent.write_text("not a directory")
    executable = _write_executable(tmp_path / "fake_mmseqs", "")

    with pytest.raises(HomologyAdapterError, match="temporary work parent"):
        run_mmseqs_search(
            query_fasta,
            reference_fasta,
            tmp_path / "raw" / "mmseqs.tsv",
            executable=executable,
            work_parent=work_parent,
        )


def test_mmseqs_runner_rejects_stale_raw_output_before_running(tmp_path: Path) -> None:
    query_fasta = tmp_path / "query.faa"
    reference_fasta = tmp_path / "reference.faa"
    raw_output = tmp_path / "raw" / "mmseqs.tsv"
    query_fasta.write_text(">query_a\nMAAA\n")
    reference_fasta.write_text(">target_a\nMAAA\n")
    raw_output.parent.mkdir()
    raw_output.write_text("stale\toutput\n")
    executable = _write_executable(tmp_path / "fake_mmseqs", "")

    with pytest.raises(HomologyAdapterError, match="already exists"):
        run_mmseqs_search(
            query_fasta,
            reference_fasta,
            raw_output,
            executable=executable,
        )

    assert raw_output.read_text() == "stale\toutput\n"


def test_mmseqs_runner_cleans_owned_workspace_after_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query_fasta = tmp_path / "query.faa"
    reference_fasta = tmp_path / "reference.faa"
    work_parent = tmp_path / "workspace"
    query_fasta.write_text(">query_a\nMAAA\n")
    reference_fasta.write_text(">target_a\nMAAA\n")
    executable = _write_executable(tmp_path / "fake_mmseqs", "")

    def interrupt(_tool: str, _command: tuple[str, ...], _timeout: float | None) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(homology, "_run_command", interrupt)

    with pytest.raises(KeyboardInterrupt):
        run_mmseqs_search(
            query_fasta,
            reference_fasta,
            tmp_path / "raw" / "mmseqs.tsv",
            executable=executable,
            work_parent=work_parent,
        )

    assert not list(work_parent.glob("deepise-mmseqs-*"))


def test_mmseqs_runner_times_out_and_cleans_owned_workspace(tmp_path: Path) -> None:
    query_fasta = tmp_path / "query.faa"
    reference_fasta = tmp_path / "reference.faa"
    work_parent = tmp_path / "workspace"
    query_fasta.write_text(">query_a\nMAAA\n")
    reference_fasta.write_text(">target_a\nMAAA\n")
    executable = _write_executable(
        tmp_path / "slow_mmseqs",
        "import time\n"
        "time.sleep(1)\n",
    )

    with pytest.raises(HomologyAdapterError, match="exceeded timeout"):
        run_mmseqs_search(
            query_fasta,
            reference_fasta,
            tmp_path / "raw" / "mmseqs.tsv",
            executable=executable,
            work_parent=work_parent,
            timeout_seconds=0.01,
        )

    assert not list(work_parent.glob("deepise-mmseqs-*"))


def _write_executable(path: Path, program: str) -> Path:
    path.write_text(f"#!/usr/bin/env python3\n{program}")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path
