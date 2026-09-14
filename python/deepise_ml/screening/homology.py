from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final


@dataclass(frozen=True, slots=True)
class HomologyAdapterError(RuntimeError):
    tool: str
    detail: str
    command: tuple[str, ...] = ()
    stdout: str | None = None
    stderr: str | None = None

    def __str__(self) -> str:
        command_summary = f"; command: {' '.join(self.command)}" if self.command else ""
        stderr_summary = f"; stderr: {self.stderr.strip()}" if self.stderr else ""
        return f"{self.tool}: {self.detail}{command_summary}{stderr_summary}"


@dataclass(frozen=True, slots=True)
class MMseqsHit:
    query_id: str
    target_id: str
    identity: float
    alignment_length: int
    query_coverage: float
    target_coverage: float
    evalue: float
    bitscore: float


@dataclass(frozen=True, slots=True)
class HmmerHit:
    query_id: str
    profile_id: str
    full_evalue: float
    full_score: float
    best_domain_evalue: float
    best_domain_score: float


@dataclass(frozen=True, slots=True)
class MMseqsRun:
    raw_output: Path
    best_hits: dict[str, MMseqsHit]
    executable: Path


@dataclass(frozen=True, slots=True)
class HmmerRun:
    raw_output: Path
    best_hits: dict[str, HmmerHit]
    executable: Path


HMM_PRESS_SUFFIXES: Final = (".h3f", ".h3i", ".h3m", ".h3p")


def resolve_executable(
    explicit_path: Path | None,
    program_name: str,
    option_name: str,
) -> Path:
    if explicit_path is not None:
        resolved = explicit_path.expanduser().resolve()
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            raise HomologyAdapterError(
                tool=program_name,
                detail=f"{program_name} executable is unavailable at {resolved}; supply an executable {option_name}",
            )
        return resolved
    discovered = shutil.which(program_name)
    if discovered is None:
        raise HomologyAdapterError(
            tool=program_name,
            detail=f"{program_name} executable was not found on PATH; supply {option_name}",
        )
    return Path(discovered).resolve()


def parse_mmseqs_tsv(raw_path: Path) -> dict[str, MMseqsHit]:
    if not raw_path.is_file():
        raise HomologyAdapterError("MMseqs2", f"raw output does not exist: {raw_path}")
    best_hits: dict[str, MMseqsHit] = {}
    for line_number, line in enumerate(raw_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        fields = line.split("\t")
        if len(fields) != 8:
            raise HomologyAdapterError("MMseqs2", f"malformed raw output at line {line_number}: expected 8 fields")
        try:
            hit = MMseqsHit(
                query_id=fields[0],
                target_id=fields[1],
                identity=float(fields[2]),
                alignment_length=int(fields[3]),
                query_coverage=float(fields[4]),
                target_coverage=float(fields[5]),
                evalue=float(fields[6]),
                bitscore=float(fields[7]),
            )
        except ValueError as error:
            raise HomologyAdapterError("MMseqs2", f"invalid numeric value at line {line_number}") from error
        current = best_hits.get(hit.query_id)
        if current is None or _is_better_mmseqs_hit(hit, current):
            best_hits[hit.query_id] = hit
    return best_hits


def parse_hmmer_tblout(raw_path: Path) -> dict[str, HmmerHit]:
    if not raw_path.is_file():
        raise HomologyAdapterError("HMMER", f"raw output does not exist: {raw_path}")
    best_hits: dict[str, HmmerHit] = {}
    for line_number, line in enumerate(raw_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 19:
            raise HomologyAdapterError("HMMER", f"malformed tblout at line {line_number}: expected 19 fields")
        try:
            hit = HmmerHit(
                query_id=fields[0],
                profile_id=fields[2],
                full_evalue=float(fields[4]),
                full_score=float(fields[5]),
                best_domain_evalue=float(fields[7]),
                best_domain_score=float(fields[8]),
            )
        except ValueError as error:
            raise HomologyAdapterError("HMMER", f"invalid numeric value at line {line_number}") from error
        current = best_hits.get(hit.query_id)
        if current is None or _is_better_hmmer_hit(hit, current):
            best_hits[hit.query_id] = hit
    return best_hits


def run_mmseqs_search(
    query_fasta: Path,
    reference_fasta: Path,
    raw_output: Path,
    *,
    executable: Path | None = None,
    work_parent: Path | None = None,
    threads: int = 1,
    sensitivity: float = 7.5,
    timeout_seconds: float | None = None,
) -> MMseqsRun:
    _validate_regular_file(query_fasta, "query FASTA", "MMseqs2")
    _validate_regular_file(reference_fasta, "reference FASTA", "MMseqs2")
    binary = resolve_executable(executable, "mmseqs", "--mmseqs-bin")
    if threads < 1 or sensitivity <= 0.0:
        raise HomologyAdapterError("MMseqs2", "threads must be positive and sensitivity must be greater than zero")
    _validate_fresh_output_destination(raw_output, "MMseqs2")
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    with _owned_work_directory(work_parent, "deepise-mmseqs-") as work_dir:
        command = (
            str(binary),
            "easy-search",
            str(query_fasta),
            str(reference_fasta),
            str(raw_output),
            str(work_dir),
            "-s",
            str(sensitivity),
            "--threads",
            str(threads),
            "--format-output",
            "query,target,fident,alnlen,qcov,tcov,evalue,bits",
        )
        _run_command("MMseqs2", command, timeout_seconds)
        _validate_tool_output(raw_output, "MMseqs2", command)
        hits = parse_mmseqs_tsv(raw_output)
    return MMseqsRun(raw_output=raw_output, best_hits=hits, executable=binary)


def run_hmmsearch(
    query_fasta: Path,
    hmm_database: Path,
    raw_output: Path,
    *,
    executable: Path | None = None,
    work_parent: Path | None = None,
    threads: int = 1,
    evalue: float = 10.0,
    timeout_seconds: float | None = None,
) -> HmmerRun:
    _validate_regular_file(query_fasta, "query FASTA", "HMMER")
    _validate_pressed_hmm_database(hmm_database)
    binary = resolve_executable(executable, "hmmsearch", "--hmmsearch-bin")
    if threads < 1 or evalue <= 0.0:
        raise HomologyAdapterError("HMMER", "threads and E-value must be greater than zero")
    _validate_fresh_output_destination(raw_output, "HMMER")
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    with _owned_work_directory(work_parent, "deepise-hmmer-"):
        command = (
            str(binary),
            "--cpu",
            str(threads),
            "-E",
            str(evalue),
            "--tblout",
            str(raw_output),
            "--noali",
            str(hmm_database),
            str(query_fasta),
        )
        _run_command("HMMER", command, timeout_seconds)
        _validate_tool_output(raw_output, "HMMER", command)
        hits = parse_hmmer_tblout(raw_output)
    return HmmerRun(raw_output=raw_output, best_hits=hits, executable=binary)


def _validate_regular_file(path: Path, label: str, tool: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise HomologyAdapterError(tool, f"{label} is missing or empty: {path}")


def _validate_pressed_hmm_database(hmm_database: Path) -> None:
    _validate_regular_file(hmm_database, "HMM database", "HMMER")
    missing = [
        f"{hmm_database}{suffix}"
        for suffix in HMM_PRESS_SUFFIXES
        if not Path(f"{hmm_database}{suffix}").is_file() or Path(f"{hmm_database}{suffix}").stat().st_size == 0
    ]
    if missing:
        raise HomologyAdapterError("HMMER", f"HMM database is not pressed; missing sidecars: {', '.join(missing)}")


def _owned_work_directory(work_parent: Path | None, prefix: str):
    parent = None
    if work_parent is not None:
        if work_parent.exists() and not work_parent.is_dir():
            raise HomologyAdapterError("workspace", f"temporary work parent is not a directory: {work_parent}")
        work_parent.mkdir(parents=True, exist_ok=True)
        parent = str(work_parent)
    return tempfile.TemporaryDirectory(prefix=prefix, dir=parent)


def _run_command(tool: str, command: tuple[str, ...], timeout_seconds: float | None) -> None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=timeout_seconds)
    except OSError as error:
        raise HomologyAdapterError(tool, f"failed to start executable: {error}", command=command) from error
    except subprocess.TimeoutExpired as error:
        raise HomologyAdapterError(
            tool,
            f"command exceeded timeout of {timeout_seconds} seconds",
            command=command,
            stdout=_coerce_process_text(error.stdout),
            stderr=_coerce_process_text(error.stderr),
        ) from error
    if result.returncode != 0:
        raise HomologyAdapterError(
            tool,
            f"command exited with status {result.returncode}",
            command=command,
            stdout=result.stdout,
            stderr=result.stderr,
        )


def _validate_tool_output(raw_output: Path, tool: str, command: tuple[str, ...]) -> None:
    if not raw_output.is_file():
        raise HomologyAdapterError(tool, f"command succeeded but did not create raw output: {raw_output}", command=command)


def _validate_fresh_output_destination(raw_output: Path, tool: str) -> None:
    if raw_output.exists():
        raise HomologyAdapterError(tool, f"raw output destination already exists: {raw_output}")


def _is_better_mmseqs_hit(candidate: MMseqsHit, current: MMseqsHit) -> bool:
    return (candidate.bitscore, -candidate.evalue) > (current.bitscore, -current.evalue)


def _is_better_hmmer_hit(candidate: HmmerHit, current: HmmerHit) -> bool:
    return (candidate.full_score, -candidate.full_evalue) > (current.full_score, -current.full_evalue)


def _coerce_process_text(value: str | bytes | None) -> str | None:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value
