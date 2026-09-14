from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import polars as pl
from deepise_ml.screening.evidence import (
    COMBINED_OUTPUT_COLUMNS,
    CandidateRoute,
    HMMEREvidence,
    HomologyEvidence,
    HomologyEvidenceSet,
    MMseqsEvidence,
    ScreeningWithHomologyRecord,
    parse_hmmer_evidence,
    parse_mmseqs_evidence,
)
from deepise_ml.screening.homology import (
    HmmerRun,
    MMseqsRun,
    _validate_fresh_output_destination,
    _validate_pressed_hmm_database,
    _validate_regular_file,
    run_hmmsearch,
    run_mmseqs_search,
)
from deepise_ml.screening.routing import CandidateRouter
from deepise_ml.screening.runtime import (
    EmbeddingClassifier,
    EmbeddingEngine,
    ProfileDefinition,
    ProteinInputRecord,
    ProteinScreeningError,
    ProteinScreeningService,
)

DEEPISE_VERSION: Final = "0.1.0"


def compute_file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_normalized_fasta(proteins: Sequence[ProteinInputRecord], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in proteins:
            handle.write(f">{record.protein_id}\n{record.sequence}\n")


@dataclass(frozen=True, slots=True)
class HomologyScreeningPaths:
    output_tsv: Path
    reference_fasta: Path
    hmm_database: Path
    mmseqs_raw_output: Path | None = None
    hmmer_raw_output: Path | None = None
    metadata_path: Path | None = None
    work_parent: Path | None = None

    def resolved_mmseqs_raw_output(self) -> Path:
        if self.mmseqs_raw_output is not None:
            return self.mmseqs_raw_output
        if self.output_tsv.name == "predictions.tsv":
            return self.output_tsv.parent / "intermediate" / "mmseqs.tsv"
        return self.output_tsv.parent / "intermediate" / f"{self.output_tsv.stem}.mmseqs.tsv"

    def resolved_hmmer_raw_output(self) -> Path:
        if self.hmmer_raw_output is not None:
            return self.hmmer_raw_output
        if self.output_tsv.name == "predictions.tsv":
            return self.output_tsv.parent / "intermediate" / "hmmer.tbl"
        return self.output_tsv.parent / "intermediate" / f"{self.output_tsv.stem}.hmmer.tbl"

    def resolved_metadata_path(self) -> Path:
        if self.metadata_path is not None:
            return self.metadata_path
        if self.output_tsv.name == "predictions.tsv":
            return self.output_tsv.parent / "metadata.json"
        return self.output_tsv.parent / f"{self.output_tsv.stem}.metadata.json"

    def validate_inputs_and_destinations(self) -> None:
        _validate_regular_file(self.reference_fasta, "reference FASTA", "pipeline")
        _validate_pressed_hmm_database(self.hmm_database)
        _validate_fresh_output_destination(self.output_tsv, "pipeline")
        _validate_fresh_output_destination(self.resolved_metadata_path(), "pipeline")
        _validate_fresh_output_destination(self.resolved_mmseqs_raw_output(), "MMseqs2")
        _validate_fresh_output_destination(self.resolved_hmmer_raw_output(), "HMMER")


class ProteinHomologyScreeningService:
    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        classifier: EmbeddingClassifier,
        profile: ProfileDefinition,
        *,
        router: CandidateRouter | None = None,
        mmseqs_binary: Path | None = None,
        hmmer_binary: Path | None = None,
        threads: int = 1,
        mmseqs_runner: Callable[..., MMseqsRun] = run_mmseqs_search,
        hmmer_runner: Callable[..., HmmerRun] = run_hmmsearch,
    ) -> None:
        self._embedding_engine = embedding_engine
        self._classifier = classifier
        self._profile = profile
        self._stage1_service = ProteinScreeningService(
            embedding_engine=embedding_engine,
            classifier=classifier,
            profile=profile,
        )
        self._router = router or CandidateRouter()
        self._mmseqs_binary = mmseqs_binary
        self._hmmer_binary = hmmer_binary
        self._threads = threads
        self._mmseqs_runner = mmseqs_runner
        self._hmmer_runner = hmmer_runner

    def screen_and_route(
        self,
        proteins: list[ProteinInputRecord],
        paths: HomologyScreeningPaths,
        batch_size: int = 32,
    ) -> list[ScreeningWithHomologyRecord]:
        if not proteins:
            raise ProteinScreeningError("no protein records provided for screening")
        paths.validate_inputs_and_destinations()

        stage1_records = self._stage1_service.predict_stage1(proteins, batch_size=batch_size)

        mmseqs_out = paths.resolved_mmseqs_raw_output()
        hmmer_out = paths.resolved_hmmer_raw_output()

        try:
            with tempfile.TemporaryDirectory(
                prefix="deepise-homology-query-", dir=paths.work_parent
            ) as temp_dir:
                temp_query_fasta = Path(temp_dir) / "query_normalized.faa"
                write_normalized_fasta(proteins, temp_query_fasta)

                mmseqs_run = self._mmseqs_runner(
                    query_fasta=temp_query_fasta,
                    reference_fasta=paths.reference_fasta,
                    raw_output=mmseqs_out,
                    executable=self._mmseqs_binary,
                    work_parent=paths.work_parent,
                    threads=self._threads,
                )

                hmmer_run = self._hmmer_runner(
                    query_fasta=temp_query_fasta,
                    hmm_database=paths.hmm_database,
                    raw_output=hmmer_out,
                    executable=self._hmmer_binary,
                    work_parent=paths.work_parent,
                    threads=self._threads,
                )

            combined_records: list[ScreeningWithHomologyRecord] = []
            for stage1_rec in stage1_records:
                pid = stage1_rec.protein_id
                mm_hit = mmseqs_run.best_hits.get(pid)
                if mm_hit is not None:
                    mm_ev = parse_mmseqs_evidence(
                        hit=True,
                        target_id=mm_hit.target_id,
                        identity_percent=mm_hit.identity,
                        alignment_length=mm_hit.alignment_length,
                        query_coverage_percent=mm_hit.query_coverage,
                        target_coverage_percent=mm_hit.target_coverage,
                        evalue=mm_hit.evalue,
                        bitscore=mm_hit.bitscore,
                    )
                else:
                    mm_ev = MMseqsEvidence.no_hit()

                hm_hit = hmmer_run.best_hits.get(pid)
                if hm_hit is not None:
                    hm_ev = parse_hmmer_evidence(
                        hit=True,
                        profile_id=hm_hit.profile_id,
                        full_evalue=hm_hit.full_evalue,
                        full_bitscore=hm_hit.full_score,
                        domain_evalue=hm_hit.best_domain_evalue,
                        domain_bitscore=hm_hit.best_domain_score,
                    )
                else:
                    hm_ev = HMMEREvidence.no_hit()

                evidence = HomologyEvidence(
                    protein_id=pid,
                    mmseqs=mm_ev,
                    hmmer=hm_ev,
                )
                decision = self._router.route(stage1_rec, evidence)
                combined_records.append(
                    ScreeningWithHomologyRecord.from_stage1(
                        stage1=stage1_rec,
                        evidence=evidence,
                        decision=decision,
                    )
                )

            HomologyEvidenceSet(tuple(record.evidence for record in combined_records))

            paths.output_tsv.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame([record.tsv_row() for record in combined_records], infer_schema_length=None).select(
                list(COMBINED_OUTPUT_COLUMNS)
            ).write_csv(paths.output_tsv, separator="\t")

            metadata_path = paths.resolved_metadata_path()
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            meta = {
                "software_version": DEEPISE_VERSION,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "profile": self._profile.name,
                "model_key": self._profile.model_key,
                "classifier_id": self._classifier.classifier_id,
                "input_protein_count": len(proteins),
                "routes": {
                    route.value: sum(1 for r in combined_records if r.decision.route == route)
                    for route in CandidateRoute
                },
                "resources": {
                    "reference_fasta": {
                        "path": str(paths.reference_fasta.resolve()),
                        "sha256": compute_file_sha256(paths.reference_fasta),
                    },
                    "hmm_database": {
                        "path": str(paths.hmm_database.resolve()),
                        "sha256": compute_file_sha256(paths.hmm_database),
                    },
                    "mmseqs_binary": str(mmseqs_run.executable),
                    "hmmer_binary": str(hmmer_run.executable),
                },
                "artifacts": {
                    "output_tsv": str(paths.output_tsv.resolve()),
                    "mmseqs_raw": str(mmseqs_out.resolve()),
                    "hmmer_raw": str(hmmer_out.resolve()),
                },
            }
            metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

            return combined_records

        except Exception:
            if paths.output_tsv.is_file():
                paths.output_tsv.unlink(missing_ok=True)
            if paths.resolved_metadata_path().is_file():
                paths.resolved_metadata_path().unlink(missing_ok=True)
            if mmseqs_out.is_file():
                mmseqs_out.unlink(missing_ok=True)
            if hmmer_out.is_file():
                hmmer_out.unlink(missing_ok=True)
            raise


__all__ = [
    "DEEPISE_VERSION",
    "HomologyScreeningPaths",
    "ProteinHomologyScreeningService",
    "compute_file_sha256",
    "write_normalized_fasta",
]
