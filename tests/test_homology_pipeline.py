import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from deepise_ml.screening.evidence import (
    COMBINED_OUTPUT_COLUMNS,
    CandidateRoute,
    RouteReasonCode,
)
from deepise_ml.screening.homology import (
    HmmerHit,
    HmmerRun,
    HomologyAdapterError,
    MMseqsHit,
    MMseqsRun,
)
from deepise_ml.screening.pipeline import (
    HomologyScreeningPaths,
    ProteinHomologyScreeningService,
)
from deepise_ml.screening.runtime import (
    LinearEmbeddingClassifier,
    ProfileDefinition,
    read_protein_fasta,
)


class StubEmbeddingEngine:
    def extract_sequence_embeddings(
        self,
        sequences: list[str],
        batch_size: int,
        verbose: bool,
    ) -> np.ndarray:
        del batch_size, verbose
        return np.asarray([[float(len(sequence)), 1.0] for sequence in sequences])


def write_linear_artifact(path: Path) -> None:
    np.savez(
        path,
        mean=np.asarray([0.0, 0.0]),
        scale=np.asarray([1.0, 1.0]),
        coefficients=np.asarray([0.2, 0.0]),
        intercept=np.asarray(-2.0),
        threshold=np.asarray(0.5),
    )


def test_homology_pipeline_writes_combined_tsv_metadata_and_sidecars(tmp_path: Path) -> None:
    # protein_a: len 15 -> score ~0.73 (>= 0.70 high score)
    # protein_b: len 4 -> score ~0.23 (< 0.35 low score)
    query_fasta = tmp_path / "query.faa"
    query_fasta.write_text(">protein_a\nMAAAAAAAAAAAAAA\n>protein_b\nMCCC\n")

    ref_fasta = tmp_path / "reference.faa"
    ref_fasta.write_text(">target_1\nMAAAAAAAAAAAAAA\n")

    hmm_db = tmp_path / "test.hmm"
    hmm_db.write_text("HMMER3/f\n")
    for sfx in (".h3f", ".h3i", ".h3m", ".h3p"):
        (tmp_path / f"test.hmm{sfx}").write_text("pressed\n")

    artifact_path = tmp_path / "linear_classifier.npz"
    write_linear_artifact(artifact_path)
    metadata_json = tmp_path / "linear_classifier.json"
    metadata_json.write_text(
        json.dumps({"model_key": "stub_2d", "embedding_dimension": 2})
    )

    profile = ProfileDefinition(
        name="fast",
        model_key="stub_2d",
        classifier_path=artifact_path,
        metadata_path=metadata_json,
        classifier_kind="linear_npz",
        embedding_dimension=2,
    )

    out_tsv = tmp_path / "results" / "predictions.tsv"
    paths = HomologyScreeningPaths(
        output_tsv=out_tsv,
        reference_fasta=ref_fasta,
        hmm_database=hmm_db,
    )

    def stub_mmseqs(
        query_fasta: Path,
        reference_fasta: Path,
        raw_output: Path,
        **kwargs,
    ) -> MMseqsRun:
        del query_fasta, reference_fasta, kwargs
        raw_output.parent.mkdir(parents=True, exist_ok=True)
        raw_output.write_text(
            "protein_a\ttarget_1\t0.95\t15\t1.0\t1.0\t1e-10\t120.0\n"
        )
        return MMseqsRun(
            raw_output=raw_output,
            best_hits={
                "protein_a": MMseqsHit(
                    query_id="protein_a",
                    target_id="target_1",
                    identity=95.0,
                    alignment_length=15,
                    query_coverage=100.0,
                    target_coverage=100.0,
                    evalue=1e-10,
                    bitscore=120.0,
                )
            },
            executable=Path("/stub/mmseqs"),
        )

    def stub_hmmer(
        query_fasta: Path,
        hmm_database: Path,
        raw_output: Path,
        **kwargs,
    ) -> HmmerRun:
        del query_fasta, hmm_database, kwargs
        raw_output.parent.mkdir(parents=True, exist_ok=True)
        raw_output.write_text(
            "# tblout mock\nprotein_a - IS1 - 1e-12 110.0 0.0 1e-12 105.0 0.0 1 1 0 0 1 1 1 1 -\n"
        )
        return HmmerRun(
            raw_output=raw_output,
            best_hits={
                "protein_a": HmmerHit(
                    query_id="protein_a",
                    profile_id="IS1",
                    full_evalue=1e-12,
                    full_score=110.0,
                    best_domain_evalue=1e-12,
                    best_domain_score=105.0,
                )
            },
            executable=Path("/stub/hmmsearch"),
        )

    service = ProteinHomologyScreeningService(
        embedding_engine=StubEmbeddingEngine(),
        classifier=LinearEmbeddingClassifier(artifact_path),
        profile=profile,
        mmseqs_runner=stub_mmseqs,
        hmmer_runner=stub_hmmer,
    )

    proteins = read_protein_fasta(query_fasta, minimum_length=4, maximum_length=20)
    records = service.screen_and_route(proteins, paths, batch_size=2)

    assert len(records) == 2
    assert out_tsv.is_file()

    # Verify combined TSV schema & data
    df = pl.read_csv(out_tsv, separator="\t")
    assert df.columns == list(COMBINED_OUTPUT_COLUMNS)
    assert df["protein_id"].to_list() == ["protein_a", "protein_b"]

    # protein_a: high score + strong homology -> ACCEPT_KNOWN
    rec_a = df.filter(pl.col("protein_id") == "protein_a").to_dicts()[0]
    assert rec_a["route"] == CandidateRoute.ACCEPT_KNOWN.value
    assert (
        rec_a["reason_code"]
        == RouteReasonCode.ACCEPT_KNOWN_HIGH_SCORE_STRONG_HOMOLOGY.value
    )
    assert rec_a["mmseqs_hit"] is True
    assert rec_a["mmseqs_target_id"] == "target_1"
    assert rec_a["hmmer_hit"] is True
    assert rec_a["hmmer_profile_id"] == "IS1"

    # protein_b: low score + no hit -> REJECT
    rec_b = df.filter(pl.col("protein_id") == "protein_b").to_dicts()[0]
    assert rec_b["route"] == CandidateRoute.REJECT.value
    assert (
        rec_b["reason_code"]
        == RouteReasonCode.REJECT_LOW_SCORE_NO_STRONG_HOMOLOGY.value
    )
    assert rec_b["mmseqs_hit"] is False
    assert rec_b["mmseqs_target_id"] is None
    assert rec_b["hmmer_hit"] is False
    assert rec_b["hmmer_profile_id"] is None

    # Sidecars
    assert paths.resolved_mmseqs_raw_output().is_file()
    assert paths.resolved_hmmer_raw_output().is_file()

    # Metadata
    metadata_path = paths.resolved_metadata_path()
    assert metadata_path.is_file()
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert meta["input_protein_count"] == 2
    assert meta["routes"][CandidateRoute.ACCEPT_KNOWN.value] == 1
    assert meta["routes"][CandidateRoute.REJECT.value] == 1
    assert meta["resources"]["reference_fasta"]["sha256"] == hashlib.sha256(
        ref_fasta.read_bytes()
    ).hexdigest()
    assert meta["resources"]["hmm_database"]["sha256"] == hashlib.sha256(
        hmm_db.read_bytes()
    ).hexdigest()


def test_homology_pipeline_rejects_existing_output_destination(
    tmp_path: Path,
) -> None:
    query_fasta = tmp_path / "query.faa"
    query_fasta.write_text(">protein_a\nMAAAAAAAAAAAAAA\n")
    ref_fasta = tmp_path / "ref.faa"
    ref_fasta.write_text(">target\nMAA\n")
    hmm_db = tmp_path / "test.hmm"
    hmm_db.write_text("HMMER3/f\n")
    for sfx in (".h3f", ".h3i", ".h3m", ".h3p"):
        (tmp_path / f"test.hmm{sfx}").write_text("pressed\n")

    artifact_path = tmp_path / "linear_classifier.npz"
    write_linear_artifact(artifact_path)
    metadata_json = tmp_path / "linear_classifier.json"
    metadata_json.write_text(
        json.dumps({"model_key": "stub_2d", "embedding_dimension": 2})
    )

    profile = ProfileDefinition(
        name="fast",
        model_key="stub_2d",
        classifier_path=artifact_path,
        metadata_path=metadata_json,
        classifier_kind="linear_npz",
        embedding_dimension=2,
    )

    out_tsv = tmp_path / "out" / "predictions.tsv"
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    out_tsv.write_text("pre-existing")

    paths = HomologyScreeningPaths(
        output_tsv=out_tsv,
        reference_fasta=ref_fasta,
        hmm_database=hmm_db,
    )

    service = ProteinHomologyScreeningService(
        embedding_engine=StubEmbeddingEngine(),
        classifier=LinearEmbeddingClassifier(artifact_path),
        profile=profile,
    )

    proteins = read_protein_fasta(query_fasta, minimum_length=4, maximum_length=20)
    with pytest.raises(HomologyAdapterError, match="destination already exists"):
        service.screen_and_route(proteins, paths)

    assert out_tsv.read_text() == "pre-existing"


def test_homology_pipeline_cleans_up_on_failure(tmp_path: Path) -> None:
    query_fasta = tmp_path / "query.faa"
    query_fasta.write_text(">protein_a\nMAAAAAAAAAAAAAA\n")
    ref_fasta = tmp_path / "ref.faa"
    ref_fasta.write_text(">target\nMAA\n")
    hmm_db = tmp_path / "test.hmm"
    hmm_db.write_text("HMMER3/f\n")
    for sfx in (".h3f", ".h3i", ".h3m", ".h3p"):
        (tmp_path / f"test.hmm{sfx}").write_text("pressed\n")

    artifact_path = tmp_path / "linear_classifier.npz"
    write_linear_artifact(artifact_path)
    metadata_json = tmp_path / "linear_classifier.json"
    metadata_json.write_text(
        json.dumps({"model_key": "stub_2d", "embedding_dimension": 2})
    )

    profile = ProfileDefinition(
        name="fast",
        model_key="stub_2d",
        classifier_path=artifact_path,
        metadata_path=metadata_json,
        classifier_kind="linear_npz",
        embedding_dimension=2,
    )

    out_tsv = tmp_path / "out" / "predictions.tsv"
    paths = HomologyScreeningPaths(
        output_tsv=out_tsv,
        reference_fasta=ref_fasta,
        hmm_database=hmm_db,
    )

    def failing_hmmer(*args, **kwargs):
        del args, kwargs
        # Simulate an adapter failure during HMMER execution
        raise HomologyAdapterError("HMMER", "simulated search failure")

    def stub_mmseqs(query_fasta, reference_fasta, raw_output, **kwargs):
        del query_fasta, reference_fasta, kwargs
        raw_output.parent.mkdir(parents=True, exist_ok=True)
        raw_output.write_text("partial mmseqs output\n")
        return MMseqsRun(
            raw_output=raw_output,
            best_hits={},
            executable=Path("/stub/mmseqs"),
        )

    service = ProteinHomologyScreeningService(
        embedding_engine=StubEmbeddingEngine(),
        classifier=LinearEmbeddingClassifier(artifact_path),
        profile=profile,
        mmseqs_runner=stub_mmseqs,
        hmmer_runner=failing_hmmer,
    )

    proteins = read_protein_fasta(query_fasta, minimum_length=4, maximum_length=20)
    with pytest.raises(HomologyAdapterError, match="simulated search failure"):
        service.screen_and_route(proteins, paths)

    # Ensure no output TSV, metadata, or intermediate raw sidecars are left behind
    assert not out_tsv.exists()
    assert not paths.resolved_metadata_path().exists()
    assert not paths.resolved_mmseqs_raw_output().exists()
    assert not paths.resolved_hmmer_raw_output().exists()

