from pathlib import Path

import numpy as np
import polars as pl
import pytest
from deepise_ml.cli import app
from deepise_ml.screening.runtime import (
    LinearEmbeddingClassifier,
    ProfileDefinition,
    ProteinScreeningError,
    ProteinScreeningService,
    load_classifier,
    read_protein_fasta,
    resolve_profile,
)
from typer.testing import CliRunner


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


def test_read_protein_fasta_normalizes_and_rejects_duplicate_ids(tmp_path: Path) -> None:
    valid_path = tmp_path / "proteins.faa"
    valid_path.write_text(">protein_a first description\nmaaa\n>protein_b\nMCCC\n")

    records = read_protein_fasta(valid_path, minimum_length=4, maximum_length=20)

    assert [record.protein_id for record in records] == ["protein_a", "protein_b"]
    assert [record.sequence for record in records] == ["MAAA", "MCCC"]

    duplicate_path = tmp_path / "duplicate.faa"
    duplicate_path.write_text(">protein_a\nMAAA\n>protein_a\nMCCC\n")

    with pytest.raises(ProteinScreeningError, match="duplicate protein ID"):
        read_protein_fasta(duplicate_path, minimum_length=4, maximum_length=20)


def test_read_protein_fasta_normalizes_stop_symbols_without_losing_length(tmp_path: Path) -> None:
    fasta_path = tmp_path / "stops.faa"
    fasta_path.write_text(">internal_stop\nMA*AA\n>terminal_stop\nMCCCC*\n")

    records = read_protein_fasta(fasta_path, minimum_length=4, maximum_length=20)

    assert [record.sequence for record in records] == ["MAXAA", "MCCCC"]
    assert [record.qc_status for record in records] == ["stop_symbol_normalized", "stop_symbol_normalized"]


def test_screening_service_writes_traceable_stage1_predictions(tmp_path: Path) -> None:
    fasta_path = tmp_path / "proteins.faa"
    fasta_path.write_text(">protein_a\nMAAAAAAAAAAA\n>protein_b\nMCCC\n")
    artifact_path = tmp_path / "linear_classifier.npz"
    write_linear_artifact(artifact_path)

    profile = ProfileDefinition(
        name="fast",
        model_key="stub_2d",
        classifier_path=artifact_path,
        metadata_path=artifact_path.with_suffix(".json"),
        classifier_kind="linear_npz",
        embedding_dimension=2,
    )
    service = ProteinScreeningService(
        embedding_engine=StubEmbeddingEngine(),
        classifier=LinearEmbeddingClassifier(artifact_path),
        profile=profile,
    )

    output_path = tmp_path / "predictions.tsv"
    records = read_protein_fasta(fasta_path, minimum_length=4, maximum_length=20)
    service.write_predictions(records, output_path, batch_size=2)

    predictions = pl.read_csv(output_path, separator="\t")
    assert predictions.columns == [
        "protein_id",
        "length",
        "sequence_sha256",
        "qc_status",
        "profile",
        "model_key",
        "classifier_id",
        "tpase_score",
        "top_family",
        "family_score",
        "final_category",
        "routing_stage",
        "notes",
    ]
    assert predictions["protein_id"].to_list() == ["protein_a", "protein_b"]
    assert predictions["final_category"].to_list() == ["Uncertain", "Negative"]
    assert predictions["routing_stage"].to_list() == ["stage1_sequence", "stage1_sequence"]


def test_linear_classifier_rejects_incompatible_embedding_dimension(tmp_path: Path) -> None:
    artifact_path = tmp_path / "linear_classifier.npz"
    write_linear_artifact(artifact_path)

    classifier = LinearEmbeddingClassifier(artifact_path)

    with pytest.raises(ProteinScreeningError, match="embedding dimension"):
        classifier.predict(np.asarray([[1.0, 2.0, 3.0]]))


def test_fast_profile_has_a_compatible_trained_classifier_artifact() -> None:
    profile = resolve_profile("fast", Path("benchmark/models"))

    classifier = load_classifier(profile)
    predictions = classifier.predict(np.zeros((2, 320), dtype=np.float32))

    assert predictions.scores.shape == (2,)


def test_screen_proteins_cli_reports_an_unavailable_profile_artifact(tmp_path: Path) -> None:
    fasta_path = tmp_path / "proteins.faa"
    fasta_path.write_text(">protein_a\nMAAAAAAAAAAA\n")

    result = CliRunner().invoke(
        app,
        [
            "screen-proteins",
            "--fasta",
            str(fasta_path),
            "--profile",
            "fast",
            "--model-dir",
            str(tmp_path / "missing_models"),
            "--output",
            str(tmp_path / "predictions.tsv"),
        ],
    )

    assert result.exit_code != 0
    assert "classifier metadata" in result.stdout
