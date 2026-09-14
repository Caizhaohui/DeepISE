import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Protocol

import numpy as np
import polars as pl
from Bio import SeqIO
from pydantic import BaseModel, ConfigDict, Field

from deepise_ml.models.plm_family import PLMFamilyClassifier
from deepise_ml.schemas.records import compute_sha256

ALLOWED_AMINO_ACIDS: Final = frozenset("ACDEFGHIKLMNPQRSTVWYXBZUO")
OUTPUT_COLUMNS: Final = [
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
ClassifierKind = Literal["linear_npz", "multitask_torch"]


class ProteinScreeningError(RuntimeError):
    pass


class ProteinInputRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    protein_id: str = Field(min_length=1)
    sequence: str = Field(min_length=1)
    length: int = Field(ge=1)
    sequence_sha256: str = Field(min_length=64, max_length=64)
    qc_status: Literal["passed", "stop_symbol_normalized"]


class ProteinScreeningRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    protein_id: str
    length: int
    sequence_sha256: str
    qc_status: Literal["passed", "stop_symbol_normalized"]
    profile: str
    model_key: str
    classifier_id: str
    tpase_score: float = Field(ge=0.0, le=1.0)
    top_family: str | None
    family_score: float | None = Field(default=None, ge=0.0, le=1.0)
    final_category: Literal["Uncertain", "Negative"]
    routing_stage: Literal["stage1_sequence"]
    notes: str

    def tsv_row(self) -> dict[str, str | int | float | None]:
        return self.model_dump()


@dataclass(frozen=True, slots=True)
class ProfileDefinition:
    name: Literal["fast", "standard"]
    model_key: str
    classifier_path: Path
    metadata_path: Path
    classifier_kind: ClassifierKind
    embedding_dimension: int


@dataclass(frozen=True, slots=True)
class ClassifierOutput:
    scores: np.ndarray
    families: tuple[str | None, ...]
    family_scores: tuple[float | None, ...]
    threshold: float


class EmbeddingEngine(Protocol):
    def extract_sequence_embeddings(
        self,
        sequences: list[str],
        batch_size: int,
        verbose: bool,
    ) -> np.ndarray: ...


class EmbeddingClassifier(Protocol):
    @property
    def classifier_id(self) -> str: ...

    def predict(self, embeddings: np.ndarray) -> ClassifierOutput: ...


class LinearEmbeddingClassifier:
    def __init__(self, artifact_path: Path):
        if not artifact_path.is_file():
            raise ProteinScreeningError(f"classifier artifact is unavailable: {artifact_path}")
        try:
            with np.load(artifact_path) as artifact:
                self._mean = np.asarray(artifact["mean"], dtype=np.float32)
                self._scale = np.asarray(artifact["scale"], dtype=np.float32)
                self._coefficients = np.asarray(artifact["coefficients"], dtype=np.float32)
                self._intercept = float(np.asarray(artifact["intercept"]).item())
                self._threshold = float(np.asarray(artifact["threshold"]).item())
        except (KeyError, OSError, ValueError) as error:
            raise ProteinScreeningError(f"invalid classifier artifact: {artifact_path}") from error
        if self._mean.shape != self._scale.shape or self._mean.shape != self._coefficients.shape:
            raise ProteinScreeningError(f"invalid classifier artifact dimensions: {artifact_path}")
        self._classifier_id = artifact_path.stem

    @property
    def classifier_id(self) -> str:
        return self._classifier_id

    def predict(self, embeddings: np.ndarray) -> ClassifierOutput:
        if embeddings.ndim != 2 or embeddings.shape[1] != self._coefficients.shape[0]:
            raise ProteinScreeningError(
                "embedding dimension does not match classifier artifact "
                f"({embeddings.shape[-1] if embeddings.ndim else 0} != {self._coefficients.shape[0]})"
            )
        safe_scale = np.where(self._scale == 0.0, 1.0, self._scale)
        logits = ((embeddings - self._mean) / safe_scale) @ self._coefficients + self._intercept
        scores = 1.0 / (1.0 + np.exp(-logits))
        count = len(scores)
        return ClassifierOutput(
            scores=np.asarray(scores, dtype=np.float32),
            families=(None,) * count,
            family_scores=(None,) * count,
            threshold=self._threshold,
        )


class MultiTaskEmbeddingClassifier:
    def __init__(self, artifact_path: Path):
        if not artifact_path.is_file():
            raise ProteinScreeningError(f"classifier artifact is unavailable: {artifact_path}")
        self._classifier = PLMFamilyClassifier()
        try:
            self._classifier.load(artifact_path)
        except (KeyError, OSError, RuntimeError, ValueError) as error:
            raise ProteinScreeningError(f"invalid classifier artifact: {artifact_path}") from error
        self._classifier_id = artifact_path.stem

    @property
    def classifier_id(self) -> str:
        return self._classifier_id

    def predict(self, embeddings: np.ndarray) -> ClassifierOutput:
        if embeddings.ndim != 2 or embeddings.shape[1] != self._classifier.input_dim:
            raise ProteinScreeningError(
                "embedding dimension does not match classifier artifact "
                f"({embeddings.shape[-1] if embeddings.ndim else 0} != {self._classifier.input_dim})"
            )
        scores, family_probabilities = self._classifier.predict_proba(embeddings)
        indices = family_probabilities.argmax(axis=1)
        families = tuple(self._classifier.idx_to_family[int(index)] for index in indices)
        family_scores = tuple(float(value) for value in family_probabilities.max(axis=1))
        return ClassifierOutput(
            scores=np.asarray(scores, dtype=np.float32),
            families=families,
            family_scores=family_scores,
            threshold=0.5,
        )


def read_protein_fasta(
    fasta_path: Path,
    minimum_length: int,
    maximum_length: int,
    skip_invalid_lengths: bool = False,
) -> list[ProteinInputRecord]:
    if not fasta_path.is_file():
        raise ProteinScreeningError(f"protein FASTA does not exist: {fasta_path}")
    if minimum_length < 1 or maximum_length < minimum_length:
        raise ProteinScreeningError("invalid protein length bounds")
    seen_ids: set[str] = set()
    records: list[ProteinInputRecord] = []
    for fasta_record in SeqIO.parse(fasta_path, "fasta"):
        protein_id = str(fasta_record.id)
        if protein_id in seen_ids:
            raise ProteinScreeningError(f"duplicate protein ID: {protein_id}")
        seen_ids.add(protein_id)
        sequence = str(fasta_record.seq).strip().upper()
        if not sequence:
            raise ProteinScreeningError(f"empty protein sequence: {protein_id}")
        without_terminal_stops = sequence.rstrip("*")
        normalized_sequence = without_terminal_stops.replace("*", "X")
        qc_status = (
            "stop_symbol_normalized"
            if normalized_sequence != sequence
            else "passed"
        )
        sequence = normalized_sequence
        if not sequence:
            raise ProteinScreeningError(f"protein sequence contains only stop symbols: {protein_id}")
        invalid_residues = sorted(set(sequence) - ALLOWED_AMINO_ACIDS)
        if invalid_residues:
            raise ProteinScreeningError(
                f"invalid amino-acid residues for {protein_id}: {''.join(invalid_residues)}"
            )
        if not minimum_length <= len(sequence) <= maximum_length:
            if skip_invalid_lengths:
                continue
            raise ProteinScreeningError(
                f"protein length outside [{minimum_length}, {maximum_length}] for {protein_id}"
            )
        records.append(
            ProteinInputRecord(
                protein_id=protein_id,
                sequence=sequence,
                length=len(sequence),
                sequence_sha256=compute_sha256(sequence),
                qc_status=qc_status,
            )
        )
    if not records:
        if skip_invalid_lengths:
            return []
        raise ProteinScreeningError(f"protein FASTA contains no records: {fasta_path}")
    return records


class ProteinScreeningService:
    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        classifier: EmbeddingClassifier,
        profile: ProfileDefinition,
    ):
        self._embedding_engine = embedding_engine
        self._classifier = classifier
        self._profile = profile

    def predict_stage1(
        self,
        proteins: list[ProteinInputRecord],
        batch_size: int,
    ) -> list[ProteinScreeningRecord]:
        embeddings = self._embedding_engine.extract_sequence_embeddings(
            [protein.sequence for protein in proteins],
            batch_size=batch_size,
            verbose=False,
        )
        predictions = self._classifier.predict(embeddings)
        if len(predictions.scores) != len(proteins):
            raise ProteinScreeningError("embedding model returned an unexpected prediction count")
        return [
            self._prediction_record(protein, predictions, index)
            for index, protein in enumerate(proteins)
        ]

    def write_predictions(
        self,
        proteins: list[ProteinInputRecord],
        output_path: Path,
        batch_size: int,
    ) -> list[ProteinScreeningRecord]:
        records = self.predict_stage1(proteins, batch_size)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame([record.tsv_row() for record in records]).select(OUTPUT_COLUMNS).write_csv(
            output_path,
            separator="\t",
        )
        return records

    def _prediction_record(
        self,
        protein: ProteinInputRecord,
        predictions: ClassifierOutput,
        index: int,
    ) -> ProteinScreeningRecord:
        score = float(predictions.scores[index])
        is_sequence_candidate = score >= predictions.threshold
        return ProteinScreeningRecord(
            protein_id=protein.protein_id,
            length=protein.length,
            sequence_sha256=protein.sequence_sha256,
            qc_status=protein.qc_status,
            profile=self._profile.name,
            model_key=self._profile.model_key,
            classifier_id=self._classifier.classifier_id,
            tpase_score=score,
            top_family=predictions.families[index],
            family_score=predictions.family_scores[index],
            final_category="Uncertain" if is_sequence_candidate else "Negative",
            routing_stage="stage1_sequence",
            notes="sequence_candidate_pending_homology" if is_sequence_candidate else "sequence_negative",
        )


def resolve_profile(profile_name: str, model_dir: Path) -> ProfileDefinition:
    match profile_name:
        case "fast":
            return ProfileDefinition(
                name="fast",
                model_key="esm2_8m",
                classifier_path=model_dir / "esm2_8m_linear_classifier.npz",
                metadata_path=model_dir / "esm2_8m_linear_classifier.json",
                classifier_kind="linear_npz",
                embedding_dimension=320,
            )
        case "standard":
            return ProfileDefinition(
                name="standard",
                model_key="esm2_35m",
                classifier_path=model_dir / "plm_family_classifier.pt",
                metadata_path=model_dir / "plm_family_classifier.json",
                classifier_kind="multitask_torch",
                embedding_dimension=480,
            )
        case _:
            raise ProteinScreeningError(f"unsupported screening profile: {profile_name}")


def load_classifier(profile: ProfileDefinition) -> EmbeddingClassifier:
    _validate_artifact_metadata(profile)
    match profile.classifier_kind:
        case "linear_npz":
            return LinearEmbeddingClassifier(profile.classifier_path)
        case "multitask_torch":
            return MultiTaskEmbeddingClassifier(profile.classifier_path)


def _validate_artifact_metadata(profile: ProfileDefinition) -> None:
    if not profile.metadata_path.is_file():
        raise ProteinScreeningError(f"classifier metadata is unavailable: {profile.metadata_path}")
    try:
        metadata = json.loads(profile.metadata_path.read_text())
        metadata_model_key = metadata["model_key"]
        metadata_dimension = metadata["embedding_dimension"]
    except (json.JSONDecodeError, KeyError, OSError, TypeError) as error:
        raise ProteinScreeningError(f"invalid classifier metadata: {profile.metadata_path}") from error
    if metadata_model_key != profile.model_key:
        raise ProteinScreeningError(
            f"classifier metadata model mismatch: {metadata_model_key} != {profile.model_key}"
        )
    if metadata_dimension != profile.embedding_dimension:
        raise ProteinScreeningError(
            "classifier metadata embedding dimension mismatch: "
            f"{metadata_dimension} != {profile.embedding_dimension}"
        )
