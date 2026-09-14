from pathlib import Path

import pytest
from deepise_ml.screening.evidence import StructureEvidence
from deepise_ml.screening.saprot import SaProtClassifier, format_saprot_bimodal
from deepise_ml.screening.structure import (
    FoldseekRunner,
    SimulatedStructurePredictor,
    StructurePredictionResult,
    StructureValidationService,
)


def test_structure_predictor_generates_pdb_and_plddt(tmp_path: Path) -> None:
    predictor = SimulatedStructurePredictor(output_dir=tmp_path / "structures")
    sequence = "MKTIIALSYIFCLVFA"
    result = predictor.predict_structure("prot_1", sequence)

    assert isinstance(result, StructurePredictionResult)
    assert result.pdb_path.is_file()
    assert 50.0 <= result.mean_plddt <= 100.0
    assert len(result.plddt_per_residue) == len(sequence)

    # Calling again reuses the cached PDB
    cached_result = predictor.predict_structure("prot_1", sequence)
    assert cached_result.pdb_path == result.pdb_path


def test_format_saprot_bimodal() -> None:
    seq = "MKTI"
    threed = "dpdc"
    bimodal = format_saprot_bimodal(seq, threed)

    # SaProt format combines AA with 3Di token
    assert bimodal == "Md Kp Td Ic"


def test_saprot_classifier_scores_bimodal_representation() -> None:
    classifier = SaProtClassifier(classifier_weights=None)  # default heuristic head
    score = classifier.predict_score(sequence="MKTI", threed_tokens="dpdc")

    assert 0.0 <= score <= 1.0


def test_foldseek_runner_extracts_alignment_metrics(tmp_path: Path) -> None:
    pdb_file = tmp_path / "test.pdb"
    lines = [
        f"ATOM  {i:5d}  CA  ALA A{i:4d}    {i*1.5:8.3f}   0.000   0.000  1.00 85.00           C\n"
        for i in range(1, 15)
    ]
    pdb_file.write_text("".join(lines))

    runner = FoldseekRunner(executable=None)  # fallback to internal geometric aligner
    hit = runner.search_against_reference(pdb_file, reference_db=None)

    assert hit.foldseek_hit is True
    assert hit.foldseek_score is not None
    assert hit.foldseek_evalue is not None


def test_structure_validation_service_orchestration(tmp_path: Path) -> None:
    predictor = SimulatedStructurePredictor(output_dir=tmp_path / "structures")
    foldseek = FoldseekRunner(executable=None)
    saprot = SaProtClassifier(classifier_weights=None)

    service = StructureValidationService(
        predictor=predictor,
        foldseek_runner=foldseek,
        saprot_classifier=saprot,
    )

    sequence = "MKTIIALSYIFCLVFA"
    evidence = service.validate_candidate(
        protein_id="prot_candidate",
        sequence=sequence,
        predicted_3di="dpdc" * 4,
    )

    assert isinstance(evidence, StructureEvidence)
    assert evidence.evaluated is True
    assert evidence.mean_plddt is not None
    assert evidence.foldseek_hit is True
    assert evidence.foldseek_score is not None
    assert evidence.saprot_score is not None
