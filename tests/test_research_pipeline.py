import json
from pathlib import Path

import polars as pl
import pytest
from deepise_ml.screening.evidence import (
    FinalCategory,
    RESEARCH_OUTPUT_COLUMNS,
)
from deepise_ml.screening.homology import HmmerHit, HmmerRun, MMseqsHit, MMseqsRun
from deepise_ml.screening.pipeline import HomologyScreeningPaths
from deepise_ml.screening.research_service import ProteinResearchScreeningService
from deepise_ml.screening.runtime import (
    LinearEmbeddingClassifier,
    ProfileDefinition,
    read_protein_fasta,
)
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.test_homology_pipeline import StubEmbeddingEngine, write_linear_artifact


def test_research_pipeline_end_to_end(tmp_path: Path) -> None:
    query_fasta = tmp_path / "query.faa"
    query_fasta.write_text(
        ">prot_known\n" + "M" + "A" * 60 + "\n"
        ">prot_novel\n" + "M" + "C" * 60 + "\n"
        ">prot_neg\n" + "M" + "G" * 60 + "\n"
    )

    ref_fasta = tmp_path / "ref.faa"
    ref_fasta.write_text(">target_known\n" + "M" + "A" * 60 + "\n")

    hmm_db = tmp_path / "profiles.hmm"
    hmm_db.write_text("HMMER3/f\n")
    for sfx in (".h3f", ".h3i", ".h3m", ".h3p"):
        (tmp_path / f"profiles.hmm{sfx}").write_text("pressed\n")

    artifact_path = tmp_path / "linear.npz"
    write_linear_artifact(artifact_path)
    metadata_json = tmp_path / "linear.json"
    metadata_json.write_text(json.dumps({"model_key": "stub_2d", "embedding_dimension": 2}))

    profile = ProfileDefinition(
        name="fast",
        model_key="stub_2d",
        classifier_path=artifact_path,
        metadata_path=metadata_json,
        classifier_kind="linear_npz",
        embedding_dimension=2,
    )

    out_tsv = tmp_path / "predictions_research.tsv"
    paths = HomologyScreeningPaths(
        output_tsv=out_tsv,
        reference_fasta=ref_fasta,
        hmm_database=hmm_db,
    )

    def stub_mmseqs(query_fasta, reference_fasta, raw_output, **kwargs):
        del query_fasta, reference_fasta, kwargs
        raw_output.parent.mkdir(parents=True, exist_ok=True)
        raw_output.write_text("prot_known\ttarget_known\t0.9\t60\t1.0\t1.0\t1e-20\t150.0\n")
        return MMseqsRun(
            raw_output=raw_output,
            best_hits={
                "prot_known": MMseqsHit(
                    query_id="prot_known",
                    target_id="target_known",
                    identity=90.0,
                    alignment_length=60,
                    query_coverage=100.0,
                    target_coverage=100.0,
                    evalue=1e-20,
                    bitscore=150.0,
                )
            },
            executable=Path("/stub/mmseqs"),
        )

    def stub_hmmer(query_fasta, hmm_database, raw_output, **kwargs):
        del query_fasta, hmm_database, kwargs
        raw_output.parent.mkdir(parents=True, exist_ok=True)
        raw_output.write_text("")
        return HmmerRun(
            raw_output=raw_output,
            best_hits={},
            executable=Path("/stub/hmmsearch"),
        )

    service = ProteinResearchScreeningService(
        embedding_engine=StubEmbeddingEngine(),
        classifier=LinearEmbeddingClassifier(artifact_path),
        profile=profile,
        mmseqs_runner=stub_mmseqs,
        hmmer_runner=stub_hmmer,
    )

    proteins = read_protein_fasta(query_fasta, minimum_length=4, maximum_length=2000)
    records = service.screen_and_fuse(
        proteins,
        paths,
        batch_size=4,
        enable_stage2a=True,
        enable_stage2b=True,
    )

    assert len(records) == 3
    assert out_tsv.is_file()

    df = pl.read_csv(out_tsv, separator="\t")
    assert list(df.columns) == list(RESEARCH_OUTPUT_COLUMNS)

    # prot_known has strong homology -> Known-like
    rec_known = df.filter(pl.col("protein_id") == "prot_known").to_dicts()[0]
    assert rec_known["final_category"] == FinalCategory.KNOWN_LIKE.value

    # Metadata exists
    meta_path = paths.resolved_metadata_path()
    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["input_protein_count"] == 3
    assert "final_categories" in meta
