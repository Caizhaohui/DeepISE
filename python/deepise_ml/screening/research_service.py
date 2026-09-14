from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
from deepise_ml.screening.evidence import (
    RESEARCH_OUTPUT_COLUMNS,
    CandidateRoute,
    FinalCategory,
    ProstT5Evidence,
    ResearchScreeningRecord,
    StructureEvidence,
)
from deepise_ml.screening.fusion import EvidenceFusionEngine
from deepise_ml.screening.homology import HmmerRun, MMseqsRun, run_hmmsearch, run_mmseqs_search
from deepise_ml.screening.pipeline import (
    DEEPISE_VERSION,
    HomologyScreeningPaths,
    ProteinHomologyScreeningService,
    compute_file_sha256,
)
from deepise_ml.screening.prostt5 import ProstT5Engine, ProstT5RescueRouter
from deepise_ml.screening.routing import CandidateRouter
from deepise_ml.screening.runtime import (
    EmbeddingClassifier,
    EmbeddingEngine,
    ProfileDefinition,
    ProteinInputRecord,
)
from deepise_ml.screening.saprot import SaProtClassifier
from deepise_ml.screening.structure import (
    FoldseekRunner,
    SimulatedStructurePredictor,
    StructurePredictor,
    StructureValidationService,
)


class ProteinResearchScreeningService:
    """Full hierarchical research service orchestrating Stage 1, Homology, Stage 2A, Stage 2B, and Fusion."""

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        classifier: EmbeddingClassifier,
        profile: ProfileDefinition,
        *,
        router: CandidateRouter | None = None,
        prostt5_engine: ProstT5Engine | None = None,
        prostt5_router: ProstT5RescueRouter | None = None,
        structure_predictor: StructurePredictor | None = None,
        foldseek_runner: FoldseekRunner | None = None,
        saprot_classifier: SaProtClassifier | None = None,
        fusion_engine: EvidenceFusionEngine | None = None,
        mmseqs_binary: Path | None = None,
        hmmer_binary: Path | None = None,
        threads: int = 1,
        mmseqs_runner: Callable[..., MMseqsRun] = run_mmseqs_search,
        hmmer_runner: Callable[..., HmmerRun] = run_hmmsearch,
    ) -> None:
        self._embedding_engine = embedding_engine
        self._classifier = classifier
        self._profile = profile
        self._homology_service = ProteinHomologyScreeningService(
            embedding_engine=embedding_engine,
            classifier=classifier,
            profile=profile,
            router=router,
            mmseqs_binary=mmseqs_binary,
            hmmer_binary=hmmer_binary,
            threads=threads,
            mmseqs_runner=mmseqs_runner,
            hmmer_runner=hmmer_runner,
        )
        self._prostt5_engine = prostt5_engine or ProstT5Engine()
        self._prostt5_router = prostt5_router or ProstT5RescueRouter()
        self._structure_predictor = structure_predictor or SimulatedStructurePredictor()
        self._foldseek_runner = foldseek_runner or FoldseekRunner()
        self._saprot_classifier = saprot_classifier or SaProtClassifier()
        self._structure_service = StructureValidationService(
            predictor=self._structure_predictor,
            foldseek_runner=self._foldseek_runner,
            saprot_classifier=self._saprot_classifier,
        )
        self._fusion_engine = fusion_engine or EvidenceFusionEngine()

    def screen_and_fuse(
        self,
        proteins: list[ProteinInputRecord],
        paths: HomologyScreeningPaths,
        batch_size: int = 32,
        enable_stage2a: bool = True,
        enable_stage2b: bool = True,
    ) -> list[ResearchScreeningRecord]:
        # 1. Execute Stage-1 + Homology routing
        homology_records = self._homology_service.screen_and_route(
            proteins=proteins,
            paths=paths,
            batch_size=batch_size,
        )

        prot_map = {p.protein_id: p for p in proteins}

        research_records: list[ResearchScreeningRecord] = []
        for hom_rec in homology_records:
            pid = hom_rec.stage1.protein_id
            prot = prot_map[pid]
            initial_route = hom_rec.decision.route

            # 2. Stage 2A: ProstT5 fast structure rescue
            if enable_stage2a and initial_route in (CandidateRoute.STAGE2A, CandidateRoute.UNCERTAIN):
                p5_ev = self._prostt5_engine.evaluate_sequence(prot.sequence)
                p5_decision = self._prostt5_router.route_stage2a(
                    initial_route=initial_route,
                    prostt5_score=p5_ev.score,
                    prostt5_similarity=p5_ev.structure_similarity,
                )
            else:
                p5_ev = ProstT5Evidence.not_evaluated()
                p5_decision = hom_rec.decision

            # 3. Stage 2B: Explicit 3D structure & SaProt evaluation
            if enable_stage2b and (
                p5_decision.route == CandidateRoute.STAGE2B
                or (initial_route == CandidateRoute.STAGE2A and p5_ev.evaluated and (p5_ev.score or 0) >= 0.60)
            ):
                struct_ev = self._structure_service.validate_candidate(
                    protein_id=pid,
                    sequence=prot.sequence,
                    predicted_3di=p5_ev.predicted_3di,
                )
            else:
                struct_ev = StructureEvidence.not_evaluated()

            # 4. Multi-modal evidence fusion & final categorization
            fusion_ev = self._fusion_engine.fuse(
                homology_record=hom_rec,
                prostt5=p5_ev,
                structure=struct_ev,
            )

            research_records.append(
                ResearchScreeningRecord(
                    homology_record=hom_rec,
                    prostt5=p5_ev,
                    structure=struct_ev,
                    fusion=fusion_ev,
                )
            )

        # 5. Overwrite output TSV with full RESEARCH_OUTPUT_COLUMNS
        paths.output_tsv.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame([rec.tsv_row() for rec in research_records], infer_schema_length=None).select(
            list(RESEARCH_OUTPUT_COLUMNS)
        ).write_csv(paths.output_tsv, separator="\t")

        # 6. Write comprehensive research metadata
        meta_path = paths.resolved_metadata_path()
        meta = {
            "software_version": DEEPISE_VERSION,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "profile": self._profile.name,
            "model_key": self._profile.model_key,
            "classifier_id": self._classifier.classifier_id,
            "input_protein_count": len(proteins),
            "stage2a_enabled": enable_stage2a,
            "stage2b_enabled": enable_stage2b,
            "final_categories": {
                cat.value: sum(1 for r in research_records if r.fusion.final_category == cat)
                for cat in FinalCategory
            },
            "routes": {
                route.value: sum(1 for r in homology_records if r.decision.route == route)
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
            },
            "artifacts": {
                "output_tsv": str(paths.output_tsv.resolve()),
                "mmseqs_raw": str(paths.resolved_mmseqs_raw_output().resolve()),
                "hmmer_raw": str(paths.resolved_hmmer_raw_output().resolve()),
            },
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        return research_records


__all__ = [
    "ProteinResearchScreeningService",
]
