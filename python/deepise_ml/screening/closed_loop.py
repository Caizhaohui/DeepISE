"""Closed-loop bidirectional integration bridging genome scanning and protein structure research."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Optional

import polars as pl
from deepise_ml.models.plm_registry import PluggablePLMEngine
from deepise_ml.screening.evidence import FinalCategory
from deepise_ml.screening.pipeline import HomologyScreeningPaths
from deepise_ml.screening.research_service import ProteinResearchScreeningService
from deepise_ml.screening.runtime import load_classifier, read_protein_fasta, resolve_profile


class DualValidationCategory(StrEnum):
    """Integrated dual-layer classification combining genomic architecture and protein PLM/structure."""
    HIGH_CONFIDENCE_NOVEL = "High_Confidence_Novel_IS"
    HIGH_CONFIDENCE_KNOWN = "High_Confidence_Known_IS"
    CANDIDATE_NOVEL = "Candidate_Novel_IS"
    CONFIRMED_STANDARD = "Confirmed_Standard_IS"
    STANDARD = "Standard_IS"


@dataclass(frozen=True, slots=True)
class ClosedLoopResult:
    research_predictions_path: Path
    novel_discoveries_path: Path
    summary_path: Optional[Path]
    total_tpases_evaluated: int
    high_confidence_novel_count: int
    high_confidence_known_count: int
    candidate_novel_count: int
    remote_count: int
    novel_candidate_count: int


def perform_closed_loop_research_validation(
    elements_tsv: Path,
    tpases_faa: Path,
    output_dir: Path,
    *,
    reference_fasta: Optional[Path] = None,
    hmm_database: Optional[Path] = None,
    profile: str = "fast",
    model_dir: Path = Path("benchmark/models"),
    device: Optional[str] = None,
    enable_stage2a: bool = True,
    enable_stage2b: bool = True,
    batch_size: int = 32,
    threads: int = 1,
    summary_json: Optional[Path] = None,
    mmseqs_binary: Optional[Path] = None,
    hmmer_binary: Optional[Path] = None,
    mmseqs_runner: Any = None,
    hmmer_runner: Any = None,
) -> ClosedLoopResult:
    """Run closed-loop protein structure research and integrate with genomic element predictions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if mmseqs_binary is None:
        cand_mm = Path(sys.executable).parent / "mmseqs"
        if cand_mm.is_file():
            mmseqs_binary = cand_mm

    if hmmer_binary is None:
        cand_hmm = Path(sys.executable).parent / "hmmsearch"
        if cand_hmm.is_file():
            hmmer_binary = cand_hmm

    research_tsv = output_dir / "deepise_tpases_research.tsv"
    novel_tsv = output_dir / "deepise_novel_discoveries.tsv"

    if not tpases_faa.exists() or tpases_faa.stat().st_size == 0:
        # Write empty discoveries table
        empty_df = pl.DataFrame(schema={
            "element_id": pl.Utf8,
            "contig_id": pl.Utf8,
            "start": pl.Int64,
            "end": pl.Int64,
            "length": pl.Int64,
            "family": pl.Utf8,
            "status": pl.Utf8,
            "composite_score": pl.Float64,
            "protein_id": pl.Utf8,
            "transposase_score": pl.Float64,
            "novelty_score": pl.Float64,
            "final_category": pl.Utf8,
            "dual_category": pl.Utf8,
            "dual_confidence": pl.Float64,
        })
        empty_df.write_csv(novel_tsv, separator="\t")
        return ClosedLoopResult(
            research_predictions_path=research_tsv,
            novel_discoveries_path=novel_tsv,
            summary_path=summary_json,
            total_tpases_evaluated=0,
            high_confidence_novel_count=0,
            high_confidence_known_count=0,
            candidate_novel_count=0,
            remote_count=0,
            novel_candidate_count=0,
        )

    # Resolve reference fasta fallback
    if reference_fasta is None or not reference_fasta.exists():
        candidates = [
            Path("data/splits/cluster30/train.faa"),
            Path("benchmark/tmp_hmmer/IS1.faa"),
            Path("data/processed/tpases.faa"),
            tpases_faa,
        ]
        for c in candidates:
            if c.exists() and c.stat().st_size > 0:
                reference_fasta = c
                break

    # Resolve HMM database fallback
    if hmm_database is None or not hmm_database.exists():
        candidates_hmm = [
            Path("benchmark/db/deepise_tpases.hmm"),
            Path("benchmark/tmp_hmmer/IS1.hmm"),
        ]
        for ch in candidates_hmm:
            if ch.exists():
                hmm_database = ch
                break

    resolved_profile = resolve_profile(profile, model_dir)
    classifier = load_classifier(resolved_profile)
    proteins = read_protein_fasta(
        tpases_faa,
        minimum_length=30,
        maximum_length=3000,
        skip_invalid_lengths=True,
    )

    if proteins:
        paths = HomologyScreeningPaths(
            output_tsv=research_tsv,
            reference_fasta=reference_fasta,
            hmm_database=hmm_database,
        )
        paths.validate_inputs_and_destinations()

        engine = PluggablePLMEngine(model_key=resolved_profile.model_key, device=device)
        service_kwargs: dict[str, Any] = {
            "embedding_engine": engine,
            "classifier": classifier,
            "profile": resolved_profile,
            "mmseqs_binary": mmseqs_binary,
            "hmmer_binary": hmmer_binary,
            "threads": threads,
        }
        if mmseqs_runner is not None:
            service_kwargs["mmseqs_runner"] = mmseqs_runner
        if hmmer_runner is not None:
            service_kwargs["hmmer_runner"] = hmmer_runner
        service = ProteinResearchScreeningService(**service_kwargs)
        research_records = service.screen_and_fuse(
            proteins=proteins,
            paths=paths,
            batch_size=batch_size,
            enable_stage2a=enable_stage2a,
            enable_stage2b=enable_stage2b,
        )
    else:
        research_records = []

    # Build lookup map of research records: protein_id -> record
    res_map = {r.protein_id: r for r in research_records}

    # Load genomic elements
    elem_df = pl.read_csv(elements_tsv, separator="\t") if elements_tsv.exists() and elements_tsv.stat().st_size > 0 else pl.DataFrame()

    discovery_rows = []
    high_conf_novel = 0
    high_conf_known = 0
    candidate_novel = 0
    remote_cnt = 0
    novel_cand_cnt = 0

    if len(elem_df) > 0:
        for row in elem_df.to_dicts():
            elem_id = str(row.get("element_id", ""))
            tp_gene_id = str(row.get("tpase_gene_id", ""))
            status = str(row.get("status", ""))
            comp_score = float(row.get("composite_score", 0.0))

            # Match protein: try {elem_id}_tpase, or elem_id, or tp_gene_id
            target_pid = None
            for cand_id in (f"{elem_id}_tpase", elem_id, tp_gene_id):
                if cand_id in res_map:
                    target_pid = cand_id
                    break
            if target_pid is None:
                # Fuzzy prefix match
                for pid in res_map:
                    if elem_id in pid or (tp_gene_id and tp_gene_id in pid):
                        target_pid = pid
                        break

            if target_pid and target_pid in res_map:
                r_rec = res_map[target_pid]
                tpase_score = r_rec.fusion.transposase_score
                novelty_score = r_rec.fusion.novelty_score
                final_cat = r_rec.fusion.final_category
            else:
                tpase_score = float(row.get("tpase_score", 0.0))
                novelty_score = 0.0
                final_cat = FinalCategory.UNCERTAIN

            if final_cat == FinalCategory.REMOTE:
                remote_cnt += 1
            elif final_cat == FinalCategory.NOVEL_CANDIDATE:
                novel_cand_cnt += 1

            # Dual-layer category assignment
            if status == "complete" and final_cat in (FinalCategory.NOVEL_CANDIDATE, FinalCategory.REMOTE):
                dual_cat = DualValidationCategory.HIGH_CONFIDENCE_NOVEL
                high_conf_novel += 1
            elif status == "complete" and final_cat == FinalCategory.KNOWN_LIKE:
                dual_cat = DualValidationCategory.HIGH_CONFIDENCE_KNOWN
                high_conf_known += 1
            elif status != "complete" and final_cat in (FinalCategory.NOVEL_CANDIDATE, FinalCategory.REMOTE):
                dual_cat = DualValidationCategory.CANDIDATE_NOVEL
                candidate_novel += 1
            elif status == "complete":
                dual_cat = DualValidationCategory.CONFIRMED_STANDARD
            else:
                dual_cat = DualValidationCategory.STANDARD

            dual_conf = round(0.50 * comp_score + 0.50 * tpase_score, 4)

            discovery_rows.append({
                "element_id": elem_id,
                "contig_id": str(row.get("contig_id", "")),
                "start": int(row.get("start", 0)),
                "end": int(row.get("end", 0)),
                "length": int(row.get("length", 0)),
                "family": str(row.get("family", "")),
                "status": status,
                "composite_score": comp_score,
                "protein_id": target_pid or tp_gene_id,
                "transposase_score": tpase_score,
                "novelty_score": novelty_score,
                "final_category": str(final_cat.value if hasattr(final_cat, "value") else final_cat),
                "dual_category": dual_cat.value,
                "dual_confidence": dual_conf,
            })

    discovery_df = pl.DataFrame(discovery_rows, infer_schema_length=None) if discovery_rows else pl.DataFrame()
    discovery_df.write_csv(novel_tsv, separator="\t")

    # Update summary JSON if present
    if summary_json and summary_json.exists():
        try:
            summary_data = json.loads(summary_json.read_text(encoding="utf-8"))
            summary_data["research_validation"] = {
                "research_enabled": True,
                "profile": profile,
                "total_tpases_evaluated": len(proteins),
                "high_confidence_novel_elements": high_conf_novel,
                "high_confidence_known_elements": high_conf_known,
                "candidate_novel_elements": candidate_novel,
                "remote_transposases": remote_cnt,
                "novel_transposase_candidates": novel_cand_cnt,
                "research_predictions_tsv": str(research_tsv.resolve()),
                "novel_discoveries_tsv": str(novel_tsv.resolve()),
            }
            summary_json.write_text(json.dumps(summary_data, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass

    return ClosedLoopResult(
        research_predictions_path=research_tsv,
        novel_discoveries_path=novel_tsv,
        summary_path=summary_json,
        total_tpases_evaluated=len(proteins),
        high_confidence_novel_count=high_conf_novel,
        high_confidence_known_count=high_conf_known,
        candidate_novel_count=candidate_novel,
        remote_count=remote_cnt,
        novel_candidate_count=novel_cand_cnt,
    )


__all__ = [
    "ClosedLoopResult",
    "DualValidationCategory",
    "perform_closed_loop_research_validation",
]
