"""Metagenomic IS discovery and edge-truncation resolution engine (Phase-4).

Optimized for multi-contig assemblies, fragmented metagenome-assembled genomes (MAGs),
and short contig streams with edge-truncation classification and memory-efficient batching.
"""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import polars as pl
import pyrodigal
from Bio import SeqIO
from pydantic import BaseModel, Field

from deepise_ml.boundary.hybrid import HybridBoundaryEngine
from deepise_ml.boundary.plan_a import FAMILY_DISTANCE_PRIORS, PlanAAdaptiveEngine
from deepise_ml.boundary.plan_b import PlanBCanonicalEngine
from deepise_ml.boundary.schemas import BoundaryPrediction
from deepise_ml.composite.scorer import ISCompositeResult, ISCompositeScorer
from deepise_ml.models.hmmer import parse_hmmsearch_predictions


class MetagenomeISElement(BaseModel):
    """Full or partial IS element detected in a metagenomic contig."""
    element_id: str
    contig_id: str
    start: int = Field(..., description="0-indexed start coordinate in contig")
    end: int = Field(..., description="0-indexed end coordinate in contig (exclusive)")
    length: int
    contig_length: int
    strand: str
    family: str
    tpase_gene_id: str
    tpase_score: float
    tpase_evalue: float
    composite_score: float
    status: str = Field(..., description="'complete', 'partial', or 'pseudo'")
    truncation_status: str = Field(
        ...,
        description="'complete', 'edge_5p_truncated', 'edge_3p_truncated', 'edge_both_truncated', or 'internal_partial'",
    )
    method: str
    tir_length: Optional[int] = None
    tir_identity: Optional[float] = None
    tsd_length: Optional[int] = None
    tsd_sequence: Optional[str] = None
    structural_evidence: Optional[str] = None
    notes: str = ""
    dna_sequence: str = Field(default="", description="Nucleotide sequence of the IS element")
    protein_sequence: str = Field(default="", description="Transposase protein translation")


class MetagenomeScanner:
    """Production metagenome scanner supporting contig streaming and edge truncation."""

    def __init__(
        self,
        hmm_path: Path = Path("benchmark/db/deepise_tpases.hmm"),
        min_bitscore: float = 10.6,
        hmmer_bin: str = "/hpcfs/fhome/caizhh/.conda/envs/DeepISE/bin/hmmsearch",
        min_contig_len: int = 500,
        batch_size: int = 1000,
        edge_tolerance_bp: int = 40,
        boundary_mode: str = "plan_a",
    ):
        self.hmm_path = Path(hmm_path)
        self.min_bitscore = min_bitscore
        self.hmmer_bin = hmmer_bin
        self.min_contig_len = min_contig_len
        self.batch_size = batch_size
        self.edge_tolerance_bp = edge_tolerance_bp
        self.boundary_mode = boundary_mode

        # Initialize engines
        self.engine_a = PlanAAdaptiveEngine()
        self.engine_b = PlanBCanonicalEngine()
        self.engine_hybrid = HybridBoundaryEngine()
        self.scorer = ISCompositeScorer()
        # Metagenomic gene finder (no per-contig training required)
        self.gf = pyrodigal.GeneFinder(meta=True)

    def scan(
        self,
        fasta_path: Path,
        threads: int = 4,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Tuple[List[MetagenomeISElement], Dict[str, Any]]:
        """Stream a metagenomic FASTA file and detect full/truncated IS elements."""
        fasta_path = Path(fasta_path)
        assert fasta_path.exists(), f"Input FASTA {fasta_path} does not exist."

        t_start = time.perf_counter()

        stats: Dict[str, Any] = {
            "deepise_version": "0.1.0",
            "pipeline": "metagenome_scan",
            "input_file": str(fasta_path.resolve()),
            "boundary_mode": self.boundary_mode,
            "min_contig_len": self.min_contig_len,
            "min_bitscore": self.min_bitscore,
            "total_contigs_read": 0,
            "total_contigs_analyzed": 0,
            "total_contigs_filtered_short": 0,
            "total_bp_scanned": 0,
            "total_orfs_predicted": 0,
            "total_is_elements_detected": 0,
            "breakdown_by_status": {"complete": 0, "partial": 0, "pseudo": 0},
            "breakdown_by_truncation": {
                "complete": 0,
                "edge_5p_truncated": 0,
                "edge_3p_truncated": 0,
                "edge_both_truncated": 0,
                "internal_partial": 0,
            },
            "breakdown_by_family": {},
            "runtime_seconds": 0.0,
            "throughput_mbp_per_sec": 0.0,
            "throughput_contigs_per_sec": 0.0,
        }

        all_detected: List[MetagenomeISElement] = []
        global_elem_idx = 0

        # Stream FASTA in batches
        batch_contigs: List[Tuple[int, str, str, bytes]] = []  # (contig_idx, id, seq, bytes)
        contig_idx = 0

        def process_batch(
            b_contigs: List[Tuple[int, str, str, bytes]]
        ) -> List[MetagenomeISElement]:
            nonlocal global_elem_idx
            if not b_contigs:
                return []

            batch_detected: List[MetagenomeISElement] = []
            gene_registry: Dict[str, Dict[str, Any]] = {}
            translations_faa: List[str] = []

            # 1. Metagenome ORF prediction for batch
            for c_idx, c_id, c_seq, c_bytes in b_contigs:
                genes = self.gf.find_genes(c_bytes)
                for g_idx, g in enumerate(genes, start=1):
                    tag = f"c{c_idx}_g{g_idx}"
                    prot_seq = g.translate()
                    stats["total_orfs_predicted"] += 1

                    gene_registry[tag] = {
                        "tag": tag,
                        "contig_idx": c_idx,
                        "contig_id": c_id,
                        "contig_seq": c_seq,
                        "contig_len": len(c_seq),
                        "gene_idx": g_idx,
                        "begin": g.begin - 1,  # 0-indexed
                        "end": g.end,
                        "strand": "+" if g.strand == 1 else "-",
                        "partial_begin": bool(g.partial_begin),
                        "partial_end": bool(g.partial_end),
                        "protein": prot_seq,
                    }
                    translations_faa.append(f">{tag}\n{prot_seq}\n")

            if not translations_faa:
                return []

            # 2. Batch HMMER search
            with tempfile.NamedTemporaryFile(suffix=".faa", mode="w") as faa_f, \
                 tempfile.NamedTemporaryFile(suffix=".tblout") as tbl_f:
                faa_f.write("".join(translations_faa))
                faa_f.flush()

                env = os.environ.copy()
                env["PATH"] = f"/hpcfs/fhome/caizhh/.conda/envs/DeepISE/bin:{env.get('PATH', '')}"

                cmd = [
                    self.hmmer_bin,
                    "--noali",
                    "--cpu",
                    str(threads),
                    "--tblout",
                    tbl_f.name,
                    str(self.hmm_path),
                    faa_f.name,
                ]
                subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                    env=env,
                )
                preds = parse_hmmsearch_predictions(Path(tbl_f.name), Path(faa_f.name))

            tp_hits = preds.filter(pl.col("score") >= self.min_bitscore)
            if len(tp_hits) == 0:
                return []

            # Group hits by seq_id (best HMM hit per ORF)
            tp_best: Dict[str, Tuple[str, float, float]] = {}
            for row in tp_hits.iter_rows(named=True):
                s_id = row["seq_id"]
                target = row["target_id"]
                family = target.replace("Tpase_", "")
                score = float(row["score"])
                evalue = float(row["evalue"])
                if s_id not in tp_best or score > tp_best[s_id][1]:
                    tp_best[s_id] = (family, score, evalue)

            # Group hits by contig
            contig_hits: Dict[int, List[Dict[str, Any]]] = {}
            for s_id, (fam, sc, ev) in tp_best.items():
                if s_id not in gene_registry:
                    continue
                g_info = dict(gene_registry[s_id])
                g_info["family"] = fam
                g_info["score"] = sc
                g_info["evalue"] = ev
                c_idx = g_info["contig_idx"]
                contig_hits.setdefault(c_idx, []).append(g_info)

            # 3. Process each contig with transposase hits
            for c_idx, hits in contig_hits.items():
                hits.sort(key=lambda x: x["begin"])
                c_id = hits[0]["contig_id"]
                c_seq = hits[0]["contig_seq"]
                c_len = hits[0]["contig_len"]

                # Cluster adjacent transposases (< 250 bp and same family)
                clusters: List[List[Dict[str, Any]]] = []
                for h in hits:
                    if not clusters:
                        clusters.append([h])
                    else:
                        prev = clusters[-1][-1]
                        if h["begin"] - prev["end"] <= 250 and h["family"] == prev["family"]:
                            clusters[-1].append(h)
                        else:
                            clusters.append([h])

                # Resolve boundaries for each cluster
                for cluster in clusters:
                    global_elem_idx += 1
                    c_start = cluster[0]["begin"]
                    c_end = cluster[-1]["end"]
                    c_strand = cluster[0]["strand"]
                    c_fam = cluster[0]["family"]
                    best_tp_score = max(c["score"] for c in cluster)
                    min_tp_evalue = min(c["evalue"] for c in cluster)
                    primary_gene_tag = cluster[0]["tag"]
                    primary_gene_id = f"{c_id}_{cluster[0]['gene_idx']}"
                    tpase_protein = cluster[0]["protein"]

                    # Assess edge truncation based on ORF proximity & partial flags
                    is_at_5p_edge = (
                        c_start <= self.edge_tolerance_bp
                        or any(
                            (c["strand"] == "+" and c["partial_begin"])
                            or (c["strand"] == "-" and c["partial_end"])
                            for c in cluster
                        )
                    )
                    is_at_3p_edge = (
                        c_end >= (c_len - self.edge_tolerance_bp)
                        or any(
                            (c["strand"] == "+" and c["partial_end"])
                            or (c["strand"] == "-" and c["partial_begin"])
                            for c in cluster
                        )
                    )

                    # Determine boundary prediction
                    if is_at_5p_edge and is_at_3p_edge:
                        # Contig fragment enclosed within IS element
                        pred_start = 0
                        pred_end = c_len
                        b_pred = BoundaryPrediction(
                            contig_id=c_id,
                            is_name=f"{c_fam}_{global_elem_idx}",
                            family=c_fam,
                            method=self.boundary_mode,
                            predicted_start=0,
                            predicted_end=c_len,
                            predicted_length=c_len,
                            is_complete=False,
                            confidence_score=0.45,
                            tir=None,
                            tsd=None,
                            structure=None,
                            latency_ms=0.0,
                        )
                        trunc_status = "edge_both_truncated"
                        elem_status = "partial"
                        notes = "Element truncated at both 5' and 3' contig edges"
                    else:
                        # Boundary prediction engine
                        if self.boundary_mode in ["hybrid", "plan_a_hybrid"]:
                            b_pred = self.engine_hybrid.predict_boundary(
                                contig=c_seq,
                                tpase_start=c_start,
                                tpase_end=c_end,
                                contig_id=c_id,
                                is_name=f"{c_fam}_{global_elem_idx}",
                                family=c_fam,
                            )
                        elif self.boundary_mode == "plan_b":
                            b_pred = self.engine_b.predict_boundary(
                                contig=c_seq,
                                tpase_start=c_start,
                                tpase_end=c_end,
                                contig_id=c_id,
                                is_name=f"{c_fam}_{global_elem_idx}",
                                family=c_fam,
                            )
                        else:
                            b_pred = self.engine_a.predict_boundary(
                                contig=c_seq,
                                tpase_start=c_start,
                                tpase_end=c_end,
                                contig_id=c_id,
                                is_name=f"{c_fam}_{global_elem_idx}",
                                family=c_fam,
                            )

                        # Family distance priors
                        priors = FAMILY_DISTANCE_PRIORS.get(c_fam, (40, 350, 20, 300))
                        min_up, max_up, min_down, max_down = priors

                        # Coordinate clamping
                        pred_start = max(0, min(c_len - 1, b_pred.predicted_start))
                        pred_end = max(pred_start + 1, min(c_len, b_pred.predicted_end))

                        # Evaluate structural evidence
                        has_tir_or_struct = (b_pred.tir is not None and b_pred.tir.length >= 8) or (b_pred.structure is not None)

                        # Evaluate truncation
                        has_5p_trunc = (
                            is_at_5p_edge
                            or pred_start <= 15
                            or (c_start < min_up and not has_tir_or_struct)
                            or (pred_start <= max(self.edge_tolerance_bp, 60) and not has_tir_or_struct)
                        )
                        has_3p_trunc = (
                            is_at_3p_edge
                            or pred_end >= (c_len - 15)
                            or ((c_len - c_end) < min_down and not has_tir_or_struct)
                            or (pred_end >= (c_len - max(self.edge_tolerance_bp, 60)) and not has_tir_or_struct)
                        )

                        if has_5p_trunc and has_3p_trunc:
                            trunc_status = "edge_both_truncated"
                            pred_start = 0
                            pred_end = c_len
                            elem_status = "partial"
                            notes = "Truncated at both 5' and 3' contig borders"
                        elif has_5p_trunc:
                            trunc_status = "edge_5p_truncated"
                            pred_start = 0
                            elem_status = "partial"
                            notes = "Truncated at 5' contig border"
                        elif has_3p_trunc:
                            trunc_status = "edge_3p_truncated"
                            pred_end = c_len
                            elem_status = "partial"
                            notes = "Truncated at 3' contig border"
                        else:
                            if b_pred.is_complete or has_tir_or_struct:
                                trunc_status = "complete"
                                elem_status = "complete"
                                notes = "Complete full-length IS element"
                            else:
                                trunc_status = "internal_partial"
                                elem_status = "partial"
                                notes = "Internal partial IS element"

                    # Composite scoring
                    comp_res = self.scorer.score_element(
                        boundary_pred=b_pred,
                        tpase_score=min(1.0, best_tp_score / 150.0),
                        tpase_length_bp=c_end - c_start,
                    )

                    # Preserve edge status if composite downgraded it
                    final_status = elem_status
                    if comp_res.status == "pseudo" and best_tp_score < 30.0:
                        final_status = "pseudo"

                    elem_len = pred_end - pred_start
                    is_dna = c_seq[pred_start:pred_end]

                    elem = MetagenomeISElement(
                        element_id=f"DeepISE_meta_{global_elem_idx:05d}",
                        contig_id=c_id,
                        start=pred_start,
                        end=pred_end,
                        length=elem_len,
                        contig_length=c_len,
                        strand=c_strand,
                        family=c_fam,
                        tpase_gene_id=primary_gene_id,
                        tpase_score=best_tp_score,
                        tpase_evalue=min_tp_evalue,
                        composite_score=comp_res.composite_score,
                        status=final_status,
                        truncation_status=trunc_status,
                        method=self.boundary_mode,
                        tir_length=b_pred.tir.length if b_pred.tir else None,
                        tir_identity=b_pred.tir.identity if b_pred.tir else None,
                        tsd_length=b_pred.tsd.length if b_pred.tsd else None,
                        tsd_sequence=b_pred.tsd.sequence if b_pred.tsd else None,
                        structural_evidence=b_pred.structure.feature_type if b_pred.structure else None,
                        notes=f"{notes}; {comp_res.notes}".strip("; "),
                        dna_sequence=is_dna,
                        protein_sequence=tpase_protein,
                    )
                    batch_detected.append(elem)

            # 4. Redundancy filtering per contig
            batch_detected.sort(key=lambda x: (x.contig_id, x.start))
            filtered: List[MetagenomeISElement] = []
            for e in batch_detected:
                if not filtered:
                    filtered.append(e)
                else:
                    last = filtered[-1]
                    overlap = min(e.end, last.end) - max(e.start, last.start)
                    if e.contig_id == last.contig_id and overlap > 0.5 * min(e.length, last.length):
                        if e.composite_score > last.composite_score:
                            filtered[-1] = e
                    else:
                        filtered.append(e)

            return filtered

        # Stream records
        for record in SeqIO.parse(fasta_path, "fasta"):
            stats["total_contigs_read"] += 1
            seq_len = len(record.seq)

            if seq_len < self.min_contig_len:
                stats["total_contigs_filtered_short"] += 1
                continue

            stats["total_contigs_analyzed"] += 1
            stats["total_bp_scanned"] += seq_len
            contig_idx += 1

            batch_contigs.append((
                contig_idx,
                str(record.id),
                str(record.seq).upper(),
                bytes(record.seq),
            ))

            if len(batch_contigs) >= self.batch_size:
                batch_elements = process_batch(batch_contigs)
                all_detected.extend(batch_elements)
                batch_contigs = []

                if progress_callback is not None:
                    progress_callback({
                        "contigs_analyzed": stats["total_contigs_analyzed"],
                        "bp_scanned": stats["total_bp_scanned"],
                        "is_detected": len(all_detected),
                    })

        # Process remaining contigs in last batch
        if batch_contigs:
            batch_elements = process_batch(batch_contigs)
            all_detected.extend(batch_elements)
            batch_contigs = []

        t_end = time.perf_counter()
        elapsed = t_end - t_start

        # Compute summary statistics
        stats["total_is_elements_detected"] = len(all_detected)
        for e in all_detected:
            stats["breakdown_by_status"][e.status] = stats["breakdown_by_status"].get(e.status, 0) + 1
            stats["breakdown_by_truncation"][e.truncation_status] = (
                stats["breakdown_by_truncation"].get(e.truncation_status, 0) + 1
            )
            stats["breakdown_by_family"][e.family] = stats["breakdown_by_family"].get(e.family, 0) + 1

        stats["runtime_seconds"] = round(elapsed, 3)
        stats["throughput_mbp_per_sec"] = (
            round((stats["total_bp_scanned"] / 1_000_000.0) / max(0.001, elapsed), 2)
        )
        stats["throughput_contigs_per_sec"] = (
            round(stats["total_contigs_analyzed"] / max(0.001, elapsed), 2)
        )

        return all_detected, stats


def export_metagenome_results(
    elements: List[MetagenomeISElement],
    stats: Dict[str, Any],
    out_dir: Path,
) -> Tuple[Path, Path, Path, Path, Path]:
    """Export metagenome IS detection results to standard production formats."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gff_path = out_dir / "deepise_is_elements.gff3"
    tsv_path = out_dir / "deepise_is_elements.tsv"
    fna_path = out_dir / "deepise_is_elements.fna"
    faa_path = out_dir / "deepise_tpases.faa"
    json_path = out_dir / "deepise_summary.json"

    # 1. GFF3
    gff_lines = ["##gff-version 3"]
    for e in elements:
        attrs = [
            f"ID={e.element_id}",
            f"Name={e.element_id}",
            f"Family={e.family}",
            f"Status={e.status}",
            f"Truncation={e.truncation_status}",
            f"Score={e.composite_score:.3f}",
            f"Contig_Length={e.contig_length}",
            f"Method={e.method}",
        ]
        if e.tir_length:
            attrs.append(f"TIR_len={e.tir_length}")
        if e.tsd_sequence:
            attrs.append(f"TSD={e.tsd_sequence}")
        if e.structural_evidence:
            attrs.append(f"Structure={e.structural_evidence}")
        attr_str = ";".join(attrs)
        # GFF coordinates are 1-based, inclusive
        gff_lines.append(
            f"{e.contig_id}\tDeepISE\tinsertion_sequence\t{e.start + 1}\t{e.end}\t{e.composite_score:.2f}\t{e.strand}\t.\t{attr_str}"
        )
    gff_path.write_text("\n".join(gff_lines) + "\n")

    # 2. TSV
    rows = []
    for e in elements:
        d = e.model_dump()
        # Drop raw sequences from TSV to keep it lightweight
        d.pop("dna_sequence", None)
        d.pop("protein_sequence", None)
        rows.append(d)
    df = pl.DataFrame(rows) if rows else pl.DataFrame()
    df.write_csv(tsv_path, separator="\t")

    # 3. FNA (IS element DNA)
    fna_lines = []
    for e in elements:
        header = (
            f">{e.element_id} {e.contig_id}:{e.start+1}-{e.end} "
            f"family={e.family} status={e.status} truncation={e.truncation_status} "
            f"score={e.composite_score:.3f} length={e.length}"
        )
        fna_lines.append(header)
        seq = e.dna_sequence
        for i in range(0, len(seq), 80):
            fna_lines.append(seq[i : i + 80])
    fna_path.write_text("\n".join(fna_lines) + "\n")

    # 4. FAA (Transposase protein)
    faa_lines = []
    for e in elements:
        if e.protein_sequence:
            header = (
                f">{e.element_id}_tpase gene={e.tpase_gene_id} "
                f"contig={e.contig_id} family={e.family} score={e.tpase_score:.1f}"
            )
            faa_lines.append(header)
            seq = e.protein_sequence
            for i in range(0, len(seq), 80):
                faa_lines.append(seq[i : i + 80])
    faa_path.write_text("\n".join(faa_lines) + "\n")

    # 5. JSON Summary
    json_path.write_text(json.dumps(stats, indent=2) + "\n")

    return gff_path, tsv_path, fna_path, faa_path, json_path


def scan_metagenome_file(
    fasta_path: Path,
    out_dir: Path,
    min_contig_len: int = 500,
    min_bitscore: float = 10.6,
    batch_size: int = 1000,
    boundary_mode: str = "plan_a",
    threads: int = 4,
) -> Tuple[List[MetagenomeISElement], Dict[str, Any]]:
    """Convenience wrapper for scanning a metagenome FASTA and exporting all artifacts."""
    scanner = MetagenomeScanner(
        min_bitscore=min_bitscore,
        min_contig_len=min_contig_len,
        batch_size=batch_size,
        boundary_mode=boundary_mode,
    )
    elements, stats = scanner.scan(fasta_path=fasta_path, threads=threads)
    export_metagenome_results(elements=elements, stats=stats, out_dir=out_dir)
    return elements, stats
