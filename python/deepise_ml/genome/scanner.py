"""End-to-end bacterial genome scanner for full-length IS element discovery."""

import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl
import pyrodigal
from Bio import SeqIO
from pydantic import BaseModel, Field

from deepise_ml.boundary.hybrid import HybridBoundaryEngine
from deepise_ml.boundary.plan_a import PlanAAdaptiveEngine
from deepise_ml.boundary.plan_b import PlanBCanonicalEngine
from deepise_ml.boundary.schemas import BoundaryPrediction
from deepise_ml.composite.scorer import ISCompositeResult, ISCompositeScorer
from deepise_ml.models.hmmer import parse_hmmsearch_predictions


class DetectedISElement(BaseModel):
    """Full-length IS element detected in a genome."""
    element_id: str
    contig_id: str
    start: int = Field(..., description="0-indexed start coordinate in contig")
    end: int = Field(..., description="0-indexed end coordinate in contig (exclusive)")
    length: int
    strand: str
    family: str
    tpase_gene_id: str
    tpase_score: float
    tpase_evalue: float
    composite_score: float
    status: str = Field(..., description="'complete', 'partial', or 'pseudo'")
    method: str
    tir_length: Optional[int] = None
    tir_identity: Optional[float] = None
    tsd_length: Optional[int] = None
    tsd_sequence: Optional[str] = None
    structural_evidence: Optional[str] = None
    protein_sequence: Optional[str] = None
    notes: str = ""


class DeepISEGenomeScanner:
    """Full genome IS element detection and annotation engine."""

    def __init__(
        self,
        hmm_path: Path = Path("benchmark/db/deepise_tpases.hmm"),
        min_bitscore: float = 10.6,
        hmmer_bin: str = "/hpcfs/fhome/caizhh/.conda/envs/DeepISE/bin/hmmsearch",
    ):
        self.hmm_path = hmm_path
        self.min_bitscore = min_bitscore
        self.hmmer_bin = hmmer_bin
        self.engine_a = PlanAAdaptiveEngine()
        self.engine_b = PlanBCanonicalEngine()
        self.engine_hybrid = HybridBoundaryEngine()
        self.scorer = ISCompositeScorer()

    def scan_genome(
        self,
        fasta_path: Path,
        mode: str = "plan_a",
        threads: int = 4,
    ) -> List[DetectedISElement]:
        """Scan a genome FASTA file and return all detected IS elements."""
        records = list(SeqIO.parse(fasta_path, "fasta"))
        if not records:
            return []

        all_detected: List[DetectedISElement] = []
        elem_idx = 0

        for record in records:
            contig_id = record.id
            contig_seq = str(record.seq).upper()
            contig_bytes = bytes(record.seq)

            # Step 1: Predict ORFs with Pyrodigal
            gf = pyrodigal.GeneFinder(meta=False)
            gf.train(contig_bytes)
            genes = gf.find_genes(contig_bytes)

            if len(genes) == 0:
                continue

            # Step 2: Screen ORFs with HMMER
            with tempfile.NamedTemporaryFile(suffix=".faa", mode="w") as faa_f, \
                 tempfile.NamedTemporaryFile(suffix=".tblout") as tbl_f:
                genes.write_translations(faa_f, sequence_id=contig_id)
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
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, env=env)
                preds = parse_hmmsearch_predictions(Path(tbl_f.name), Path(faa_f.name))

            # Filter significant transposase hits
            tp_hits = preds.filter(pl.col("score") >= self.min_bitscore)
            if len(tp_hits) == 0:
                continue

            # Map seq_id -> HMM hit info (best scoring per ORF)
            tp_map: Dict[str, Tuple[str, float, float]] = {}
            for row in tp_hits.iter_rows(named=True):
                s_id = row["seq_id"]
                target = row["target_id"]  # e.g. Tpase_IS4
                family = target.replace("Tpase_", "")
                score = float(row["score"])
                evalue = float(row["evalue"])
                if s_id not in tp_map or score > tp_map[s_id][1]:
                    tp_map[s_id] = (family, score, evalue)

            # Step 3: Cluster adjacent transposases belonging to the same element (e.g. IS1 InsA/InsB)
            hit_genes = []
            for i, g in enumerate(genes, start=1):
                gene_id = f"{contig_id}_{i}"
                if gene_id in tp_map:
                    fam, sc, ev = tp_map[gene_id]
                    hit_genes.append({
                        "gene_id": gene_id,
                        "begin": g.begin - 1,  # 0-indexed
                        "end": g.end,
                        "strand": "+" if g.strand == 1 else "-",
                        "family": fam,
                        "score": sc,
                        "evalue": ev,
                        "translation": g.translate(),
                    })

            # Group overlapping / tightly linked ORFs (< 250 bp apart and same family)
            clusters: List[List[Dict]] = []
            for g_info in hit_genes:
                if not clusters:
                    clusters.append([g_info])
                else:
                    prev = clusters[-1][-1]
                    if g_info["begin"] - prev["end"] <= 250 and g_info["family"] == prev["family"]:
                        clusters[-1].append(g_info)
                    else:
                        clusters.append([g_info])

            # Step 4: Resolve boundaries for each transposase cluster
            for cluster in clusters:
                elem_idx += 1
                c_start = cluster[0]["begin"]
                c_end = cluster[-1]["end"]
                c_strand = cluster[0]["strand"]
                c_fam = cluster[0]["family"]
                primary_gene = max(cluster, key=lambda c: c["score"])
                best_tp_score = primary_gene["score"]
                min_tp_evalue = min(c["evalue"] for c in cluster)
                primary_gene_id = primary_gene["gene_id"]
                tpase_seq = primary_gene.get("translation")

                # Boundary prediction
                if mode in ["hybrid", "plan_a_hybrid"]:
                    b_pred = self.engine_hybrid.predict_boundary(
                        contig=contig_seq,
                        tpase_start=c_start,
                        tpase_end=c_end,
                        contig_id=contig_id,
                        is_name=f"{c_fam}_{elem_idx}",
                        family=c_fam,
                    )
                elif mode == "plan_a":
                    b_pred = self.engine_a.predict_boundary(
                        contig=contig_seq,
                        tpase_start=c_start,
                        tpase_end=c_end,
                        contig_id=contig_id,
                        is_name=f"{c_fam}_{elem_idx}",
                        family=c_fam,
                    )
                else:
                    b_pred = self.engine_b.predict_boundary(
                        contig=contig_seq,
                        tpase_start=c_start,
                        tpase_end=c_end,
                        contig_id=contig_id,
                        is_name=f"{c_fam}_{elem_idx}",
                        family=c_fam,
                    )

                # Composite scoring
                comp_res = self.scorer.score_element(
                    boundary_pred=b_pred,
                    tpase_score=min(1.0, best_tp_score / 150.0),
                    tpase_length_bp=c_end - c_start,
                )

                tir_len = b_pred.tir.length if b_pred.tir else None
                tir_id = b_pred.tir.identity if b_pred.tir else None
                tsd_len = b_pred.tsd.length if b_pred.tsd else None
                tsd_seq = b_pred.tsd.sequence if b_pred.tsd else None
                struct_ev = b_pred.structure.feature_type if b_pred.structure else None

                elem = DetectedISElement(
                    element_id=f"DeepISE_{contig_id}_{elem_idx:04d}",
                    contig_id=contig_id,
                    start=b_pred.predicted_start,
                    end=b_pred.predicted_end,
                    length=b_pred.predicted_length,
                    strand=c_strand,
                    family=c_fam,
                    tpase_gene_id=primary_gene_id,
                    tpase_score=best_tp_score,
                    tpase_evalue=min_tp_evalue,
                    composite_score=comp_res.composite_score,
                    status=comp_res.status,
                    method=mode,
                    tir_length=tir_len,
                    tir_identity=tir_id,
                    tsd_length=tsd_len,
                    tsd_sequence=tsd_seq,
                    structural_evidence=struct_ev,
                    protein_sequence=tpase_seq,
                    notes=comp_res.notes,
                )
                all_detected.append(elem)

        # Step 5: Filter overlapping predictions (keep higher composite score)
        all_detected.sort(key=lambda x: (x.contig_id, x.start))
        non_redundant: List[DetectedISElement] = []
        for e in all_detected:
            if not non_redundant:
                non_redundant.append(e)
            else:
                last = non_redundant[-1]
                overlap = min(e.end, last.end) - max(e.start, last.start)
                if e.contig_id == last.contig_id and overlap > 0.5 * min(e.length, last.length):
                    if e.composite_score > last.composite_score:
                        non_redundant[-1] = e
                else:
                    non_redundant.append(e)

        return non_redundant


def export_genome_results(
    elements: List[DetectedISElement],
    contigs_dict: Dict[str, str],
    out_dir: Path,
) -> Tuple[Path, Path, Path]:
    """Export detected IS elements to GFF3, TSV, and FASTA."""
    out_dir.mkdir(parents=True, exist_ok=True)
    gff_path = out_dir / "deepise_is_elements.gff3"
    tsv_path = out_dir / "deepise_is_elements.tsv"
    fna_path = out_dir / "deepise_is_elements.fna"

    # 1. GFF3
    gff_lines = ["##gff-version 3"]
    for e in elements:
        attrs = [
            f"ID={e.element_id}",
            f"Family={e.family}",
            f"Status={e.status}",
            f"Score={e.composite_score:.3f}",
            f"Method={e.method}",
        ]
        if e.tir_length:
            attrs.append(f"TIR_len={e.tir_length}")
        if e.tsd_sequence:
            attrs.append(f"TSD={e.tsd_sequence}")
        attr_str = ";".join(attrs)
        # GFF coordinates are 1-based, inclusive
        gff_lines.append(
            f"{e.contig_id}\tDeepISE\tinsertion_sequence\t{e.start + 1}\t{e.end}\t{e.composite_score:.2f}\t{e.strand}\t.\t{attr_str}"
        )
    gff_path.write_text("\n".join(gff_lines) + "\n")

    # 2. TSV
    rows = [e.model_dump() for e in elements]
    df = pl.DataFrame(rows) if rows else pl.DataFrame()
    df.write_csv(tsv_path, separator="\t")

    # 3. FASTA
    fna_lines = []
    for e in elements:
        contig_s = contigs_dict.get(e.contig_id, "")
        elem_dna = contig_s[e.start : e.end]
        header = f">{e.element_id} {e.contig_id}:{e.start+1}-{e.end} family={e.family} status={e.status} score={e.composite_score}"
        fna_lines.append(header)
        for i in range(0, len(elem_dna), 80):
            fna_lines.append(elem_dna[i : i + 80])
    fna_path.write_text("\n".join(fna_lines) + "\n")

    # 4. FAA (Transposase protein)
    faa_path = out_dir / "deepise_tpases.faa"
    faa_lines = []
    for e in elements:
        if e.protein_sequence:
            header = f">{e.element_id}_tpase gene={e.tpase_gene_id} contig={e.contig_id} family={e.family} score={e.tpase_score:.1f}"
            faa_lines.append(header)
            seq = e.protein_sequence
            for i in range(0, len(seq), 80):
                faa_lines.append(seq[i : i + 80])
    if faa_lines:
        faa_path.write_text("\n".join(faa_lines) + "\n")

    return gff_path, tsv_path, fna_path
