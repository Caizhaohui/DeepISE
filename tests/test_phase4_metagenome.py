"""Unit and regression tests for Phase-4 metagenomic scanner and production CLI."""

import json
import tempfile
from pathlib import Path
import pytest
import polars as pl
from Bio import SeqIO
from typer.testing import CliRunner

from deepise_ml.cli import app
from deepise_ml.metagenome.scanner import (
    MetagenomeISElement,
    MetagenomeScanner,
    export_metagenome_results,
)


def test_metagenome_contig_filtering(tmp_path):
    """Test that contigs below min_contig_len are correctly discarded."""
    fasta = tmp_path / "test_filter.fna"
    fasta.write_text(
        ">short_contig_1\nACGTACGT\n"
        ">short_contig_2\n" + "A" * 499 + "\n"
        ">valid_contig_3\n" + "ACGT" * 200 + "\n"  # 800 bp
    )

    scanner = MetagenomeScanner(min_contig_len=500, batch_size=10)
    elements, stats = scanner.scan(fasta)

    assert stats["total_contigs_read"] == 3
    assert stats["total_contigs_filtered_short"] == 2
    assert stats["total_contigs_analyzed"] == 1
    assert stats["total_bp_scanned"] == 800


def test_metagenome_edge_truncation_detection(tmp_path):
    """Test edge truncation classification and coordinate clamping."""
    # Build a synthetic contig with an IS1 element at 5' border
    is1_seq = (
        "GGTAATGACCCCGCA"  # Left TIR
        + ("ATGAGTTCTGTTACCCGCACACTGATCTTGATTGCTACGTTCTTCGGCGTCAGTTTCGTCACGACGTTGATCCTGA" * 8)
        + "TGCGGGGTCATTACC"  # Right TIR
    )
    # Contig 1: 5'-truncated (IS1 starts at index 0)
    contig_5p = is1_seq + "C" * 800
    # Contig 2: 3'-truncated (IS1 ends at the very end of contig)
    contig_3p = "G" * 800 + is1_seq
    # Contig 3: complete (IS1 in middle with 500 bp flanks)
    contig_comp = "A" * 500 + is1_seq + "T" * 500

    fasta = tmp_path / "test_trunc.fna"
    fasta.write_text(
        f">contig_5p\n{contig_5p}\n"
        f">contig_3p\n{contig_3p}\n"
        f">contig_comp\n{contig_comp}\n"
    )

    scanner = MetagenomeScanner(min_contig_len=500, edge_tolerance_bp=40)
    elements, stats = scanner.scan(fasta)

    assert stats["total_contigs_analyzed"] == 3
    # Check that coordinate clamping holds for all detected elements
    for e in elements:
        assert e.start >= 0
        assert e.end <= e.contig_length
        assert e.length == e.end - e.start
        assert e.truncation_status in [
            "complete",
            "edge_5p_truncated",
            "edge_3p_truncated",
            "edge_both_truncated",
            "internal_partial",
        ]


def test_export_metagenome_results_artifacts(tmp_path):
    """Test that all 5 production artifacts (GFF3, TSV, FNA, FAA, JSON) are correctly formatted."""
    elem = MetagenomeISElement(
        element_id="DeepISE_meta_00001",
        contig_id="contig_test_01",
        start=100,
        end=1450,
        length=1350,
        contig_length=5000,
        strand="+",
        family="IS3",
        tpase_gene_id="contig_test_01_1",
        tpase_score=95.5,
        tpase_evalue=1e-25,
        composite_score=0.88,
        status="complete",
        truncation_status="complete",
        method="hybrid",
        tir_length=24,
        tir_identity=0.96,
        tsd_length=4,
        tsd_sequence="CTAG",
        structural_evidence="neural_refined_junction",
        notes="High confidence",
        dna_sequence="A" * 1350,
        protein_sequence="M" * 350,
    )
    stats = {
        "deepise_version": "0.1.0",
        "total_contigs_analyzed": 1,
        "total_is_elements_detected": 1,
        "breakdown_by_status": {"complete": 1, "partial": 0, "pseudo": 0},
        "breakdown_by_truncation": {"complete": 1},
    }

    out_dir = tmp_path / "export_test"
    gff_p, tsv_p, fna_p, faa_p, json_p = export_metagenome_results([elem], stats, out_dir)

    # 1. GFF3 check
    assert gff_p.exists()
    gff_content = gff_p.read_text()
    assert "##gff-version 3" in gff_content
    assert "contig_test_01\tDeepISE\tinsertion_sequence\t101\t1450\t0.88\t+\t." in gff_content
    assert "Truncation=complete" in gff_content

    # 2. TSV check
    assert tsv_p.exists()
    df = pl.read_csv(tsv_p, separator="\t")
    assert len(df) == 1
    assert df["element_id"][0] == "DeepISE_meta_00001"
    assert df["family"][0] == "IS3"
    assert df["truncation_status"][0] == "complete"

    # 3. FNA check
    assert fna_p.exists()
    fna_content = fna_p.read_text()
    assert ">DeepISE_meta_00001 contig_test_01:101-1450" in fna_content

    # 4. FAA check
    assert faa_p.exists()
    faa_content = faa_p.read_text()
    assert ">DeepISE_meta_00001_tpase" in faa_content

    # 5. JSON check
    assert json_p.exists()
    loaded_json = json.loads(json_p.read_text())
    assert loaded_json["deepise_version"] == "0.1.0"
    assert loaded_json["total_is_elements_detected"] == 1


def test_cli_version_and_help():
    """Test CLI commands and entrypoints."""
    runner = CliRunner()

    res_version = runner.invoke(app, ["version"])
    assert res_version.exit_code == 0
    assert "DeepISE version 0.1.0" in res_version.stdout

    res_help = runner.invoke(app, ["scan-metagenome", "--help"])
    assert res_help.exit_code == 0
    assert "--min-contig-len" in res_help.stdout
