"""CLI interface for DeepISE dataset preparation and benchmarking."""

import os
import sys
from pathlib import Path
from typing import Optional
import polars as pl
import typer
import yaml
from rich.console import Console

from deepise_ml.dataset.isfinder import (
    load_isfinder_metadata,
    extract_tpases,
    extract_is_elements,
)
from deepise_ml.dataset.deduplicate import (
    deduplicate_tpases,
    deduplicate_is_elements,
    export_fasta,
    export_dna_fasta,
)
from deepise_ml.dataset.cluster import (
    run_mmseqs_clustering,
    parse_cluster_tsv,
    compute_cluster_statistics,
)
from deepise_ml.dataset.split import cluster_aware_split, save_splits
from deepise_ml.dataset.leakage import (
    run_mmseqs_leakage_search,
    audit_leakage,
    generate_leakage_report_markdown,
    LeakageViolationError,
)
from deepise_ml.dataset.negatives import (
    stream_sprot_candidates,
    sample_negatives,
    export_negatives,
)
from deepise_ml.dataset.negative_split import build_combined_splits
from deepise_ml.models.blast import build_blast_database, run_blastp, parse_blast_predictions
from deepise_ml.models.mmseqs import run_mmseqs_search, parse_mmseqs_predictions
from deepise_ml.models.hmmer import build_family_hmms, run_hmmsearch, parse_hmmsearch_predictions
from deepise_ml.benchmark.evaluation import (
    compute_overall_metrics,
    compute_identity_stratified_recall,
    compute_family_recall,
    bootstrap_cluster_ci,
)
from deepise_ml.benchmark.report import (
    export_overall_metrics_table,
    export_stratified_metrics_table,
    export_family_metrics_table,
    generate_phase1_benchmark_report,
    generate_go_no_go_report,
)
from deepise_ml.boundary.benchmark_dataset import build_benchmark_contigs
from deepise_ml.boundary.plan_a import PlanAAdaptiveEngine
from deepise_ml.boundary.plan_b import PlanBCanonicalEngine
from deepise_ml.boundary.evaluator import evaluate_engine_on_benchmark
from deepise_ml.boundary.report import export_phase2_comparison_tables, generate_phase2_report_markdown
from deepise_ml.genome.scanner import DeepISEGenomeScanner, export_genome_results
from deepise_ml.genome.benchmark_genome import run_full_genome_benchmark
from deepise_ml.metagenome.scanner import (
    MetagenomeScanner,
    export_metagenome_results,
    scan_metagenome_file,
)
from deepise_ml.metagenome.benchmark import (
    build_synthetic_metagenome,
    run_full_metagenome_benchmark,
)
from deepise_ml.models.plm_registry import PluggablePLMEngine
from deepise_ml.screening.closed_loop import perform_closed_loop_research_validation
from deepise_ml.screening.evidence import HomologyContractError
from deepise_ml.screening.homology import HomologyAdapterError
from deepise_ml.screening.pipeline import (
    HomologyScreeningPaths,
    ProteinHomologyScreeningService,
)
from deepise_ml.screening.research_service import ProteinResearchScreeningService
from deepise_ml.screening.runtime import (
    ProteinScreeningError,
    ProteinScreeningService,
    load_classifier,
    read_protein_fasta,
    resolve_profile,
)

app = typer.Typer(help="DeepISE Data Management CLI")
console = Console()


@app.command()
def normalize(
    raw_dir: Path = typer.Option(Path("data/raw/isfinder/2026-09-10"), help="Raw ISfinder snapshot directory"),
    output_tpase_parquet: Path = typer.Option(Path("data/interim/tpases_normalized.parquet"), help="Output interim tpase parquet"),
    output_is_parquet: Path = typer.Option(Path("data/interim/is_elements_normalized.parquet"), help="Output interim IS element parquet"),
):
    """Normalize raw ISfinder snapshot files into canonical parquet format."""
    console.print(f"[bold blue]Step 1: Normalizing ISfinder snapshot from {raw_dir}...[/bold blue]")
    csv_path = raw_dir / "IS.csv"
    faa_path = raw_dir / "IS.faa"
    fna_path = raw_dir / "IS.fna"

    assert csv_path.exists(), f"Missing {csv_path}"
    assert faa_path.exists(), f"Missing {faa_path}"
    assert fna_path.exists(), f"Missing {fna_path}"

    meta = load_isfinder_metadata(csv_path)
    console.print(f"Loaded metadata for {len(meta)} IS elements.")

    tpases = extract_tpases(faa_path, meta, confirmed_only=True)
    console.print(f"Extracted {len(tpases)} confirmed Transposase proteins.")

    elements = extract_is_elements(fna_path, meta)
    console.print(f"Extracted {len(elements)} nucleotide IS elements.")

    output_tpase_parquet.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([t.model_dump() for t in tpases]).write_parquet(output_tpase_parquet)
    pl.DataFrame([e.model_dump() for e in elements]).write_parquet(output_is_parquet)

    console.print(f"[green]Normalized records saved to {output_tpase_parquet} and {output_is_parquet}[/green]")


@app.command()
def deduplicate(
    input_tpase_parquet: Path = typer.Option(Path("data/interim/tpases_normalized.parquet"), help="Interim tpase parquet"),
    input_is_parquet: Path = typer.Option(Path("data/interim/is_elements_normalized.parquet"), help="Interim IS parquet"),
    output_processed_tpase: Path = typer.Option(Path("data/processed/tpases.parquet"), help="Processed tpases parquet"),
    output_tpase_faa: Path = typer.Option(Path("data/processed/tpases.faa"), help="Processed tpases FASTA"),
    output_processed_is: Path = typer.Option(Path("data/processed/is_elements.parquet"), help="Processed IS parquet"),
    output_is_fna: Path = typer.Option(Path("data/processed/is_elements.fna"), help="Processed IS FASTA"),
):
    """Perform 100% exact sequence deduplication."""
    console.print("[bold blue]Step 2: Performing 100% exact deduplication...[/bold blue]")
    from deepise_ml.schemas.records import CanonicalTpaseRecord, CanonicalISRecord

    tpase_df = pl.read_parquet(input_tpase_parquet)
    tpase_recs = [CanonicalTpaseRecord(**row) for row in tpase_df.iter_rows(named=True)]

    rep_tpases, tpase_cluster_df = deduplicate_tpases(tpase_recs)
    console.print(f"Tpase exact dedup: {len(tpase_recs)} -> {len(rep_tpases)} unique sequence representatives.")

    # Save exact clusters
    exact_clu_path = Path("data/interim/tpase_exact_clusters.parquet")
    exact_clu_path.parent.mkdir(parents=True, exist_ok=True)
    tpase_cluster_df.write_parquet(exact_clu_path)

    # Export processed tpases
    rep_df = pl.DataFrame([r.model_dump() for r in rep_tpases])
    rep_df.write_parquet(output_processed_tpase)
    export_fasta(rep_tpases, output_tpase_faa)
    console.print(f"Exported {len(rep_tpases)} deduplicated tpases to {output_processed_tpase} and {output_tpase_faa}")

    # IS elements
    is_df = pl.read_parquet(input_is_parquet)
    is_recs = [CanonicalISRecord(**row) for row in is_df.iter_rows(named=True)]
    rep_elements, is_cluster_df = deduplicate_is_elements(is_recs)
    console.print(f"IS element exact dedup: {len(is_recs)} -> {len(rep_elements)} unique DNA representatives.")

    is_exact_path = Path("data/interim/is_element_exact_clusters.parquet")
    is_cluster_df.write_parquet(is_exact_path)
    pl.DataFrame([r.model_dump() for r in rep_elements]).write_parquet(output_processed_is)
    export_dna_fasta(rep_elements, output_is_fna)
    console.print(f"[green]Deduplication complete.[/green]")


@app.command()
def cluster(
    input_faa: Path = typer.Option(Path("data/processed/tpases.faa"), help="Processed tpases FASTA"),
    processed_parquet: Path = typer.Option(Path("data/processed/tpases.parquet"), help="Processed tpases Parquet"),
    output_tsv: Path = typer.Option(Path("data/interim/tpase_cluster30.tsv"), help="Output cluster TSV"),
    identity: float = typer.Option(0.30, help="Minimum sequence identity threshold"),
    coverage: float = typer.Option(0.80, help="Minimum coverage threshold"),
    threads: int = typer.Option(8, help="Number of CPU threads"),
):
    """Run MMseqs2 30% sequence identity clustering and annotate Parquet dataset."""
    console.print(f"[bold blue]Step 3: Running MMseqs2 clustering (id={identity}, cov={coverage})...[/bold blue]")
    tmp_dir = Path("data/interim/mmseqs_tpase")
    run_mmseqs_clustering(
        fasta_path=input_faa,
        output_tsv=output_tsv,
        tmp_dir=tmp_dir,
        min_seq_id=identity,
        coverage=coverage,
        threads=threads,
    )

    member_to_cluster, cluster_df = parse_cluster_tsv(output_tsv, prefix="cluster30_")
    console.print(f"Parsed {len(cluster_df['cluster_id'].unique())} distinct 30% clusters.")

    # Annotate processed tpases parquet with cluster30_id
    tpase_df = pl.read_parquet(processed_parquet)
    annotated_rows = []
    for row in tpase_df.iter_rows(named=True):
        t_id = row["tpase_id"]
        row["cluster30_id"] = member_to_cluster.get(t_id)
        annotated_rows.append(row)

    annotated_df = pl.DataFrame(annotated_rows)
    annotated_df.write_parquet(processed_parquet)
    console.print(f"[green]Cluster IDs mapped back into {processed_parquet}[/green]")


@app.command()
def split(
    processed_parquet: Path = typer.Option(Path("data/processed/tpases.parquet"), help="Processed tpases Parquet"),
    output_dir: Path = typer.Option(Path("data/splits/cluster30"), help="Output splits directory"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Perform cluster-aware 70/15/15 dataset splitting."""
    console.print(f"[bold blue]Step 4: Performing cluster-aware split (seed={seed})...[/bold blue]")
    tpase_df = pl.read_parquet(processed_parquet)
    train_df, val_df, test_df, manifest = cluster_aware_split(tpase_df, ratios=(0.70, 0.15, 0.15), seed=seed)

    save_splits(train_df, val_df, test_df, manifest, output_dir)
    console.print(f"[green]Splits saved: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}[/green]")


@app.command()
def audit(
    split_dir: Path = typer.Option(Path("data/splits/cluster30"), help="Split directory containing train.faa and test.faa"),
    report_output: Path = typer.Option(Path("benchmark/reports/leakage_report.md"), help="Markdown report output"),
    identity_threshold: float = 0.30,
    coverage_threshold: float = 0.80,
    threads: int = 8,
):
    """Run all-vs-all MMseqs2 search to audit test vs train homology leakage."""
    console.print("[bold blue]Step 5: Running all-vs-all homology leakage audit...[/bold blue]")
    test_faa = split_dir / "test.faa"
    train_faa = split_dir / "train.faa"
    search_tsv = split_dir / "test_vs_train.tsv"
    tmp_dir = split_dir / "leakage_tmp"

    run_mmseqs_leakage_search(test_faa, train_faa, search_tsv, tmp_dir, threads=threads)
    hits_df, violations_df, summary = audit_leakage(
        search_tsv=search_tsv,
        query_faa=test_faa,
        identity_threshold=identity_threshold,
        coverage_threshold=coverage_threshold,
    )

    generate_leakage_report_markdown(summary, violations_df, report_output)

    if not summary["pass_audit"]:
        console.print(f"[bold red]LEAKAGE AUDIT FAILED! {summary['total_violations']} violations detected.[/bold red]")
        raise LeakageViolationError(f"Found {summary['total_violations']} test-train pairs with identity > 0.30 and coverage >= 0.80")

    console.print(f"[bold green]LEAKAGE AUDIT PASSED! 0 violations detected.[/bold green]")
    console.print(f"Report saved to {report_output}")


@app.command()
def build_negatives(
    config_path: Path = typer.Option(Path("config/negatives.yaml"), help="Negative sampling config"),
    positives_parquet: Path = typer.Option(Path("data/processed/tpases.parquet"), help="Positives Parquet file"),
    output_parquet: Path = typer.Option(Path("data/processed/negatives.parquet"), help="Output negatives Parquet"),
    output_faa: Path = typer.Option(Path("data/processed/negatives.faa"), help="Output negatives FASTA"),
):
    """Sample stratified multi-tiered negatives from Swiss-Prot database."""
    console.print(f"[bold blue]Step 6: Building negative dataset from Swiss-Prot...[/bold blue]")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sprot_gz = Path(cfg["source_sprot_fasta"])
    assert sprot_gz.exists(), f"Missing Swiss-Prot FASTA at {sprot_gz}"

    pos_df = pl.read_parquet(positives_parquet)
    pos_lengths = pos_df["protein_length"].to_list()
    target_count = int(len(pos_df) * cfg.get("ratio_to_positive", 3.0))

    console.print(f"Streaming Swiss-Prot to collect negative candidate pools (target={target_count})...")
    hard_pool, easy_pool, general_pool = stream_sprot_candidates(
        sprot_fasta_gz=sprot_gz,
        exclude_keywords=cfg.get("exclude_keywords", []),
        hard_keywords=cfg.get("hard_negative_keywords", []),
    )
    console.print(f"Collected candidates: Hard={len(hard_pool)}, Easy={len(easy_pool)}, General={len(general_pool)}")

    class_ratios = cfg.get("class_ratios", {"hard": 0.50, "matched": 0.30, "easy": 0.20})
    neg_records = sample_negatives(
        hard_pool=hard_pool,
        easy_pool=easy_pool,
        general_pool=general_pool,
        positive_lengths=pos_lengths,
        target_count=target_count,
        hard_ratio=class_ratios.get("hard", 0.50),
        matched_ratio=class_ratios.get("matched", 0.30),
        easy_ratio=class_ratios.get("easy", 0.20),
    )

    export_negatives(neg_records, output_parquet, output_faa)
    console.print(f"[green]Saved {len(neg_records)} negative proteins to {output_parquet} and {output_faa}[/green]")


@app.command()
def summary(
    tpases_parquet: Path = typer.Option(Path("data/processed/tpases.parquet"), help="Processed tpases Parquet"),
    cluster_tsv: Path = typer.Option(Path("data/interim/tpase_cluster30.tsv"), help="Cluster30 TSV"),
    negatives_parquet: Path = typer.Option(Path("data/processed/negatives.parquet"), help="Negatives Parquet"),
    split_dir: Path = typer.Option(Path("data/splits/cluster30"), help="Splits directory"),
    output_summary_md: Path = typer.Option(Path("benchmark/reports/dataset_summary.md"), help="Dataset summary report"),
    output_cluster_md: Path = typer.Option(Path("benchmark/reports/cluster_statistics.md"), help="Cluster statistics report"),
):
    """Generate Phase-0 markdown summary reports."""
    console.print("[bold blue]Step 7: Generating Phase-0 Summary Reports...[/bold blue]")
    tpase_df = pl.read_parquet(tpases_parquet)
    _, cluster_df = parse_cluster_tsv(cluster_tsv)
    stats = compute_cluster_statistics(cluster_df, tpase_df)

    # 1. Cluster statistics report
    cluster_lines = [
        "# DeepISE Cluster Statistics (MMseqs2 30% Identity, 80% Coverage)",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| Total Analyzed Sequences | {stats['total_sequences']} |",
        f"| Number of 30% Clusters | {stats['num_clusters']} |",
        f"| Number of Singletons | {stats['num_singletons']} |",
        f"| Singleton Fraction | {stats['singleton_fraction']:.2%} |",
        f"| Median Cluster Size | {stats['median_cluster_size']} |",
        f"| Maximum Cluster Size | {stats['max_cluster_size']} |",
        f"| Clusters Spanning Multiple Families | {stats['num_multi_family_clusters']} |",
        "",
        "## Top IS Families by Cluster Count",
        "",
        "| IS Family | Protein Sequences | 30% Clusters |",
        "| :--- | :--- | :--- |",
    ]
    for row in stats["family_stats"][:25]:
        cluster_lines.append(f"| {row['family']} | {row['sequence_count']} | {row['cluster_count']} |")

    output_cluster_md.parent.mkdir(parents=True, exist_ok=True)
    with open(output_cluster_md, "w", encoding="utf-8") as f:
        f.write("\n".join(cluster_lines))

    # 2. Overall dataset summary report
    neg_df = pl.read_parquet(negatives_parquet) if negatives_parquet.exists() else pl.DataFrame()
    train_df = pl.read_parquet(split_dir / "train.parquet")
    val_df = pl.read_parquet(split_dir / "validation.parquet")
    test_df = pl.read_parquet(split_dir / "test.parquet")

    summary_lines = [
        "# DeepISE Phase-0 Dataset Summary",
        "",
        "## Overall Dataset Counts",
        "",
        "| Dataset Component | Count |",
        "| :--- | :--- |",
        f"| Positive Transposases (Deduplicated) | {len(tpase_df)} |",
        f"| Negative Proteins (Swiss-Prot Curated) | {len(neg_df)} |",
        f"| 30% Identity Clusters | {stats['num_clusters']} |",
        f"| Train Set (70%) | {len(train_df)} |",
        f"| Validation Set (15%) | {len(val_df)} |",
        f"| Test Set (15%) | {len(test_df)} |",
        "",
        "## Negative Dataset Stratification",
        "",
        "| Negative Class | Count | Fraction |",
        "| :--- | :--- | :--- |",
    ]
    if len(neg_df) > 0:
        neg_counts = neg_df.group_by("negative_type").len().sort("len", descending=True)
        for r in neg_counts.iter_rows(named=True):
            summary_lines.append(f"| {r['negative_type']} | {r['len']} | {r['len']/len(neg_df):.2%} |")

    output_summary_md.parent.mkdir(parents=True, exist_ok=True)
    with open(output_summary_md, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    console.print(f"[green]Reports generated: {output_summary_md} and {output_cluster_md}[/green]")


@app.command()
def prepare_eval(
    split_dir: Path = typer.Option(Path("data/splits/cluster30"), help="Phase-0 split directory"),
    negatives_parquet: Path = typer.Option(Path("data/processed/negatives.parquet"), help="Negative proteins Parquet"),
    output_dir: Path = typer.Option(Path("data/splits/cluster30"), help="Output directory"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Assemble balanced positive+negative evaluation datasets (1:3 ratio) for validation and test."""
    console.print("[bold blue]Phase-1 Step 1: Building stratified evaluation datasets...[/bold blue]")
    build_combined_splits(split_dir, negatives_parquet, output_dir, seed=seed)

    train_faa = split_dir / "train.faa"
    test_combined = pl.read_parquet(output_dir / "test_combined.parquet")
    val_combined = pl.read_parquet(output_dir / "validation_combined.parquet")

    # Annotate max_train_identity
    test_hits = split_dir / "test_eval_vs_train.tsv"
    run_mmseqs_search(output_dir / "test_combined.faa", train_faa, test_hits, output_dir / "tmp_test_ident", sensitivity=7.5)

    max_id_map = {}
    with open(test_hits, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                q, t, fid = parts[0], parts[1], float(parts[2])
                if q not in max_id_map or fid > max_id_map[q]:
                    max_id_map[q] = fid

    test_annotated = test_combined.with_columns(
        pl.col("seq_id").map_elements(lambda s: max_id_map.get(s, 0.0), return_dtype=pl.Float64).alias("max_train_identity")
    )
    test_annotated.write_parquet(output_dir / "test_combined.parquet")

    val_hits = split_dir / "val_eval_vs_train.tsv"
    run_mmseqs_search(output_dir / "validation_combined.faa", train_faa, val_hits, output_dir / "tmp_val_ident", sensitivity=7.5)
    val_id_map = {}
    with open(val_hits, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                q, t, fid = parts[0], parts[1], float(parts[2])
                if q not in val_id_map or fid > val_id_map[q]:
                    val_id_map[q] = fid

    val_annotated = val_combined.with_columns(
        pl.col("seq_id").map_elements(lambda s: val_id_map.get(s, 0.0), return_dtype=pl.Float64).alias("max_train_identity")
    )
    val_annotated.write_parquet(output_dir / "validation_combined.parquet")
    console.print(f"[green]Evaluation datasets assembled in {output_dir}[/green]")


@app.command()
def bench_blast(
    split_dir: Path = typer.Option(Path("data/splits/cluster30"), help="Evaluation splits directory"),
    results_dir: Path = typer.Option(Path("benchmark/results"), help="Results directory"),
    threads: int = typer.Option(16, help="Threads"),
):
    """Run BLASTP baseline on validation and test sets."""
    console.print("[bold blue]Running BLASTP baseline benchmark...[/bold blue]")
    train_faa = split_dir / "train.faa"
    val_faa = split_dir / "validation_combined.faa"
    test_faa = split_dir / "test_combined.faa"

    db_dir = Path("benchmark/db")
    db_prefix = build_blast_database(train_faa, db_dir, db_name="blast_tpase")

    val_raw = results_dir / "blast_val_raw.tsv"
    test_raw = results_dir / "blast_test_raw.tsv"
    run_blastp(val_faa, db_prefix, val_raw, threads=threads)
    run_blastp(test_faa, db_prefix, test_raw, threads=threads)

    val_preds = parse_blast_predictions(val_raw, val_faa)
    test_preds = parse_blast_predictions(test_raw, test_faa)

    results_dir.mkdir(parents=True, exist_ok=True)
    val_preds.write_parquet(results_dir / "blast_val_preds.parquet")
    test_preds.write_parquet(results_dir / "blast.parquet")
    test_preds.write_csv(results_dir / "blast.tsv", separator="\t")
    console.print(f"[green]BLASTP predictions saved to {results_dir / 'blast.tsv'}[/green]")


@app.command()
def bench_mmseqs(
    split_dir: Path = typer.Option(Path("data/splits/cluster30"), help="Evaluation splits directory"),
    results_dir: Path = typer.Option(Path("benchmark/results"), help="Results directory"),
    threads: int = typer.Option(16, help="Threads"),
):
    """Run MMseqs2 baseline on validation and test sets."""
    console.print("[bold blue]Running MMseqs2 baseline benchmark...[/bold blue]")
    train_faa = split_dir / "train.faa"
    val_faa = split_dir / "validation_combined.faa"
    test_faa = split_dir / "test_combined.faa"

    val_raw = results_dir / "mmseqs_val_raw.tsv"
    test_raw = results_dir / "mmseqs_test_raw.tsv"
    tmp_dir = Path("benchmark/tmp_mmseqs")

    run_mmseqs_search(val_faa, train_faa, val_raw, tmp_dir / "val", threads=threads)
    run_mmseqs_search(test_faa, train_faa, test_raw, tmp_dir / "test", threads=threads)

    val_preds = parse_mmseqs_predictions(val_raw, val_faa)
    test_preds = parse_mmseqs_predictions(test_raw, test_faa)

    results_dir.mkdir(parents=True, exist_ok=True)
    val_preds.write_parquet(results_dir / "mmseqs_val_preds.parquet")
    test_preds.write_parquet(results_dir / "mmseqs.parquet")
    test_preds.write_csv(results_dir / "mmseqs.tsv", separator="\t")
    console.print(f"[green]MMseqs2 predictions saved to {results_dir / 'mmseqs.tsv'}[/green]")


@app.command()
def bench_hmmer(
    split_dir: Path = typer.Option(Path("data/splits/cluster30"), help="Evaluation splits directory"),
    results_dir: Path = typer.Option(Path("benchmark/results"), help="Results directory"),
    threads: int = typer.Option(16, help="Threads"),
):
    """Run HMMER family profile baseline on validation and test sets."""
    console.print("[bold blue]Running HMMER family profile baseline benchmark...[/bold blue]")
    train_parquet = split_dir / "train.parquet"
    val_faa = split_dir / "validation_combined.faa"
    test_faa = split_dir / "test_combined.faa"

    hmm_db = Path("benchmark/db/deepise_tpases.hmm")
    tmp_dir = Path("benchmark/tmp_hmmer")
    build_family_hmms(train_parquet, hmm_db, tmp_dir, threads=threads)

    val_tbl = results_dir / "hmmer_val_raw.tbl"
    test_tbl = results_dir / "hmmer_test_raw.tbl"

    run_hmmsearch(val_faa, hmm_db, val_tbl, threads=threads)
    run_hmmsearch(test_faa, hmm_db, test_tbl, threads=threads)

    val_preds = parse_hmmsearch_predictions(val_tbl, val_faa)
    test_preds = parse_hmmsearch_predictions(test_tbl, test_faa)

    results_dir.mkdir(parents=True, exist_ok=True)
    val_preds.write_parquet(results_dir / "hmmer_val_preds.parquet")
    test_preds.write_parquet(results_dir / "hmmer.parquet")
    test_preds.write_csv(results_dir / "hmmer.tsv", separator="\t")
    console.print(f"[green]HMMER predictions saved to {results_dir / 'hmmer.tsv'}[/green]")


@app.command()
def bench_esm2(
    split_dir: Path = typer.Option(Path("data/splits/cluster30"), help="Evaluation splits directory"),
    results_dir: Path = typer.Option(Path("benchmark/results"), help="Results directory"),
    embeddings_dir: Path = typer.Option(Path("data/processed/embeddings"), help="Embeddings directory"),
    model_name: str = typer.Option("facebook/esm2_t12_35M_UR50D", help="ESM-2 checkpoint"),
    batch_size: int = typer.Option(16, help="Batch size"),
):
    """Extract ESM-2 embeddings and train LR & MLP baseline models."""
    import numpy as np
    from deepise_ml.models.esm2 import extract_esm2_embeddings, ESM2LinearClassifier, ESM2MLPClassifier

    console.print(f"[bold blue]Extracting ESM-2 embeddings ({model_name})...[/bold blue]")
    train_faa = split_dir / "train_combined.faa"
    val_faa = split_dir / "validation_combined.faa"
    test_faa = split_dir / "test_combined.faa"

    train_npy, _ = extract_esm2_embeddings(train_faa, embeddings_dir / "esm2_train.npy", embeddings_dir / "esm2_train_index.parquet", model_name=model_name, batch_size=batch_size)
    val_npy, _ = extract_esm2_embeddings(val_faa, embeddings_dir / "esm2_val.npy", embeddings_dir / "esm2_val_index.parquet", model_name=model_name, batch_size=batch_size)
    test_npy, _ = extract_esm2_embeddings(test_faa, embeddings_dir / "esm2_test.npy", embeddings_dir / "esm2_test_index.parquet", model_name=model_name, batch_size=batch_size)

    X_train = np.load(train_npy)
    X_val = np.load(val_npy)
    X_test = np.load(test_npy)

    y_train = pl.read_parquet(split_dir / "train_combined.parquet")["label"].to_numpy()
    y_val = pl.read_parquet(split_dir / "validation_combined.parquet")["label"].to_numpy()
    y_test = pl.read_parquet(split_dir / "test_combined.parquet")["label"].to_numpy()

    val_ids = pl.read_parquet(split_dir / "validation_combined.parquet")["seq_id"].to_list()
    test_ids = pl.read_parquet(split_dir / "test_combined.parquet")["seq_id"].to_list()

    # 1. ESM2-LR
    console.print("Training ESM2-LR (Logistic Regression)...")
    clf_lr = ESM2LinearClassifier(C=1.0)
    clf_lr.fit(X_train, y_train)
    val_scores_lr = clf_lr.predict_proba(X_val)
    test_scores_lr = clf_lr.predict_proba(X_test)

    results_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"seq_id": val_ids, "score": val_scores_lr}).write_parquet(results_dir / "esm2_lr_val_preds.parquet")
    test_lr_df = pl.DataFrame({"seq_id": test_ids, "score": test_scores_lr})
    test_lr_df.write_parquet(results_dir / "esm2_lr.parquet")
    test_lr_df.write_csv(results_dir / "esm2_lr.tsv", separator="\t")

    # 2. ESM2-MLP
    console.print("Training ESM2-MLP (1-hidden-layer MLP)...")
    clf_mlp = ESM2MLPClassifier(hidden_dim=256, dropout=0.2, epochs=25)
    clf_mlp.fit(X_train, y_train)
    val_scores_mlp = clf_mlp.predict_proba(X_val)
    test_scores_mlp = clf_mlp.predict_proba(X_test)

    pl.DataFrame({"seq_id": val_ids, "score": val_scores_mlp}).write_parquet(results_dir / "esm2_mlp_val_preds.parquet")
    test_mlp_df = pl.DataFrame({"seq_id": test_ids, "score": test_scores_mlp})
    test_mlp_df.write_parquet(results_dir / "esm2_mlp.parquet")
    test_mlp_df.write_csv(results_dir / "esm2_mlp.tsv", separator="\t")

    console.print(f"[green]ESM-2 predictions saved to {results_dir}[/green]")


@app.command()
def eval_phase1(
    split_dir: Path = typer.Option(Path("data/splits/cluster30"), help="Evaluation splits directory"),
    results_dir: Path = typer.Option(Path("benchmark/results"), help="Results directory"),
    tables_dir: Path = typer.Option(Path("benchmark/tables"), help="Tables directory"),
    reports_dir: Path = typer.Option(Path("benchmark/reports"), help="Reports directory"),
):
    """Evaluate all available baselines, tune thresholds on validation, and generate benchmark tables and Go/No-Go report."""
    console.print("[bold yellow]=== Evaluating Phase-1 Models & Generating Benchmark Reports ===[/bold yellow]")
    val_df = pl.read_parquet(split_dir / "validation_combined.parquet")
    test_df = pl.read_parquet(split_dir / "test_combined.parquet")

    val_true = val_df["label"].to_numpy()
    test_true = test_df["label"].to_numpy()

    models_to_eval = [
        ("BLASTP", "blast_val_preds.parquet", "blast.parquet"),
        ("MMseqs2", "mmseqs_val_preds.parquet", "mmseqs.parquet"),
        ("Whole-pHMM", "hmmer_val_preds.parquet", "hmmer.parquet"),
        ("ESM2-LR", "esm2_lr_val_preds.parquet", "esm2_lr.parquet"),
        ("ESM2-MLP", "esm2_mlp_val_preds.parquet", "esm2_mlp.parquet"),
    ]

    metrics_by_model = {}
    stratified_by_model = {}
    family_by_model = {}

    for model_name, val_file, test_file in models_to_eval:
        val_path = results_dir / val_file
        test_path = results_dir / test_file
        if not val_path.exists() or not test_path.exists():
            continue

        val_preds = pl.read_parquet(val_path)
        test_preds = pl.read_parquet(test_path)

        val_joined = val_df.join(val_preds.select(["seq_id", "score"]), on="seq_id", how="left")
        test_joined = test_df.join(test_preds.select(["seq_id", "score"]), on="seq_id", how="left")

        val_scores = val_joined["score"].fill_null(0.0).to_numpy()
        test_scores = test_joined["score"].fill_null(0.0).to_numpy()

        m = compute_overall_metrics(val_true, val_scores, test_true, test_scores)
        th_5 = m["threshold_5pct_fdr"]

        strat_df = compute_identity_stratified_recall(test_df, test_preds, th_5)
        fam_df, macro_rec = compute_family_recall(test_df, test_preds, th_5)
        m["macro_recall"] = macro_rec

        metrics_by_model[model_name] = m
        stratified_by_model[model_name] = strat_df
        family_by_model[model_name] = fam_df
        console.print(f"[{model_name}] AUPRC: {m['auprc']:.4f} | Recall@5%FDR: {m['test_recall_at_5pct_fdr']:.4f} | Macro: {macro_rec:.4f}")

    # Export tables
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    overall_df = export_overall_metrics_table(metrics_by_model, tables_dir / "overall_metrics.tsv")
    strat_df_all = export_stratified_metrics_table(stratified_by_model, tables_dir / "identity_stratified_metrics.tsv")
    fam_df_all = export_family_metrics_table(family_by_model, tables_dir / "family_metrics.tsv")

    generate_phase1_benchmark_report(overall_df, strat_df_all, reports_dir / "phase1_benchmark.md")
    generate_go_no_go_report(overall_df, strat_df_all, reports_dir / "go_no_go.md")

    console.print(f"[bold green]Phase-1 Benchmark Reports generated in {reports_dir}[/bold green]")


@app.command()
def build_boundary_benchmark(
    elements_parquet: Path = typer.Option(Path("data/processed/is_elements.parquet"), help="Processed elements parquet"),
    test_split_parquet: Path = typer.Option(Path("data/splits/cluster30/test_combined.parquet"), help="Test split parquet"),
    val_split_parquet: Path = typer.Option(Path("data/splits/cluster30/validation_combined.parquet"), help="Validation split parquet"),
    tpases_parquet: Path = typer.Option(Path("data/processed/tpases.parquet"), help="Processed tpases parquet"),
    output_parquet: Path = typer.Option(Path("data/benchmark/phase2_ground_truth.parquet"), help="Output benchmark contigs"),
    max_elements: int = typer.Option(350, help="Max benchmark elements"),
):
    """Build Phase-2 IS element boundary ground-truth benchmark contigs."""
    console.print("[bold blue]Generating Phase-2 boundary ground-truth benchmark contigs...[/bold blue]")
    df = build_benchmark_contigs(
        elements_parquet=elements_parquet,
        test_split_parquet=test_split_parquet,
        val_split_parquet=val_split_parquet,
        tpases_parquet=tpases_parquet,
        output_parquet=output_parquet,
        max_elements=max_elements,
    )
    console.print(f"[bold green]Successfully generated {len(df)} benchmark contigs in {output_parquet}[/bold green]")


@app.command()
def eval_phase2(
    benchmark_parquet: Path = typer.Option(Path("data/benchmark/phase2_ground_truth.parquet"), help="Benchmark contigs parquet"),
    tables_dir: Path = typer.Option(Path("benchmark/tables"), help="Output tables directory"),
    reports_dir: Path = typer.Option(Path("benchmark/reports"), help="Output reports directory"),
):
    """Run dual-track comparative evaluation of Plan A vs Plan B on the ground-truth benchmark."""
    console.print("[bold yellow]=== Running Phase-2 Dual-Track Comparative Benchmark (Plan A vs Plan B) ===[/bold yellow]")
    assert benchmark_parquet.exists(), f"Benchmark file {benchmark_parquet} does not exist. Run build-boundary-benchmark first."
    bench_df = pl.read_parquet(benchmark_parquet)
    console.print(f"Loaded {len(bench_df)} benchmark contigs across {bench_df['family'].n_unique()} IS families.")

    # 1. Run Plan B
    console.print("[bold cyan]Evaluating Plan B (Canonical TIR/TSD Prototype Engine)...[/bold cyan]")
    engine_b = PlanBCanonicalEngine()
    preds_b, metrics_b, fam_b = evaluate_engine_on_benchmark(engine_b, bench_df, "plan_b")
    console.print(f"Plan B Near Match (<=3bp): {metrics_b['near_match_rate_3bp']*100:.2f}% | Latency: {metrics_b['avg_latency_ms']:.2f}ms")

    # 2. Run Plan A
    console.print("[bold green]Evaluating Plan A (Full-Family Adaptive Engine)...[/bold green]")
    engine_a = PlanAAdaptiveEngine()
    preds_a, metrics_a, fam_a = evaluate_engine_on_benchmark(engine_a, bench_df, "plan_a")
    console.print(f"Plan A Near Match (<=3bp): {metrics_a['near_match_rate_3bp']*100:.2f}% | Latency: {metrics_a['avg_latency_ms']:.2f}ms")

    # 3. Export comparative tables & report
    comp_df, fam_joined = export_phase2_comparison_tables(metrics_a, metrics_b, fam_a, fam_b, tables_dir)
    report_path = reports_dir / "phase2_plan_a_vs_b.md"
    generate_phase2_report_markdown(comp_df, fam_joined, report_path)

    console.print(f"[bold green]Phase-2 Benchmark Complete! Report saved to {report_path}[/bold green]")


@app.command()
def scan_genome(
    fasta_path: Path = typer.Option(..., help="Path to input genome FASTA"),
    output_dir: Path = typer.Option(Path("results/genome_scan"), help="Output directory for GFF3, TSV, FNA"),
    mode: str = typer.Option("plan_a", help="Boundary engine mode: 'plan_a' (adaptive) or 'plan_b' (canonical)"),
    threads: int = typer.Option(4, help="Number of CPU threads for gene prediction and HMM search"),
    min_bitscore: float = typer.Option(10.6, help="Bitscore threshold for transposase filtering"),
):
    """Scan a bacterial genome FASTA file and detect full-length IS elements."""
    console.print(f"[bold blue]Scanning genome {fasta_path} using DeepISE ({mode.upper()})...[/bold blue]")
    scanner = DeepISEGenomeScanner(min_bitscore=min_bitscore)
    elements = scanner.scan_genome(fasta_path, mode=mode, threads=threads)
    console.print(f"Detected [bold green]{len(elements)}[/bold green] non-redundant IS elements.")

    from Bio import SeqIO
    contigs = {r.id: str(r.seq).upper() for r in SeqIO.parse(fasta_path, "fasta")}
    gff_p, tsv_p, fna_p = export_genome_results(elements, contigs, output_dir)
    console.print(f"Exported results to [green]{gff_p}[/green], [green]{tsv_p}[/green], [green]{fna_p}[/green]")


@app.command()
def benchmark_genome(
    fasta_path: Path = typer.Option(Path("data/genomes/NC_000913.3.fna"), help="Reference genome FASTA"),
    feature_table_path: Path = typer.Option(Path("data/genomes/NC_000913.3.gff"), help="NCBI curated feature table"),
    tables_dir: Path = typer.Option(Path("benchmark/tables"), help="Tables output directory"),
    reports_dir: Path = typer.Option(Path("benchmark/reports"), help="Reports output directory"),
    results_dir: Path = typer.Option(Path("benchmark/results"), help="Results output directory"),
):
    """Run full-genome end-to-end benchmark on reference genome (e.g. E. coli K-12)."""
    console.print("[bold yellow]=== Running Real Bacterial Genome End-to-End Benchmark ===[/bold yellow]")
    run_full_genome_benchmark(
        fasta_path=fasta_path,
        feature_table_path=feature_table_path,
        tables_dir=tables_dir,
        reports_dir=reports_dir,
        results_dir=results_dir,
    )
    console.print("[bold green]Genome Benchmark Completed Successfully![/bold green]")


@app.command("screen-proteins")
def screen_proteins(
    fasta_path: Path = typer.Option(..., "--fasta", "-f", help="Protein FASTA input"),
    profile: str = typer.Option("standard", "--profile", help="Stage-1 profile: 'fast' or 'standard'"),
    output_path: Path = typer.Option(Path("results/protein_screening/predictions.tsv"), "--output", "-o", help="Prediction TSV output"),
    model_dir: Path = typer.Option(Path("benchmark/models"), "--model-dir", help="Classifier artifact directory"),
    device: Optional[str] = typer.Option(None, "--device", help="Embedding device, such as 'cpu' or 'cuda'"),
    batch_size: int = typer.Option(32, "--batch-size", min=1, help="Protein embedding batch size"),
    minimum_length: int = typer.Option(50, "--minimum-length", min=1, help="Minimum accepted protein length"),
    maximum_length: int = typer.Option(2000, "--maximum-length", min=1, help="Maximum accepted protein length"),
):
    try:
        resolved_profile = resolve_profile(profile, model_dir)
        classifier = load_classifier(resolved_profile)
        proteins = read_protein_fasta(fasta_path, minimum_length, maximum_length)
        engine = PluggablePLMEngine(model_key=resolved_profile.model_key, device=device)
        service = ProteinScreeningService(
            embedding_engine=engine,
            classifier=classifier,
            profile=resolved_profile,
        )
        records = service.write_predictions(proteins, output_path, batch_size)
    except ProteinScreeningError as error:
        console.print(f"[red]Protein screening failed: {error}[/red]")
        raise typer.Exit(code=1)
    console.print(
        f"[green]Wrote {len(records)} Stage-1 protein predictions to {output_path}[/green]"
    )


@app.command("screen-proteins-homology")
def screen_proteins_homology(
    fasta_path: Path = typer.Option(..., "--fasta", "-f", help="Protein FASTA input"),
    reference_fasta: Path = typer.Option(..., "--reference-fasta", help="Reference protein FASTA for MMseqs2"),
    hmm_database: Path = typer.Option(..., "--hmm-database", help="Pressed HMM database for HMMER"),
    profile: str = typer.Option("standard", "--profile", help="Stage-1 profile: 'fast' or 'standard'"),
    output_path: Path = typer.Option(Path("results/protein_screening_homology/predictions.tsv"), "--output", "-o", help="Combined prediction TSV output"),
    mmseqs_bin: Optional[Path] = typer.Option(None, "--mmseqs-bin", help="Explicit path to mmseqs executable"),
    hmmsearch_bin: Optional[Path] = typer.Option(None, "--hmmsearch-bin", help="Explicit path to hmmsearch executable"),
    model_dir: Path = typer.Option(Path("benchmark/models"), "--model-dir", help="Classifier artifact directory"),
    device: Optional[str] = typer.Option(None, "--device", help="Embedding device, such as 'cpu' or 'cuda'"),
    batch_size: int = typer.Option(32, "--batch-size", min=1, help="Protein embedding batch size"),
    threads: int = typer.Option(1, "--threads", "-t", min=1, help="Number of CPU threads for homology searches"),
    minimum_length: int = typer.Option(50, "--minimum-length", min=1, help="Minimum accepted protein length"),
    maximum_length: int = typer.Option(2000, "--maximum-length", min=1, help="Maximum accepted protein length"),
):
    """Stage-1 PLM screening combined with MMseqs2 and HMMER homology evidence and deterministic routing."""
    try:
        resolved_profile = resolve_profile(profile, model_dir)
        classifier = load_classifier(resolved_profile)
        proteins = read_protein_fasta(fasta_path, minimum_length, maximum_length)
        paths = HomologyScreeningPaths(
            output_tsv=output_path,
            reference_fasta=reference_fasta,
            hmm_database=hmm_database,
        )
        paths.validate_inputs_and_destinations()
        engine = PluggablePLMEngine(model_key=resolved_profile.model_key, device=device)
        service = ProteinHomologyScreeningService(
            embedding_engine=engine,
            classifier=classifier,
            profile=resolved_profile,
            mmseqs_binary=mmseqs_bin,
            hmmer_binary=hmmsearch_bin,
            threads=threads,
        )
        records = service.screen_and_route(proteins, paths, batch_size=batch_size)
    except (ProteinScreeningError, HomologyAdapterError, HomologyContractError) as error:
        console.print(f"[red]Homology screening failed: {error}[/red]")
        raise typer.Exit(code=1)
    console.print(
        f"[green]Wrote {len(records)} routed protein predictions to {output_path}[/green]"
    )


@app.command("screen-proteins-research")
def screen_proteins_research(
    fasta_path: Path = typer.Option(..., "--fasta", "-f", help="Protein FASTA input"),
    reference_fasta: Path = typer.Option(..., "--reference-fasta", help="Reference protein FASTA for MMseqs2"),
    hmm_database: Path = typer.Option(..., "--hmm-database", help="Pressed HMM database for HMMER"),
    profile: str = typer.Option("standard", "--profile", help="Stage-1 profile: 'fast' or 'standard'"),
    output_path: Path = typer.Option(Path("results/protein_screening_research/predictions.tsv"), "--output", "-o", help="Comprehensive research prediction TSV output"),
    mmseqs_bin: Optional[Path] = typer.Option(None, "--mmseqs-bin", help="Explicit path to mmseqs executable"),
    hmmsearch_bin: Optional[Path] = typer.Option(None, "--hmmsearch-bin", help="Explicit path to hmmsearch executable"),
    model_dir: Path = typer.Option(Path("benchmark/models"), "--model-dir", help="Classifier artifact directory"),
    device: Optional[str] = typer.Option(None, "--device", help="Embedding device, such as 'cpu' or 'cuda'"),
    batch_size: int = typer.Option(32, "--batch-size", min=1, help="Protein embedding batch size"),
    threads: int = typer.Option(1, "--threads", "-t", min=1, help="Number of CPU threads for homology searches"),
    minimum_length: int = typer.Option(50, "--minimum-length", min=1, help="Minimum accepted protein length"),
    maximum_length: int = typer.Option(2000, "--maximum-length", min=1, help="Maximum accepted protein length"),
    enable_stage2a: bool = typer.Option(True, "--enable-stage2a/--disable-stage2a", help="Enable ProstT5 fast 3Di structure rescue"),
    enable_stage2b: bool = typer.Option(True, "--enable-stage2b/--disable-stage2b", help="Enable explicit 3D modeling and SaProt validation"),
):
    """Full hierarchical research screening combining Stage 1, Homology, Stage 2A ProstT5, Stage 2B SaProt, and Evidence Fusion."""
    try:
        resolved_profile = resolve_profile(profile, model_dir)
        classifier = load_classifier(resolved_profile)
        proteins = read_protein_fasta(fasta_path, minimum_length, maximum_length)
        paths = HomologyScreeningPaths(
            output_tsv=output_path,
            reference_fasta=reference_fasta,
            hmm_database=hmm_database,
        )
        paths.validate_inputs_and_destinations()
        engine = PluggablePLMEngine(model_key=resolved_profile.model_key, device=device)
        service = ProteinResearchScreeningService(
            embedding_engine=engine,
            classifier=classifier,
            profile=resolved_profile,
            mmseqs_binary=mmseqs_bin,
            hmmer_binary=hmmsearch_bin,
            threads=threads,
        )
        records = service.screen_and_fuse(
            proteins=proteins,
            paths=paths,
            batch_size=batch_size,
            enable_stage2a=enable_stage2a,
            enable_stage2b=enable_stage2b,
        )
    except (ProteinScreeningError, HomologyAdapterError, HomologyContractError) as error:
        console.print(f"[red]Research screening failed: {error}[/red]")
        raise typer.Exit(code=1)
    console.print(
        f"[green]Wrote {len(records)} fully fused research predictions to {output_path}[/green]"
    )


@app.command()
def scan(
    fasta_path: Path = typer.Option(..., "--fasta", "-f", help="Path to input FASTA file (complete genome or metagenome contigs)"),
    output_dir: Path = typer.Option(Path("results/deepise_scan"), "--outdir", "-o", help="Output directory for results"),
    mode: str = typer.Option("hybrid", "--mode", "-m", help="Boundary engine mode: 'hybrid' (physics+CNN), 'plan_a', or 'plan_b'"),
    threads: int = typer.Option(4, "--threads", "-t", help="Number of CPU threads"),
    meta: bool = typer.Option(False, "--meta", "--metagenome", help="Force metagenomic mode (Pyrodigal meta + contig streaming)"),
    min_contig_len: int = typer.Option(500, "--min-contig-len", help="Minimum contig length filter in metagenome mode"),
    min_bitscore: float = typer.Option(10.6, "--min-bitscore", help="Bitscore threshold for transposase filtering"),
    batch_size: int = typer.Option(1000, "--batch-size", help="Batch size of contigs for streaming processing"),
    research: bool = typer.Option(False, "--research/--no-research", help="Enable closed-loop Stage 2 structure validation & novelty discovery on detected transposases"),
    reference_fasta: Optional[Path] = typer.Option(None, "--reference-fasta", help="Reference protein FASTA for research validation"),
    hmm_database: Optional[Path] = typer.Option(None, "--hmm-database", help="Pressed HMM database for research validation"),
    research_profile: str = typer.Option("fast", "--research-profile", help="Research screening profile: 'fast' (ESM2-8M) or 'standard' (ESM2-35M)"),
    enable_stage2a: bool = typer.Option(True, "--enable-stage2a/--disable-stage2a", help="Enable ProstT5 fast 3Di structure rescue in research validation"),
    enable_stage2b: bool = typer.Option(True, "--enable-stage2b/--disable-stage2b", help="Enable explicit 3D modeling and SaProt in research validation"),
):
    """Universal DeepISE scanner for complete bacterial genomes, draft assemblies, and metagenomes."""
    from Bio import SeqIO
    is_multi_contig = meta
    if not is_multi_contig:
        records_sample = []
        for r in SeqIO.parse(fasta_path, "fasta"):
            records_sample.append(r)
            if len(records_sample) >= 5:
                break
        if len(records_sample) > 1 or any(len(r.seq) < 100_000 for r in records_sample):
            is_multi_contig = True

    if is_multi_contig:
        console.print(f"[bold cyan]Running DeepISE Metagenomic Production Scanner on {fasta_path}...[/bold cyan]")
        scanner = MetagenomeScanner(
            min_bitscore=min_bitscore,
            min_contig_len=min_contig_len,
            batch_size=batch_size,
            boundary_mode=mode,
        )
        elements, stats = scanner.scan(fasta_path=fasta_path, threads=threads)
        gff_p, tsv_p, fna_p, faa_p, json_p = export_metagenome_results(elements, stats, output_dir)
        console.print(f"Detected [bold green]{len(elements)}[/bold green] IS elements in {stats['runtime_seconds']}s ({stats['throughput_mbp_per_sec']} Mbp/s).")
        console.print(f"Breakdown: {stats['breakdown_by_status']['complete']} complete, {stats['breakdown_by_status']['partial']} partial, {stats['breakdown_by_status']['pseudo']} pseudo.")
        console.print(f"Outputs written to [green]{output_dir}[/green]:")
        console.print(f"  - GFF3: [green]{gff_p.name}[/green]")
        console.print(f"  - TSV:  [green]{tsv_p.name}[/green]")
        console.print(f"  - FNA:  [green]{fna_p.name}[/green]")
        console.print(f"  - FAA:  [green]{faa_p.name}[/green]")
        console.print(f"  - JSON: [green]{json_p.name}[/green]")
    else:
        console.print(f"[bold blue]Running DeepISE Single-Genome Scanner on {fasta_path}...[/bold blue]")
        scanner = DeepISEGenomeScanner(min_bitscore=min_bitscore)
        elements = scanner.scan_genome(fasta_path, mode=mode, threads=threads)
        contigs = {r.id: str(r.seq).upper() for r in SeqIO.parse(fasta_path, "fasta")}
        gff_p, tsv_p, fna_p = export_genome_results(elements, contigs, output_dir)
        console.print(f"Detected [bold green]{len(elements)}[/bold green] IS elements.")
        console.print(f"Exported results to [green]{gff_p}[/green], [green]{tsv_p}[/green], [green]{fna_p}[/green]")

    if research:
        console.print("[bold magenta]Executing closed-loop Stage 2 structure validation and novelty discovery...[/bold magenta]")
        faa_file = output_dir / "deepise_tpases.faa"
        tsv_file = output_dir / "deepise_is_elements.tsv"
        json_file = output_dir / "deepise_summary.json"
        res = perform_closed_loop_research_validation(
            elements_tsv=tsv_file,
            tpases_faa=faa_file,
            output_dir=output_dir,
            reference_fasta=reference_fasta,
            hmm_database=hmm_database,
            profile=research_profile,
            enable_stage2a=enable_stage2a,
            enable_stage2b=enable_stage2b,
            threads=threads,
            summary_json=json_file,
        )
        console.print(f"[green]Research validation completed on {res.total_tpases_evaluated} transposases:[/green]")
        console.print(f"  - High Confidence Novel IS: [bold cyan]{res.high_confidence_novel_count}[/bold cyan]")
        console.print(f"  - High Confidence Known IS: [bold green]{res.high_confidence_known_count}[/bold green]")
        console.print(f"  - Candidate Novel IS:       [yellow]{res.candidate_novel_count}[/yellow]")
        console.print(f"  - Research Predictions:     [green]{res.research_predictions_path.name}[/green]")
        console.print(f"  - Novel Discoveries:        [bold green]{res.novel_discoveries_path.name}[/bold green]")


@app.command()
def scan_metagenome(
    fasta_path: Path = typer.Option(..., "--fasta", "-f", help="Path to metagenomic contigs FASTA"),
    output_dir: Path = typer.Option(Path("results/metagenome_scan"), "--outdir", "-o", help="Output directory"),
    mode: str = typer.Option("hybrid", "--mode", "-m", help="Boundary engine mode: 'hybrid', 'plan_a', or 'plan_b'"),
    threads: int = typer.Option(4, "--threads", "-t", help="Number of CPU threads"),
    min_contig_len: int = typer.Option(500, "--min-contig-len", help="Minimum contig length filter"),
    min_bitscore: float = typer.Option(10.6, "--min-bitscore", help="Bitscore threshold for transposases"),
    batch_size: int = typer.Option(1000, "--batch-size", help="Batch size of contigs for streaming processing"),
    research: bool = typer.Option(False, "--research/--no-research", help="Enable closed-loop Stage 2 structure validation & novelty discovery on detected transposases"),
    reference_fasta: Optional[Path] = typer.Option(None, "--reference-fasta", help="Reference protein FASTA for research validation"),
    hmm_database: Optional[Path] = typer.Option(None, "--hmm-database", help="Pressed HMM database for research validation"),
    research_profile: str = typer.Option("fast", "--research-profile", help="Research screening profile: 'fast' (ESM2-8M) or 'standard' (ESM2-35M)"),
    enable_stage2a: bool = typer.Option(True, "--enable-stage2a/--disable-stage2a", help="Enable ProstT5 fast 3Di structure rescue in research validation"),
    enable_stage2b: bool = typer.Option(True, "--enable-stage2b/--disable-stage2b", help="Enable explicit 3D modeling and SaProt in research validation"),
):
    """Scan fragmented metagenomes or multi-contig MAGs with edge-truncation classification."""
    console.print(f"[bold cyan]Scanning metagenome contigs from {fasta_path} using mode={mode.upper()}...[/bold cyan]")
    scanner = MetagenomeScanner(
        min_bitscore=min_bitscore,
        min_contig_len=min_contig_len,
        batch_size=batch_size,
        boundary_mode=mode,
    )
    elements, stats = scanner.scan(fasta_path=fasta_path, threads=threads)
    gff_p, tsv_p, fna_p, faa_p, json_p = export_metagenome_results(elements, stats, output_dir)
    console.print(f"Detected [bold green]{len(elements)}[/bold green] IS elements in {stats['runtime_seconds']}s ({stats['throughput_mbp_per_sec']} Mbp/s).")
    console.print(f"Outputs written to [green]{output_dir}[/green]: GFF3, TSV, FNA, FAA, JSON.")

    if research:
        console.print("[bold magenta]Executing closed-loop Stage 2 structure validation and novelty discovery...[/bold magenta]")
        res = perform_closed_loop_research_validation(
            elements_tsv=tsv_p,
            tpases_faa=faa_p,
            output_dir=output_dir,
            reference_fasta=reference_fasta,
            hmm_database=hmm_database,
            profile=research_profile,
            enable_stage2a=enable_stage2a,
            enable_stage2b=enable_stage2b,
            threads=threads,
            summary_json=json_p,
        )
        console.print(f"[green]Research validation completed on {res.total_tpases_evaluated} transposases:[/green]")
        console.print(f"  - High Confidence Novel IS: [bold cyan]{res.high_confidence_novel_count}[/bold cyan]")
        console.print(f"  - High Confidence Known IS: [bold green]{res.high_confidence_known_count}[/bold green]")
        console.print(f"  - Candidate Novel IS:       [yellow]{res.candidate_novel_count}[/yellow]")
        console.print(f"  - Research Predictions:     [green]{res.research_predictions_path.name}[/green]")
        console.print(f"  - Novel Discoveries:        [bold green]{res.novel_discoveries_path.name}[/bold green]")


@app.command()
def benchmark_metagenome(
    synthetic_fna: Path = typer.Option(Path("data/metagenomes/synthetic_metagenome_benchmark.fna"), help="Synthetic benchmark FASTA"),
    synthetic_gt_tsv: Path = typer.Option(Path("data/metagenomes/synthetic_metagenome_ground_truth.tsv"), help="Ground truth TSV"),
    real_fna: Path = typer.Option(Path("data/metagenomes/klebsiella_pneumoniae_draft.fna"), help="Real draft WGS FASTA"),
    tables_dir: Path = typer.Option(Path("benchmark/tables"), help="Tables output directory"),
    reports_dir: Path = typer.Option(Path("benchmark/reports"), help="Reports output directory"),
    results_dir: Path = typer.Option(Path("benchmark/results"), help="Results output directory"),
    threads: int = typer.Option(4, help="CPU threads"),
):
    """Execute complete Phase-4 metagenomic benchmark across synthetic contigs and real draft assemblies."""
    console.print("[bold yellow]=== Running Phase-4 Metagenomics Production Benchmark ===[/bold yellow]")
    tbl_p, rep_p = run_full_metagenome_benchmark(
        synthetic_fna=synthetic_fna,
        synthetic_gt_tsv=synthetic_gt_tsv,
        real_fna=real_fna,
        tables_dir=tables_dir,
        reports_dir=reports_dir,
        results_dir=results_dir,
        threads=threads,
    )
    console.print(f"[bold green]Phase-4 Benchmark complete! Table: {tbl_p}, Report: {rep_p}[/bold green]")


@app.command()
def version():
    """Print DeepISE version and execution environment info."""
    import torch
    console.print("[bold cyan]DeepISE[/bold cyan] version [green]0.1.0[/green]")
    console.print(f"PyTorch: {torch.__version__} (CUDA available: {torch.cuda.is_available()})")
    console.print("Neural Boundary Refiner: [green]benchmark/models/neural_boundary_refiner.pt[/green]")
    console.print("PLM Multi-Task Classifier: [green]benchmark/models/plm_family_classifier.pt[/green]")
    console.print("Profile HMM Database: [green]benchmark/db/deepise_tpases.hmm[/green]")


if __name__ == "__main__":
    app()
