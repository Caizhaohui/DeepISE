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
def run_phase0():
    """Execute the complete Phase-0 pipeline from raw data to frozen benchmark."""
    console.print("[bold yellow]=== Executing Complete Phase-0 Pipeline ===[/bold yellow]")
    normalize()
    deduplicate()
    cluster()
    split()
    audit()
    build_negatives()
    summary()
    console.print("[bold green]=== Phase-0 Data Closed Loop Complete! ===[/bold green]")


if __name__ == "__main__":
    app()
