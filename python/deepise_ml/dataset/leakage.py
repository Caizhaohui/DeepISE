import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import polars as pl


class LeakageViolationError(Exception):
    """Raised when test/train pairs violate the strict homology threshold."""
    pass


def run_mmseqs_leakage_search(
    query_faa: Path,
    target_faa: Path,
    output_tsv: Path,
    tmp_dir: Path,
    sensitivity: float = 7.5,
    threads: int = 8,
) -> Path:
    """Run all-vs-all MMseqs2 search between query (test/val) and target (train)."""
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    query_db = tmp_dir / "query_db"
    target_db = tmp_dir / "target_db"
    aln_db = tmp_dir / "aln_db"
    run_tmp = tmp_dir / "search_tmp"
    run_tmp.mkdir(parents=True, exist_ok=True)

    # createdb
    subprocess.run(["mmseqs", "createdb", str(query_faa), str(query_db)], check=True)
    subprocess.run(["mmseqs", "createdb", str(target_faa), str(target_db)], check=True)

    # search
    subprocess.run(
        [
            "mmseqs", "search",
            str(query_db),
            str(target_db),
            str(aln_db),
            str(run_tmp),
            "-s", str(sensitivity),
            "--threads", str(threads),
        ],
        check=True,
    )

    # convertalis
    subprocess.run(
        [
            "mmseqs", "convertalis",
            str(query_db),
            str(target_db),
            str(aln_db),
            str(output_tsv),
            "--format-output",
            "query,target,fident,alnlen,qcov,tcov,evalue,bits",
            "--threads", str(threads),
        ],
        check=True,
    )

    return output_tsv


def audit_leakage(
    search_tsv: Path,
    query_faa: Path,
    identity_threshold: float = 0.30,
    coverage_threshold: float = 0.80,
) -> Tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Analyze all-vs-all search results and identify any leakage violations.
    
    Returns:
        all_hits_df: DataFrame of all hits
        violations_df: DataFrame of hits violating the threshold
        summary: summary dictionary
    """
    # Read all query IDs from query_faa to capture queries with zero hits
    all_query_ids = []
    with open(query_faa, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(">"):
                all_query_ids.append(line[1:].strip().split()[0])

    hits = []
    if search_tsv.exists() and search_tsv.stat().st_size > 0:
        with open(search_tsv, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 8:
                    q, t, fid, aln, qcov, tcov, ev, bits = parts[:8]
                    fid = float(fid)
                    qcov = float(qcov)
                    tcov = float(tcov)
                    cov = max(qcov, tcov)
                    hits.append({
                        "query_id": q,
                        "nearest_train_id": t,
                        "identity": fid,
                        "alignment_length": int(aln),
                        "query_coverage": qcov,
                        "target_coverage": tcov,
                        "max_coverage": cov,
                        "evalue": float(ev),
                        "bitscore": float(bits),
                        "is_violation": (fid > identity_threshold and cov >= coverage_threshold),
                    })

    if hits:
        hits_df = pl.DataFrame(hits)
    else:
        hits_df = pl.DataFrame(schema={
            "query_id": pl.Utf8,
            "nearest_train_id": pl.Utf8,
            "identity": pl.Float64,
            "alignment_length": pl.Int64,
            "query_coverage": pl.Float64,
            "target_coverage": pl.Float64,
            "max_coverage": pl.Float64,
            "evalue": pl.Float64,
            "bitscore": pl.Float64,
            "is_violation": pl.Boolean,
        })

    violations_df = hits_df.filter(pl.col("is_violation"))

    # Compute max identity per query
    if len(hits_df) > 0:
        max_id_per_query = (
            hits_df.group_by("query_id")
            .agg([
                pl.col("identity").max().alias("max_identity_to_train"),
                pl.col("nearest_train_id").first(),
            ])
        )
    else:
        max_id_per_query = pl.DataFrame(schema={"query_id": pl.Utf8, "max_identity_to_train": pl.Float64})

    summary = {
        "total_queries": len(all_query_ids),
        "queries_with_hits": len(hits_df["query_id"].unique()) if len(hits_df) > 0 else 0,
        "total_violations": len(violations_df),
        "violating_queries": len(violations_df["query_id"].unique()) if len(violations_df) > 0 else 0,
        "max_identity_overall": float(hits_df["identity"].max()) if len(hits_df) > 0 else 0.0,
        "pass_audit": len(violations_df) == 0,
    }

    return hits_df, violations_df, summary


def generate_leakage_report_markdown(
    summary: dict,
    violations_df: pl.DataFrame,
    output_path: Path,
):
    """Write comprehensive leakage audit markdown report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    status_str = "PASSED (0 VIOLATIONS)" if summary["pass_audit"] else "FAILED (LEAKAGE DETECTED)"
    badge = "🟢" if summary["pass_audit"] else "🔴"

    lines = [
        f"# DeepISE Homology Leakage Audit Report",
        f"",
        f"**Audit Status:** {badge} **{status_str}**  ",
        f"**Strategy:** Cluster30 MMseqs2 Disjoint Split  ",
        f"**Thresholds:** `identity > 0.30` AND `coverage >= 0.80`  ",
        f"",
        f"## Summary Metrics",
        f"",
        f"| Metric | Value |",
        f"| :--- | :--- |",
        f"| Total Test Sequences | {summary['total_queries']} |",
        f"| Test Sequences with Train Hits | {summary['queries_with_hits']} |",
        f"| Total Violations | {summary['total_violations']} |",
        f"| Violating Queries | {summary['violating_queries']} |",
        f"| Maximum Test-Train Sequence Identity | {summary['max_identity_overall']:.4f} |",
        f"",
    ]

    if summary["total_violations"] > 0:
        lines.append("## Violating Sequence Pairs\n")
        lines.append("| Query ID | Nearest Train ID | Identity | Coverage | E-value |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")
        for row in violations_df.iter_rows(named=True):
            lines.append(
                f"| {row['query_id']} | {row['nearest_train_id']} | {row['identity']:.3f} | {row['max_coverage']:.3f} | {row['evalue']:.2e} |"
            )
    else:
        lines.append("## Conclusion\n")
        lines.append("The test dataset is strictly cluster-disjoint with the training dataset under the 30% sequence identity and 80% alignment coverage criteria. No homology leakage was detected.\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
