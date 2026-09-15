import shutil
import subprocess
from collections import defaultdict
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


def audit_homology_v2(
    search_tsv: Path,
    query_ids: List[str],
    labels: Dict[str, int] = None,
    families: Dict[str, str] = None,
    identity_threshold: float = 0.30,
    coverage_threshold: float = 0.80,
) -> Tuple[pl.DataFrame, dict]:
    """Audit query-vs-target homology using reciprocal full-length criteria.

    Strict Full-length match is defined as:
        qcov >= coverage_threshold AND tcov >= coverage_threshold
    
    max_full_length_identity is the maximum identity among all full-length matches.
    """
    labels = labels or {}
    families = families or {}

    query_hits = defaultdict(list)
    if search_tsv.exists() and search_tsv.stat().st_size > 0:
        with open(search_tsv, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 8:
                    q, t, fid, aln, qcov, tcov, ev, bits = parts[:8]
                    query_hits[q].append({
                        "target": t,
                        "fident": float(fid),
                        "alnlen": int(aln),
                        "qcov": float(qcov),
                        "tcov": float(tcov),
                        "evalue": float(ev),
                        "bitscore": float(bits),
                    })

    rows = []
    l1_exact_count = 0
    l2_close_count = 0
    l3_domain_only_count = 0
    strict_remote_30_count = 0
    strict_remote_20_count = 0
    no_full_length_count = 0

    for qid in query_ids:
        hits = query_hits.get(qid, [])
        label = labels.get(qid, None)
        family = families.get(qid, "UNKNOWN")

        if not hits:
            rows.append({
                "test_id": qid,
                "label": label,
                "family": family,
                "max_full_length_identity": None,
                "max_identity_train_id": None,
                "max_identity_qcov": None,
                "max_identity_tcov": None,
                "max_identity_bitscore": None,
                "max_identity_evalue": None,
                "best_bitscore_identity": None,
                "best_bitscore_train_id": None,
                "best_bitscore_qcov": None,
                "best_bitscore_tcov": None,
                "max_local_identity": None,
                "local_hit_train_id": None,
                "local_qcov": None,
                "local_tcov": None,
                "homology_class": "NO_FULL_LENGTH_HOMOLOG",
            })
            no_full_length_count += 1
            strict_remote_30_count += 1
            strict_remote_20_count += 1
            continue

        # Local hit (maximum fident regardless of coverage)
        best_local = max(hits, key=lambda h: (h["fident"], h["bitscore"]))

        # Full length hits (reciprocal coverage >= coverage_threshold)
        full_hits = [
            h for h in hits
            if h["qcov"] >= coverage_threshold and h["tcov"] >= coverage_threshold
        ]

        if full_hits:
            best_id_hit = max(full_hits, key=lambda h: (h["fident"], h["bitscore"]))
            best_bit_hit = max(full_hits, key=lambda h: (h["bitscore"], h["fident"]))

            max_fl_id = best_id_hit["fident"]
            max_fl_train_id = best_id_hit["target"]
            max_fl_qcov = best_id_hit["qcov"]
            max_fl_tcov = best_id_hit["tcov"]
            max_fl_bits = best_id_hit["bitscore"]
            max_fl_ev = best_id_hit["evalue"]

            bb_id = best_bit_hit["fident"]
            bb_train_id = best_bit_hit["target"]
            bb_qcov = best_bit_hit["qcov"]
            bb_tcov = best_bit_hit["tcov"]

            if max_fl_id >= 0.999 and max_fl_qcov >= 0.999 and max_fl_tcov >= 0.999:
                h_class = "L1_EXACT"
                l1_exact_count += 1
            elif max_fl_id >= identity_threshold:
                h_class = "L2_CLOSE"
                l2_close_count += 1
            elif max_fl_id < 0.20:
                h_class = "REMOTE_20_STRICT"
                strict_remote_20_count += 1
                strict_remote_30_count += 1
            else:
                h_class = "REMOTE_30_STRICT"
                strict_remote_30_count += 1
        else:
            max_fl_id = None
            max_fl_train_id = None
            max_fl_qcov = None
            max_fl_tcov = None
            max_fl_bits = None
            max_fl_ev = None

            bb_id = None
            bb_train_id = None
            bb_qcov = None
            bb_tcov = None

            if best_local["fident"] >= identity_threshold:
                h_class = "L3_DOMAIN_ONLY"
                l3_domain_only_count += 1
            else:
                h_class = "NO_FULL_LENGTH_HOMOLOG"
                no_full_length_count += 1

            # Even with domain hit or no hit, it has no full-length homolog >= 30%
            strict_remote_30_count += 1
            strict_remote_20_count += 1

        rows.append({
            "test_id": qid,
            "label": label,
            "family": family,
            "max_full_length_identity": max_fl_id,
            "max_identity_train_id": max_fl_train_id,
            "max_identity_qcov": max_fl_qcov,
            "max_identity_tcov": max_fl_tcov,
            "max_identity_bitscore": max_fl_bits,
            "max_identity_evalue": max_fl_ev,
            "best_bitscore_identity": bb_id,
            "best_bitscore_train_id": bb_train_id,
            "best_bitscore_qcov": bb_qcov,
            "best_bitscore_tcov": bb_tcov,
            "max_local_identity": best_local["fident"],
            "local_hit_train_id": best_local["target"],
            "local_qcov": best_local["qcov"],
            "local_tcov": best_local["tcov"],
            "homology_class": h_class,
        })

    homology_df = pl.DataFrame(rows)
    total_q = len(query_ids)

    summary = {
        "total_queries": total_q,
        "l1_exact_count": l1_exact_count,
        "l1_exact_pct": (l1_exact_count / total_q * 100) if total_q > 0 else 0.0,
        "l2_close_count": l2_close_count,
        "l2_close_pct": (l2_close_count / total_q * 100) if total_q > 0 else 0.0,
        "l3_domain_only_count": l3_domain_only_count,
        "l3_domain_only_pct": (l3_domain_only_count / total_q * 100) if total_q > 0 else 0.0,
        "strict_remote_30_count": strict_remote_30_count,
        "strict_remote_30_pct": (strict_remote_30_count / total_q * 100) if total_q > 0 else 0.0,
        "strict_remote_20_count": strict_remote_20_count,
        "strict_remote_20_pct": (strict_remote_20_count / total_q * 100) if total_q > 0 else 0.0,
        "no_full_length_count": no_full_length_count,
        "no_full_length_pct": (no_full_length_count / total_q * 100) if total_q > 0 else 0.0,
        "pass_strict_split": (l1_exact_count == 0 and l2_close_count == 0),
    }

    return homology_df, summary

