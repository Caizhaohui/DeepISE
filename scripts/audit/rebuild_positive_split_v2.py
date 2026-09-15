"""Rebuild positive 30% homology-disjoint split (cluster30_v2) using reciprocal full-length criteria.

Steps:
1. Load deduplicated positive transposases (data/processed/tpases.parquet, n=7,057).
2. Run MMseqs2 all-vs-all search with high sensitivity and permissive e-value.
3. Construct connected components (Union-Find) with edges defined as:
   identity >= 0.30 AND qcov >= 0.80 AND tcov >= 0.80 (reciprocal coverage >= 80%).
4. Partition clusters into train (70%), validation (15%), test (15%) using cluster_aware_split.
5. Verify zero cross-split violations:
   train vs test, train vs val, val vs test.
6. Save to data/splits/cluster30_v2/ with fastas and manifest.yaml.
"""

import os
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import polars as pl
from Bio import SeqIO

from deepise_ml.dataset.split import cluster_aware_split

ENV_BIN = Path("/hpcfs/fhome/caizhh/.conda/envs/DeepISE/bin")
os.environ["PATH"] = f"{ENV_BIN}:{os.environ.get('PATH', '')}"


def build_reciprocal_clusters(
    tpases_faa: Path,
    tmp_dir: Path,
    min_identity: float = 0.30,
    coverage: float = 0.80,
    threads: int = 16,
) -> Tuple[Dict[str, str], pl.DataFrame]:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    db_path = tmp_dir / "tpase_db"
    aln_db = tmp_dir / "aln_db"
    search_tmp = tmp_dir / "search_tmp"
    search_tsv = tmp_dir / "all_vs_all.tsv"

    if not search_tsv.exists() or search_tsv.stat().st_size == 0:
        print("  [1/4] Running MMseqs2 createdb...")
        subprocess.run(["mmseqs", "createdb", str(tpases_faa), str(db_path)], check=True, capture_output=True)

        print("  [2/4] Running MMseqs2 all-vs-all search (s=7.5, -e 10.0)...")
        subprocess.run([
            "mmseqs", "search",
            str(db_path), str(db_path),
            str(aln_db), str(search_tmp),
            "-s", "7.5",
            "-e", "10.0",
            "--threads", str(threads),
        ], check=True, capture_output=True)

        print("  [3/4] Converting alignments...")
        subprocess.run([
            "mmseqs", "convertalis",
            str(db_path), str(db_path),
            str(aln_db), str(search_tsv),
            "--format-output", "query,target,fident,alnlen,qcov,tcov,evalue,bits",
            "--threads", str(threads),
        ], check=True, capture_output=True)
    else:
        print("  [1-3/4] Found existing all_vs_all.tsv, loading directly...")

    records = list(SeqIO.parse(tpases_faa, "fasta"))
    all_sids = [r.id for r in records]
    seq_lengths = {r.id: len(r.seq) for r in records}

    parent = {sid: sid for sid in all_sids}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    print("  [4/4] Building connected components with reciprocal coverage >= 0.80 and identity >= 0.30...")
    edge_count = 0
    with open(search_tsv, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 6:
                q, t = parts[0], parts[1]
                if q == t:
                    continue
                fid = float(parts[2])
                qcov = float(parts[4])
                tcov = float(parts[5])
                # Strict reciprocal coverage condition
                if fid >= min_identity and qcov >= coverage and tcov >= coverage:
                    union(q, t)
                    edge_count += 1

    print(f"    Reciprocal edges: {edge_count}")

    clus = defaultdict(list)
    for sid in all_sids:
        clus[find(sid)].append(sid)

    # Sort clusters by size descending
    sorted_clusters = sorted(clus.values(), key=len, reverse=True)
    cluster_mapping = {}
    cluster_stats = []

    for idx, members in enumerate(sorted_clusters):
        cid = f"cluster30_v2_{idx:04d}"
        rep = max(members, key=lambda m: (seq_lengths.get(m, 0), m))
        for mem in members:
            cluster_mapping[mem] = cid
        cluster_stats.append({
            "cluster30_id": cid,
            "representative": rep,
            "size": len(members),
        })

    print(f"    Total clusters: {len(sorted_clusters)}")
    sizes = [len(m) for m in sorted_clusters]
    singletons = sum(1 for s in sizes if s == 1)
    print(f"    Singletons: {singletons} ({singletons / len(sizes) * 100:.1f}%)")
    print(f"    Largest cluster: {max(sizes)}, Median: {int(np.median(sizes))}")

    stats_df = pl.DataFrame(cluster_stats)
    return parent, all_sids, cluster_mapping, stats_df


def main():
    print("=== DeepISE Scientific Audit v1.1: Rebuilding Positive Split (cluster30_v2) ===")
    tpases_parquet = Path("data/processed/tpases.parquet")
    tpases_faa = Path("data/processed/tpases.faa")
    output_dir = Path("data/splits/cluster30_v2")
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path("data/processed/audit/positive_split_tmp")

    df = pl.read_parquet(tpases_parquet)
    print(f"Loaded {len(df)} positive transposases.")

    # 1. Build reciprocal clusters
    parent, all_sids, cluster_map, stats_df = build_reciprocal_clusters(
        tpases_faa,
        tmp_dir=tmp_dir,
        min_identity=0.30,
        coverage=0.80,
        threads=16,
    )

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    # 2. Iterative cluster-aware split with automatic edge closure
    print("\nSplitting clusters into train (70%), val (15%), test (15%) with iterative edge closure...")
    check_tmp = tmp_dir / "check_homology"
    check_tmp.mkdir(parents=True, exist_ok=True)
    seq_dict = {r.id: str(r.seq) for r in SeqIO.parse(tpases_faa, "fasta")}

    iteration = 0
    while True:
        iteration += 1
        print(f"\n--- Split Iteration {iteration} ---")
        # Map current cluster IDs
        clus = defaultdict(list)
        for sid in all_sids:
            clus[find(sid)].append(sid)

        sorted_clusters = sorted(clus.values(), key=len, reverse=True)
        cluster_map = {}
        for idx, members in enumerate(sorted_clusters):
            cid = f"cluster30_v2_{idx:04d}"
            for mem in members:
                cluster_map[mem] = cid

        df = df.with_columns(
            pl.col("tpase_id").replace_strict(cluster_map).alias("cluster30_id")
        )

        train_df, val_df, test_df, manifest = cluster_aware_split(
            df,
            ratios=(0.70, 0.15, 0.15),
            seed=42,
        )

        print(f"  Train: {len(train_df)} seqs ({manifest['clusters']['train']} clusters)")
        print(f"  Validation: {len(val_df)} seqs ({manifest['clusters']['validation']} clusters)")
        print(f"  Test: {len(test_df)} seqs ({manifest['clusters']['test']} clusters)")

        # Write fastas for checking
        for name, split_df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
            faa_out = output_dir / f"{name}_positives.faa"
            with open(faa_out, "w", encoding="utf-8") as f:
                for row in split_df.iter_rows(named=True):
                    sid = row["tpase_id"]
                    s = seq_dict.get(sid, row["protein_sequence"])
                    f.write(f">{sid} family={row['family']}\n{s}\n")

        # Check all cross-split pairs
        all_violations = []
        for q_name, t_name in [("test", "train"), ("validation", "train"), ("test", "validation")]:
            pair_tmp = check_tmp / f"{q_name}_{t_name}"
            if pair_tmp.exists():
                shutil.rmtree(pair_tmp)
            pair_tmp.mkdir(parents=True, exist_ok=True)

            q_faa = output_dir / f"{q_name}_positives.faa"
            t_faa = output_dir / f"{t_name}_positives.faa"
            q_db = pair_tmp / f"{q_name}_db"
            t_db = pair_tmp / f"{t_name}_db"
            a_db = pair_tmp / f"{q_name}_vs_{t_name}_db"
            out_tsv = pair_tmp / f"{q_name}_vs_{t_name}.tsv"

            subprocess.run(["mmseqs", "createdb", str(q_faa), str(q_db)], check=True, capture_output=True)
            subprocess.run(["mmseqs", "createdb", str(t_faa), str(t_db)], check=True, capture_output=True)
            subprocess.run([
                "mmseqs", "search", str(q_db), str(t_db), str(a_db), str(pair_tmp / "s_tmp"),
                "-s", "7.5", "-e", "10.0", "--max-seqs", "20000", "--threads", "16"
            ], check=True, capture_output=True)
            subprocess.run([
                "mmseqs", "convertalis", str(q_db), str(t_db), str(a_db), str(out_tsv),
                "--format-output", "query,target,fident,alnlen,qcov,tcov,evalue,bits",
                "--threads", "16"
            ], check=True, capture_output=True)

            with open(out_tsv) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 6:
                        q, t = parts[0], parts[1]
                        fid, qc, tc = float(parts[2]), float(parts[4]), float(parts[5])
                        if fid >= 0.30 and qc >= 0.80 and tc >= 0.80:
                            all_violations.append((q, t, fid, qc, tc))

        print(f"  Total cross-split reciprocal violations: {len(all_violations)}")
        if len(all_violations) == 0:
            print("  ✅ ZERO cross-split violations confirmed!")
            break

        print(f"  Merging {len(all_violations)} cross-split violating pairs into same clusters and re-splitting...")
        for q, t, fid, qc, tc in all_violations:
            union(q, t)

    # Save final Parquet files
    train_df.write_parquet(output_dir / "train_positives.parquet")
    val_df.write_parquet(output_dir / "validation_positives.parquet")
    test_df.write_parquet(output_dir / "test_positives.parquet")

    sizes = [len(m) for m in sorted_clusters]
    singletons = sum(1 for s in sizes if s == 1)
    stats_df = pl.DataFrame([
        {"cluster30_id": f"cluster30_v2_{idx:04d}", "size": len(members)}
        for idx, members in enumerate(sorted_clusters)
    ])

    # Write positive split report
    report_path = Path("benchmark/reports/positive_split_audit_v2.md")
    report_lines = [
        "# DeepISE Scientific Audit v1.1: Positive Homology Split Audit v2",
        "",
        "> **Dataset:** ISfinder Deduplicated Positives ($N=7,057$)",
        "> **Clustering Algorithm:** Reciprocal Full-Length Connected Components (`identity >= 0.30`, `min(qcov, tcov) >= 0.80`)",
        "> **Split Strategy:** Cluster-atomic Group Split (70% Train, 15% Validation, 15% Test)",
        "",
        "## 1. Cluster & Split Summary",
        "",
        "| Split | Sequence Count | Percentage | Cluster Count | Singleton Clusters | Largest Cluster | Median Size |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **Train** | {len(train_df)} | {len(train_df)/len(df)*100:.1f}% | {manifest['clusters']['train']} | - | - | - |",
        f"| **Validation** | {len(val_df)} | {len(val_df)/len(df)*100:.1f}% | {manifest['clusters']['validation']} | - | - | - |",
        f"| **Test** | {len(test_df)} | {len(test_df)/len(df)*100:.1f}% | {manifest['clusters']['test']} | - | - | - |",
        f"| **Total** | {len(df)} | 100.0% | {len(stats_df)} | {singletons} ({singletons/len(sizes)*100:.1f}%) | {max(sizes)} | {int(np.median(sizes))} |",
        "",
        "## 2. Cross-Split Homology Verification",
        "",
        "| Comparison Pair | Identity Threshold | Coverage Threshold | Verified Violations | Integrity Status |",
        "| :--- | :---: | :---: | :---: | :---: |",
        "| **Test vs Train** | $\ge 30\%$ | $\ge 80\%$ reciprocal | **0** | ✅ Strictly Disjoint |",
        "| **Validation vs Train** | $\ge 30\%$ | $\ge 80\%$ reciprocal | **0** | ✅ Strictly Disjoint |",
        "| **Test vs Validation** | $\ge 30\%$ | $\ge 80\%$ reciprocal | **0** | ✅ Strictly Disjoint |",
        "",
        "## 3. Scientific Audit Conclusion",
        "Under the strict reciprocal full-length criteria ($\min(\\text{qcov}, \\text{tcov}) \\ge 80\\%$ at $\\text{identity} \\ge 30\\%$), the newly generated `cluster30_v2` split achieves **mathematically guaranteed zero cross-split homology leakage**.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"Generated audit report at {report_path}")


if __name__ == "__main__":
    from typing import Dict, List, Tuple
    main()
