"""DeepISE Scientific Audit v1.1 - Steps 16, 17, 18: Build Stronger HMM Baselines.

Constructs three distinct levels of HMM baselines:
1. HMM-A: Whole-family HMM v2 (benchmark/db/deepise_tpases_v2.hmm)
   - One profile per IS family from train_positives (aligned via MAFFT FFT-NS-1).
2. HMM-B: Cluster-specific HMM (benchmark/db/deepise_cluster_tpases_v2.hmm)
   - MMseqs2 40% sub-clustering within each family, sub-cluster specific HMMs.
3. HMM-C: Catalytic-domain HMM (benchmark/db/deepise_domain_tpases_v2.hmm)
   - Domain-level profiles extracted from Pfam-A (Pfam 37.0 on local cluster mirror).
"""

import gzip
import os
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import polars as pl
from Bio import SeqIO

# Ensure conda env binaries are in PATH
ENV_BIN = Path("/hpcfs/fhome/caizhh/.conda/envs/DeepISE/bin")
if str(ENV_BIN) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{ENV_BIN}:{os.environ.get('PATH', '')}"


def build_hmm_a_whole_family(
    train_positives_parquet: Path,
    output_hmm: Path,
    tmp_dir: Path,
) -> Path:
    """Step 16: Build HMM-A (Whole-family HMM v2) from training positives."""
    print("=" * 70)
    print("STEP 16: Building HMM-A (Whole-family HMM v2)...")
    print("=" * 70)
    output_hmm.parent.mkdir(parents=True, exist_ok=True)
    work_dir = tmp_dir / "hmm_a"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(train_positives_parquet)
    fams = defaultdict(list)
    for row in df.iter_rows(named=True):
        fams[row["family"]].append(row)

    built_hmms = []
    for fam_name, members in sorted(fams.items()):
        clean_fam = "".join(c if c.isalnum() else "_" for c in fam_name)
        fam_faa = work_dir / f"{clean_fam}.faa"
        fam_aln = work_dir / f"{clean_fam}.aln"
        fam_hmm = work_dir / f"{clean_fam}.hmm"

        # Write FASTA
        with open(fam_faa, "w", encoding="utf-8") as f:
            for m in members:
                f.write(f">{m['tpase_id']}\n{m['protein_sequence']}\n")

        if len(members) == 1:
            cmd = ["hmmbuild", "-n", f"Tpase_{clean_fam}", str(fam_hmm), str(fam_faa)]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        else:
            # Fast progressive alignment via MAFFT FFT-NS-1
            cmd_mafft = f"mafft --retree 1 --maxiterate 0 --quiet {fam_faa} > {fam_aln}"
            subprocess.run(cmd_mafft, shell=True, check=True)
            cmd_hmmbuild = ["hmmbuild", "-n", f"Tpase_{clean_fam}", str(fam_hmm), str(fam_aln)]
            subprocess.run(cmd_hmmbuild, check=True, capture_output=True, text=True)

        built_hmms.append(fam_hmm)
        print(f"  [HMM-A] Family {fam_name} ({len(members)} seqs) -> {fam_hmm.name}")

    # Concatenate all family profiles
    with open(output_hmm, "w", encoding="utf-8") as out_f:
        for hmm_file in built_hmms:
            with open(hmm_file, "r", encoding="utf-8") as in_f:
                out_f.write(in_f.read())

    # Press HMM
    subprocess.run(["hmmpress", "-f", str(output_hmm)], check=True, capture_output=True, text=True)
    print(f"HMM-A successfully built: {len(built_hmms)} family profiles in {output_hmm}")
    return output_hmm


def build_hmm_b_cluster_specific(
    train_positives_parquet: Path,
    output_hmm: Path,
    tmp_dir: Path,
    min_identity: float = 0.40,
    coverage: float = 0.80,
) -> Path:
    """Step 17: Build HMM-B (Cluster-specific HMM) from 40% sub-clusters."""
    print("=" * 70)
    print("STEP 17: Building HMM-B (Cluster-specific HMM)...")
    print("=" * 70)
    output_hmm.parent.mkdir(parents=True, exist_ok=True)
    work_dir = tmp_dir / "hmm_b"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(train_positives_parquet)
    fams = defaultdict(list)
    for row in df.iter_rows(named=True):
        fams[row["family"]].append(row)

    all_cluster_hmms = []
    total_clusters = 0

    for fam_name, members in sorted(fams.items()):
        clean_fam = "".join(c if c.isalnum() else "_" for c in fam_name)
        fam_dir = work_dir / clean_fam
        fam_dir.mkdir(parents=True, exist_ok=True)

        fam_faa = fam_dir / f"{clean_fam}_all.faa"
        with open(fam_faa, "w", encoding="utf-8") as f:
            for m in members:
                f.write(f">{m['tpase_id']}\n{m['protein_sequence']}\n")

        if len(members) <= 2:
            fam_hmm = fam_dir / f"{clean_fam}_c0.hmm"
            if len(members) == 1:
                subprocess.run(
                    ["hmmbuild", "-n", f"Tpase_{clean_fam}_c0", str(fam_hmm), str(fam_faa)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            else:
                fam_aln = fam_dir / f"{clean_fam}_c0.aln"
                subprocess.run(f"mafft --retree 1 --maxiterate 0 --quiet {fam_faa} > {fam_aln}", shell=True, check=True)
                subprocess.run(
                    ["hmmbuild", "-n", f"Tpase_{clean_fam}_c0", str(fam_hmm), str(fam_aln)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            all_cluster_hmms.append(fam_hmm)
            total_clusters += 1
            continue

        # Run MMseqs2 cluster at 40% identity
        mmseqs_prefix = fam_dir / "mmseqs_clu"
        cmd_clu = [
            "mmseqs",
            "easy-cluster",
            str(fam_faa),
            str(mmseqs_prefix),
            str(fam_dir / "mmseqs_tmp"),
            "--min-seq-id", str(min_identity),
            "-c", str(coverage),
            "--cov-mode", "0",
            "-v", "0",
        ]
        subprocess.run(cmd_clu, check=True)

        # Parse clusters: cluster.tsv has rep_id \t member_id
        cluster_tsv = fam_dir / "mmseqs_clu_cluster.tsv"
        clusters = defaultdict(list)
        with open(cluster_tsv, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    clusters[parts[0]].append(parts[1])

        seq_dict = {m["tpase_id"]: m["protein_sequence"] for m in members}

        c_idx = 0
        for rep_id, member_ids in clusters.items():
            sub_faa = fam_dir / f"sub_{c_idx}.faa"
            sub_aln = fam_dir / f"sub_{c_idx}.aln"
            sub_hmm = fam_dir / f"sub_{c_idx}.hmm"
            model_name = f"Tpase_{clean_fam}_c{c_idx}"

            with open(sub_faa, "w", encoding="utf-8") as f:
                for mid in member_ids:
                    f.write(f">{mid}\n{seq_dict[mid]}\n")

            if len(member_ids) == 1:
                subprocess.run(
                    ["hmmbuild", "-n", model_name, str(sub_hmm), str(sub_faa)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            else:
                subprocess.run(
                    f"mafft --retree 1 --maxiterate 0 --quiet {sub_faa} > {sub_aln}",
                    shell=True,
                    check=True,
                )
                subprocess.run(
                    ["hmmbuild", "-n", model_name, str(sub_hmm), str(sub_aln)],
                    check=True,
                    capture_output=True,
                    text=True,
                )

            all_cluster_hmms.append(sub_hmm)
            c_idx += 1
            total_clusters += 1

        print(f"  [HMM-B] Family {fam_name} ({len(members)} seqs) -> {len(clusters)} sub-clusters")

    # Concatenate all cluster profiles
    with open(output_hmm, "w", encoding="utf-8") as out_f:
        for hmm_file in all_cluster_hmms:
            with open(hmm_file, "r", encoding="utf-8") as in_f:
                out_f.write(in_f.read())

    # Press HMM
    subprocess.run(["hmmpress", "-f", str(output_hmm)], check=True, capture_output=True, text=True)
    print(f"HMM-B successfully built: {total_clusters} sub-cluster profiles in {output_hmm}")
    return output_hmm


def build_hmm_c_catalytic_domain(
    pfam_gz: Path,
    output_hmm: Path,
) -> Path:
    """Step 18: Build HMM-C (Catalytic-domain HMM) from Pfam mirror."""
    print("=" * 70)
    print("STEP 18: Building HMM-C (Catalytic-domain HMM)...")
    print("=" * 70)
    output_hmm.parent.mkdir(parents=True, exist_ok=True)

    include_keywords = [
        "transposase", "transposon", "integrase", "recombinase",
        "resolvase", "dde endonuclease", "dde domain", "tnpb",
        "tnpa", "rep_trans", "is110", "is200", "is605", "is91",
    ]

    exclude_keywords = [
        "bladder cancer", "centromere", "cysteine cluster", "splicing",
        "peptidase", "capsid", "gag", "transporting motif", "plant transposon",
    ]

    extracted_profiles = 0
    cur_model = []
    cur_name = ""
    cur_desc = ""

    with open(output_hmm, "w", encoding="utf-8") as out_f:
        with gzip.open(pfam_gz, "rt", errors="replace") as in_f:
            for line in in_f:
                cur_model.append(line)
                if line.startswith("NAME  "):
                    cur_name = line.strip().split()[1]
                elif line.startswith("DESC  "):
                    cur_desc = line[6:].strip()
                elif line.startswith("//"):
                    text = f"{cur_name} {cur_desc}".lower()
                    has_inc = any(k in text for k in include_keywords)
                    has_exc = any(k in text for k in exclude_keywords)
                    if has_inc and not has_exc:
                        out_f.writelines(cur_model)
                        extracted_profiles += 1
                    cur_model = []
                    cur_name = ""
                    cur_desc = ""

    # Press HMM
    subprocess.run(["hmmpress", "-f", str(output_hmm)], check=True, capture_output=True, text=True)
    print(f"HMM-C successfully built: {extracted_profiles} catalytic domain profiles in {output_hmm}")
    return output_hmm


def main():
    root = Path("/hpcfs/fhome/caizhh/Desktop/03_Tool_Development/05_DeepISE")
    train_parquet = root / "data/splits/cluster30_v2/train_positives.parquet"
    db_dir = root / "benchmark/db"
    db_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(tempfile.mkdtemp(prefix="deepise_hmm_fast_"))
    try:
        t0 = time.time()
        # 1. HMM-A
        hmm_a = db_dir / "deepise_tpases_v2.hmm"
        build_hmm_a_whole_family(train_parquet, hmm_a, tmp_dir)

        # 2. HMM-B
        hmm_b = db_dir / "deepise_cluster_tpases_v2.hmm"
        build_hmm_b_cluster_specific(train_parquet, hmm_b, tmp_dir)

        # 3. HMM-C
        pfam_gz = Path("/hpcfs/fpublic/database/pfam/releases/Pfam37.0/Pfam-A.hmm.gz")
        hmm_c = db_dir / "deepise_domain_tpases_v2.hmm"
        build_hmm_c_catalytic_domain(pfam_gz, hmm_c)

        elapsed = time.time() - t0
        print(f"\nAll three HMM baselines successfully constructed in {elapsed:.2f} seconds!")
        print(f"  HMM-A: {hmm_a} ({hmm_a.stat().st_size / 1e6:.2f} MB)")
        print(f"  HMM-B: {hmm_b} ({hmm_b.stat().st_size / 1e6:.2f} MB)")
        print(f"  HMM-C: {hmm_c} ({hmm_c.stat().st_size / 1e6:.2f} MB)")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
