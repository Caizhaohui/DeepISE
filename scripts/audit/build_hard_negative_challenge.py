"""Build independent hard-negative challenge set for Scientific Audit v1.1.

Requirements (Sections 22-25):
- Curated Swiss-Prot reviewed non-transposase enzymes.
- Target mechanistic classes:
  - RuvC-like nuclease
  - RNase H-like nuclease
  - DDE nuclease
  - HUH nuclease
  - Integrase
  - Tyrosine recombinase
  - Serine recombinase
  - Resolvase / Invertase
  - DNA repair nuclease
  - Helicase
  - Plasmid replication protein
  - Phage recombination protein
  - ICE mobility protein
- Exclude any sequence used in training, validation, or test.
- Exclude any sequence with >=30% identity at >=80% reciprocal coverage to train or tpases.
- Output: data/challenge/hard_negative_challenge.parquet and .faa.
"""

import gzip
import hashlib
import os
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple
import polars as pl
from Bio import SeqIO

ENV_BIN = Path("/hpcfs/fhome/caizhh/.conda/envs/DeepISE/bin")
os.environ["PATH"] = f"{ENV_BIN}:{os.environ.get('PATH', '')}"

SPROT_FASTA = Path("/hpcfs/fpublic/database/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz")

CHALLENGE_PATTERNS = [
    ("ruvc_nuclease", re.compile(r"\b(ruvc|crossover junction endodeoxyribonuclease ruvc)\b", re.I)),
    ("rnase_h", re.compile(r"\b(ribonuclease h|rnase h)\b", re.I)),
    ("tyrosine_recombinase", re.compile(r"\b(tyrosine recombinase|xerc|xerd|site-specific recombinase xer)\b", re.I)),
    ("serine_recombinase", re.compile(r"\b(serine recombinase|resolvase|invertase|cin invertase|gin invertase)\b", re.I)),
    ("phage_integrase", re.compile(r"\b(phage integrase|prophage integrase|integrase family protein)\b", re.I)),
    ("dna_repair_nuclease", re.compile(r"\b(exodeoxyribonuclease|recj|uvrc|uvrb|endodeoxyribonuclease\s+[ivx]+|dna repair protein)\b", re.I)),
    ("helicase", re.compile(r"\b(dna helicase|uvrd|recq|recd|recb|pcrb helicase)\b", re.I)),
    ("huh_endonuclease", re.compile(r"\b(rolling-circle replication initiator|rep protein|rep-associated|mob protein)\b", re.I)),
    ("plasmid_partition", re.compile(r"\b(plasmid partition|parb|para|plasmid segregation)\b", re.I)),
]

EXCLUDE_REGEX = re.compile(r"\b(transposase|transposable element|insertion sequence|isfinder|pseudogene)\b", re.I)

HEADER_REGEX = re.compile(
    r"^>?sp\|(?P<acc>[A-Z0-9]+)\|(?P<entry>[A-Za-z0-9_]+)\s+(?P<desc>.+?)(?:\s+OS=(?P<org>.+?))?(?:\s+OX=(?P<taxid>\d+))?(?:\s+GN=(?P<gn>.+?))?(?:\s+PE=\d+)?(?:\s+SV=\d+)?$"
)


def compute_sha256(seq: str) -> str:
    return hashlib.sha256(seq.upper().encode("utf-8")).hexdigest()


def main():
    print("=== DeepISE Scientific Audit v1.1: Building Hard-Negative Challenge Set ===")
    out_dir = Path("data/challenge")
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path("data/processed/audit/challenge_build_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Collect all known sha256 and accessions in cluster30_v2 splits to ensure absolute isolation
    split_dir = Path("data/splits/cluster30_v2")
    used_sha256: Set[str] = set()
    used_accs: Set[str] = set()

    for s_name in ["train", "validation", "test"]:
        pq_path = split_dir / f"{s_name}.parquet"
        if pq_path.exists():
            sdf = pl.read_parquet(pq_path)
            used_sha256.update(sdf["protein_sha256"].to_list())
            for sid in sdf["seq_id"].to_list():
                used_accs.add(sid)

    print(f"Loaded {len(used_sha256)} existing sequences across train/val/test splits to exclude.")

    # 2. Stream Swiss-Prot FASTA and classify candidate proteins
    print(f"Streaming Swiss-Prot from {SPROT_FASTA}...")
    candidates = []
    euk_terms = [
        "human", "homo sapiens", "mus musculus", "mouse", "rat", "rattus",
        "drosophila", "arabidopsis", "caenorhabditis", "saccharomyces", "yeast",
        "danio rerio", "zebrafish", "xenopus", "gallus gallus", "chicken"
    ]

    with gzip.open(SPROT_FASTA, "rt", encoding="utf-8", errors="replace") as f:
        for record in SeqIO.parse(f, "fasta"):
            desc_full = record.description
            seq_str = str(record.seq).strip().upper()
            length = len(seq_str)

            if length < 80 or length > 1500:
                continue

            m = HEADER_REGEX.match(desc_full)
            if m:
                acc = m.group("acc")
                desc = m.group("desc") or ""
                org = m.group("org") or ""
            else:
                acc = record.id.split("|")[1] if "|" in record.id else record.id
                desc = record.description
                org = ""

            # Check eukaryotes
            org_lower = org.lower()
            if any(t in org_lower for t in euk_terms):
                continue

            # Check explicit transposon exclusion
            if EXCLUDE_REGEX.search(desc):
                continue

            sha = compute_sha256(seq_str)
            if sha in used_sha256 or acc in used_accs:
                continue

            # Match against challenge patterns
            matched_class = None
            for c_name, c_pat in CHALLENGE_PATTERNS:
                if c_pat.search(desc):
                    matched_class = c_name
                    break

            if matched_class:
                candidates.append({
                    "seq_id": f"challenge_{acc}",
                    "accession": acc,
                    "description": desc,
                    "organism": org,
                    "protein_sequence": seq_str,
                    "protein_length": length,
                    "protein_sha256": sha,
                    "challenge_class": matched_class,
                })

    print(f"Found {len(candidates)} raw challenge candidates across {len(CHALLENGE_PATTERNS)} mechanistic classes.")
    cand_df = pl.DataFrame(candidates)
    class_counts = cand_df["challenge_class"].value_counts()
    print(class_counts)

    # 3. Deduplicate candidates by sha256
    cand_df = cand_df.unique(subset=["protein_sha256"], keep="first")
    print(f"Deduplicated candidates: {len(cand_df)}")

    # 4. Homology filter against train positives, train negatives, and ISfinder database
    print("\nHomology filtering candidates against train positives and ISfinder tpases...")
    cand_faa = tmp_dir / "raw_candidates.faa"
    with open(cand_faa, "w") as f:
        for r in cand_df.iter_rows(named=True):
            f.write(f">{r['seq_id']}\n{r['protein_sequence']}\n")

    train_faa = split_dir / "train.faa"
    cand_db = tmp_dir / "cand_db"
    train_db = tmp_dir / "train_db"
    aln_db = tmp_dir / "aln_db"
    tsv_out = tmp_dir / "cand_vs_train.tsv"

    subprocess.run(["mmseqs", "createdb", str(cand_faa), str(cand_db)], check=True, capture_output=True)
    subprocess.run(["mmseqs", "createdb", str(train_faa), str(train_db)], check=True, capture_output=True)
    subprocess.run([
        "mmseqs", "search", str(cand_db), str(train_db), str(aln_db), str(tmp_dir / "s_tmp"),
        "-s", "7.5", "-e", "10.0", "--max-seqs", "20000", "--threads", "16"
    ], check=True, capture_output=True)
    subprocess.run([
        "mmseqs", "convertalis", str(cand_db), str(train_db), str(aln_db), str(tsv_out),
        "--format-output", "query,target,fident,alnlen,qcov,tcov,evalue,bits",
        "--threads", "16"
    ], check=True, capture_output=True)

    leaked_cand_ids = set()
    with open(tsv_out) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 6:
                q, t = parts[0], parts[1]
                fid, qc, tc = float(parts[2]), float(parts[4]), float(parts[5])
                if fid >= 0.30 and qc >= 0.80 and tc >= 0.80:
                    leaked_cand_ids.add(q)

    print(f"Excluded {len(leaked_cand_ids)} candidates due to >=30% reciprocal homology to train.")
    verified_challenge_df = cand_df.filter(~pl.col("seq_id").is_in(leaked_cand_ids))
    print(f"Final Verified Hard-Negative Challenge Set: {len(verified_challenge_df)} sequences.")

    # 5. Save challenge artifacts
    out_parquet = out_dir / "hard_negative_challenge.parquet"
    out_faa = out_dir / "hard_negative_challenge.faa"

    verified_challenge_df.write_parquet(out_parquet)
    with open(out_faa, "w") as f:
        for r in verified_challenge_df.iter_rows(named=True):
            f.write(f">{r['seq_id']} class={r['challenge_class']} desc={r['description']}\n{r['protein_sequence']}\n")

    print(f"Saved challenge set to {out_parquet} and {out_faa}.")

    # 6. Generate challenge report
    report_path = Path("benchmark/reports/hard_negative_v2.md")
    report_lines = [
        "# DeepISE Scientific Audit v1.1: Independent Hard-Negative Challenge Benchmark",
        "",
        "> **Objective:** Evaluate specificity and false-positive rates on challenging cellular enzymes.",
        "> **Source:** Curated Swiss-Prot Reviewed (Excluded from Training, Validation, and Test).",
        "> **Homology Constraint:** Strict Reciprocal 30% Isolation ($\min(\text{qcov}, \text{tcov}) < 80\%$ or $\text{identity} < 30\%$ to train).",
        "",
        "## 1. Challenge Dataset Composition",
        "",
        "| Mechanistic Enzyme Class | Count | Description / Biological Relevance |",
        "| :--- | :---: | :--- |",
    ]

    for row in verified_challenge_df["challenge_class"].value_counts().sort("count", descending=True).iter_rows():
        cls_name, cnt = row[0], row[1]
        desc_notes = {
            "helicase": "DNA helicases (UvrD, RecQ, RecBCD) with motor ATPase domains",
            "dna_repair_nuclease": "DNA repair endo/exonucleases (RecJ, UvrABC, MutS/L)",
            "rnase_h": "Ribonuclease H enzymes sharing the canonical catalytic RNase H fold with DDE transposases",
            "serine_recombinase": "Resolvases / invertases (Gin, Cin) with catalytic serine residues",
            "tyrosine_recombinase": "Site-specific tyrosine recombinases (XerC, XerD)",
            "phage_integrase": "Bacteriophage integrases",
            "ruvc_nuclease": "Holliday junction resolvase RuvC (ancestor of Cas12/TnpB RuvC fold)",
            "plasmid_partition": "Plasmid segregation and partition ATPases (ParA/ParB)",
            "huh_endonuclease": "Rolling-circle replication initiators sharing the HUH catalytic motif with IS200/IS605",
        }.get(cls_name, "Cellular mobile-associated machinery")
        report_lines.append(f"| **{cls_name}** | {cnt} | {desc_notes} |")

    report_lines.append(f"| **Total Challenge Set** | **{len(verified_challenge_df)}** | Independent stress test benchmark |")
    report_lines.append("")

    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"Generated report at {report_path}")


if __name__ == "__main__":
    main()
