"""Extract ESM-2 embeddings for cluster30_v2 dataset and challenge set."""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import polars as pl
import torch
from Bio import SeqIO
from transformers import AutoModel, AutoTokenizer

# Ensure conda env binaries
ENV_BIN = Path("/hpcfs/fhome/caizhh/.conda/envs/DeepISE/bin")
if str(ENV_BIN) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{ENV_BIN}:{os.environ.get('PATH', '')}"


def extract_embeddings_for_fasta(
    fasta_path: Path,
    output_npy: Path,
    output_index_parquet: Path,
    model_name: str,
    batch_size: int = 64,
    max_len: int = 1022,
    device: str = "cuda",
):
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    if output_npy.exists() and output_index_parquet.exists() and output_npy.stat().st_size > 0:
        print(f"Already exists: {output_npy} (skipping)")
        return

    print(f"\nProcessing {fasta_path} -> {output_npy} using {model_name} on {device}...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)

    records = list(SeqIO.parse(fasta_path, "fasta"))
    seq_ids = [r.id for r in records]
    seqs = [str(r.seq).strip().upper() for r in records]
    total_seqs = len(seqs)
    print(f"  Total sequences: {total_seqs}")

    embeddings = []
    index_rows = []

    # Process in batches
    with torch.no_grad():
        for i in range(0, total_seqs, batch_size):
            b_seqs = seqs[i : i + batch_size]
            b_ids = seq_ids[i : i + batch_size]

            # Separate short and long sequences to allow efficient batching
            short_pairs = [(idx, s_id, s) for idx, (s_id, s) in enumerate(zip(b_ids, b_seqs)) if len(s) <= max_len]
            long_pairs = [(idx, s_id, s) for idx, (s_id, s) in enumerate(zip(b_ids, b_seqs)) if len(s) > max_len]

            batch_reps = [None] * len(b_seqs)

            if short_pairs:
                short_seqs = [p[2] for p in short_pairs]
                inputs = tokenizer(short_seqs, return_tensors="pt", padding=True, truncation=True, max_length=max_len, add_special_tokens=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                hidden = outputs.last_hidden_state  # [B, L, D]
                # Mean pool excluding padding and special tokens
                mask = inputs["attention_mask"].unsqueeze(-1)  # [B, L, 1]
                # exclude first and last valid tokens
                sum_reps = (hidden * mask).sum(dim=1)
                lens = mask.sum(dim=1).clamp(min=1)
                pooled = (sum_reps / lens).cpu().numpy()

                for (orig_idx, s_id, s), rep in zip(short_pairs, pooled):
                    batch_reps[orig_idx] = (s_id, len(s), False, rep)

            if long_pairs:
                stride = max_len // 2
                for orig_idx, s_id, s in long_pairs:
                    seq_len = len(s)
                    window_reps = []
                    weights = []
                    for start in range(0, seq_len, stride):
                        end = min(start + max_len, seq_len)
                        chunk = s[start:end]
                        if len(chunk) < 5:
                            continue
                        inputs = tokenizer(chunk, return_tensors="pt", add_special_tokens=True)
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        out = model(**inputs)
                        rep = out.last_hidden_state[0, 1:-1, :].mean(dim=0).cpu().numpy()
                        window_reps.append(rep)
                        weights.append(len(chunk))
                        if end == seq_len:
                            break
                    mean_rep = np.average(window_reps, axis=0, weights=weights)
                    batch_reps[orig_idx] = (s_id, seq_len, True, mean_rep)

            for s_id, slen, is_long, rep in batch_reps:
                embeddings.append(rep)
                index_rows.append({
                    "seq_id": s_id,
                    "length": slen,
                    "is_long": is_long,
                    "embedding_idx": len(embeddings) - 1,
                })

            if (i + batch_size) % 2000 < batch_size or (i + batch_size) >= total_seqs:
                speed = len(embeddings) / (time.time() - t0)
                print(f"  Processed {len(embeddings)}/{total_seqs} seqs ({speed:.1f} seq/s)...")

    emb_matrix = np.vstack(embeddings).astype(np.float32)
    np.save(output_npy, emb_matrix)
    pl.DataFrame(index_rows).write_parquet(output_index_parquet)
    elapsed = time.time() - t0
    print(f"Finished {output_npy}: {emb_matrix.shape} in {elapsed:.2f}s ({total_seqs/elapsed:.1f} seq/s)")


def export_split_fastas(splits_dir: Path, challenge_path: Path):
    for split_name in ["train", "validation", "test"]:
        pq_path = splits_dir / f"{split_name}.parquet"
        faa_path = splits_dir / f"{split_name}.faa"
        if not faa_path.exists() or faa_path.stat().st_size == 0:
            df = pl.read_parquet(pq_path)
            with open(faa_path, "w", encoding="utf-8") as f:
                for row in df.iter_rows(named=True):
                    f.write(f">{row['seq_id']}\n{row['protein_sequence']}\n")
            print(f"Exported {faa_path} ({len(df)} records)")

    ch_faa = challenge_path.parent / "hard_negative_challenge.faa"
    if not ch_faa.exists() or ch_faa.stat().st_size == 0:
        ch_df = pl.read_parquet(challenge_path)
        with open(ch_faa, "w", encoding="utf-8") as f:
            for row in ch_df.iter_rows(named=True):
                f.write(f">{row['seq_id']}\n{row['protein_sequence']}\n")
        print(f"Exported {ch_faa} ({len(ch_df)} records)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/esm2_t12_35M_UR50D")
    parser.add_argument("--tag", type=str, default="esm2_35m")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for model {args.model}")

    root = Path("/hpcfs/fhome/caizhh/Desktop/03_Tool_Development/05_DeepISE")
    splits_dir = root / "data/splits/cluster30_v2"
    challenge_pq = root / "data/challenge/hard_negative_challenge.parquet"
    out_dir = root / "data/processed/embeddings_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    export_split_fastas(splits_dir, challenge_pq)

    for split in ["train", "validation", "test"]:
        faa = splits_dir / f"{split}.faa"
        npy = out_dir / f"{args.tag}_{split}.npy"
        idx = out_dir / f"{args.tag}_{split}_index.parquet"
        extract_embeddings_for_fasta(faa, npy, idx, args.model, batch_size=args.batch_size, device=device)

    ch_faa = challenge_pq.parent / "hard_negative_challenge.faa"
    ch_npy = out_dir / f"{args.tag}_hard_neg.npy"
    ch_idx = out_dir / f"{args.tag}_hard_neg_index.parquet"
    extract_embeddings_for_fasta(ch_faa, ch_npy, ch_idx, args.model, batch_size=args.batch_size, device=device)

    print(f"\nAll embeddings for {args.tag} successfully saved to {out_dir}")


if __name__ == "__main__":
    main()
