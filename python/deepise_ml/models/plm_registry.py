"""Pluggable Protein Language Model (pLM) Registry and Multi-Model Embedding Engine.

Supports multiple pLM backbones:
- ESM-2 (35M, default production backbone)
- SaProt (35M, structure-aware protein language model)
- ESM-2 (8M, ultra-lightweight edge deployment)
- ESM-2 (650M / ProtBERT extensible)
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
import torch
from Bio import SeqIO
from transformers import AutoModel, AutoTokenizer, EsmModel

# Set default HuggingFace mirror for accelerated downloads in domestic HPC
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

PLM_MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "esm2_35m": {
        "model_id": "facebook/esm2_t12_35M_UR50D",
        "name": "ESM-2 (35M)",
        "type": "esm",
        "dim": 480,
        "layers": 12,
        "format": "standard",
        "description": "Standard BERT-style pLM; default production baseline",
    },
    "saprot_35m": {
        "model_id": "westlake-repl/SaProt_35M_AF2",
        "name": "SaProt (35M)",
        "type": "saprot",
        "dim": 480,
        "layers": 12,
        "format": "saprot_seq",
        "description": "Structure-aware pLM (Foldseek-tokenized alphabet with sequence fallback)",
    },
    "esm2_8m": {
        "model_id": "facebook/esm2_t6_8M_UR50D",
        "name": "ESM-2 (8M)",
        "type": "esm",
        "dim": 320,
        "layers": 6,
        "format": "standard",
        "description": "Ultra-lightweight 6-layer pLM for memory-constrained edge deployment",
    },
}


def format_sequence_for_plm(seq: str, fmt: str) -> str:
    """Format raw amino acid sequence for specific model tokenization."""
    clean_seq = seq.strip().upper()
    if fmt == "standard":
        return clean_seq
    elif fmt == "saprot_seq":
        # SaProt sequence-only representation: each AA followed by '#'
        return "".join(f"{aa}#" for aa in clean_seq)
    elif fmt == "spaced":
        # ProtBERT / ProtT5 format
        return " ".join(list(clean_seq))
    return clean_seq


class PluggablePLMEngine:
    """Unified inference engine for pluggable protein language models."""

    def __init__(
        self,
        model_key: str = "esm2_35m",
        device: Optional[str] = None,
    ):
        if model_key not in PLM_MODEL_CONFIGS:
            raise ValueError(f"Unknown pLM model key: {model_key}. Choices: {list(PLM_MODEL_CONFIGS.keys())}")

        self.model_key = model_key
        self.config = PLM_MODEL_CONFIGS[model_key]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        model_id = self.config["model_id"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        if self.config["type"] in ["esm", "saprot"]:
            self.model = EsmModel.from_pretrained(model_id, use_safetensors=True)
        else:
            self.model = AutoModel.from_pretrained(model_id)

        self.model.eval()
        self.model.to(self.device)

    def extract_sequence_embeddings(
        self,
        sequences: List[str],
        batch_size: int = 32,
        max_len: int = 1022,
        verbose: bool = True,
    ) -> np.ndarray:
        """Extract mean-pooled protein representations with batched GPU acceleration and sliding window support."""
        fmt = self.config["format"]
        embeddings: List[np.ndarray] = []

        total = len(sequences)
        start_time = time.time()

        with torch.no_grad():
            for i in range(0, total, batch_size):
                batch_seqs = sequences[i : i + batch_size]
                has_long = any(len(s) > max_len for s in batch_seqs)

                if not has_long:
                    batch_fmts = [format_sequence_for_plm(s, fmt) for s in batch_seqs]
                    inputs = self.tokenizer(batch_fmts, padding=True, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    outputs = self.model(**inputs)
                    last_hidden = outputs.last_hidden_state  # (B, L, D)
                    masks = inputs["attention_mask"]          # (B, L)

                    for j in range(len(batch_seqs)):
                        valid_len = int(masks[j].sum().item())
                        rep = last_hidden[j, 1 : valid_len - 1, :].mean(dim=0).cpu().numpy()
                        embeddings.append(rep)
                else:
                    for s_raw in batch_seqs:
                        seq_len = len(s_raw)
                        if seq_len <= max_len:
                            s_fmt = format_sequence_for_plm(s_raw, fmt)
                            inputs = self.tokenizer(s_fmt, return_tensors="pt", add_special_tokens=True)
                            inputs = {k: v.to(self.device) for k, v in inputs.items()}
                            outputs = self.model(**inputs)
                            rep = outputs.last_hidden_state[0, 1:-1, :].mean(dim=0).cpu().numpy()
                        else:
                            stride = max_len // 2
                            win_reps = []
                            weights = []
                            for start in range(0, seq_len, stride):
                                end = min(start + max_len, seq_len)
                                chunk = s_raw[start:end]
                                if len(chunk) < 5:
                                    continue
                                chunk_fmt = format_sequence_for_plm(chunk, fmt)
                                inputs = self.tokenizer(chunk_fmt, return_tensors="pt", add_special_tokens=True)
                                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                                outputs = self.model(**inputs)
                                rep = outputs.last_hidden_state[0, 1:-1, :].mean(dim=0).cpu().numpy()
                                win_reps.append(rep)
                                weights.append(len(chunk))
                                if end == seq_len:
                                    break
                            rep = np.average(win_reps, axis=0, weights=weights)
                        embeddings.append(rep)

                if verbose and ((i + batch_size >= total) or ((i // batch_size) % 25 == 0)):
                    elapsed = time.time() - start_time
                    done = min(i + batch_size, total)
                    rate = done / (elapsed + 1e-5)
                    print(f"[{self.config['name']}] Embedded {done}/{total} seqs ({rate:.1f} seq/s)")

        return np.vstack(embeddings).astype(np.float32)


def extract_embeddings_for_parquet(
    parquet_path: Path,
    output_npy: Path,
    model_key: str = "esm2_35m",
    batch_size: int = 32,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[Path, np.ndarray]:
    """Extract embeddings for sequences stored in a parquet split."""
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    if output_npy.exists() and output_npy.stat().st_size > 0:
        return output_npy, np.load(output_npy)

    df = pl.read_parquet(parquet_path)
    seq_col = "protein_sequence" if "protein_sequence" in df.columns else "sequence"
    seqs = df[seq_col].to_list()

    engine = PluggablePLMEngine(model_key=model_key, device=device)
    embs = engine.extract_sequence_embeddings(seqs, batch_size=batch_size, verbose=verbose)
    np.save(output_npy, embs)
    return output_npy, embs
