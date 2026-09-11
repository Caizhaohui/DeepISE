"""ESM-2 frozen representation extraction and baseline classifiers (LR & MLP)."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import polars as pl
from Bio import SeqIO


def extract_esm2_embeddings(
    fasta_path: Path,
    output_npy: Path,
    output_index_parquet: Path,
    model_name: str = "facebook/esm2_t12_35M_UR50D",
    batch_size: int = 16,
    max_len: int = 1022,
    device: Optional[str] = None,
) -> Tuple[Path, Path]:
    """Extract mean-pooled frozen ESM-2 embeddings for protein sequences.
    
    Handles sequences longer than max_len with overlapping sliding windows.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    output_npy.parent.mkdir(parents=True, exist_ok=True)
    output_index_parquet.parent.mkdir(parents=True, exist_ok=True)
    if output_npy.exists() and output_index_parquet.exists() and output_npy.stat().st_size > 0:
        return output_npy, output_index_parquet

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)

    records = list(SeqIO.parse(fasta_path, "fasta"))
    seq_ids = [r.id for r in records]
    seqs = [str(r.seq).strip().upper() for r in records]

    embeddings = []
    index_rows = []

    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch_seqs = seqs[i : i + batch_size]
            batch_ids = seq_ids[i : i + batch_size]

            for s_id, s_str in zip(batch_ids, batch_seqs):
                seq_len = len(s_str)
                is_long = seq_len > max_len

                if not is_long:
                    # Single window
                    inputs = tokenizer(s_str, return_tensors="pt", add_special_tokens=True)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = model(**inputs)
                    # Exclude [CLS] and [EOS] tokens from mean pool
                    token_reps = outputs.last_hidden_state[0, 1:-1, :]
                    mean_rep = token_reps.mean(dim=0).cpu().numpy()
                else:
                    # Sliding window (stride = max_len // 2)
                    stride = max_len // 2
                    window_reps = []
                    weights = []
                    for start in range(0, seq_len, stride):
                        end = min(start + max_len, seq_len)
                        chunk = s_str[start:end]
                        if len(chunk) < 5:
                            continue
                        inputs = tokenizer(chunk, return_tensors="pt", add_special_tokens=True)
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        outputs = model(**inputs)
                        rep = outputs.last_hidden_state[0, 1:-1, :].mean(dim=0).cpu().numpy()
                        window_reps.append(rep)
                        weights.append(len(chunk))
                        if end == seq_len:
                            break

                    mean_rep = np.average(window_reps, axis=0, weights=weights)

                embeddings.append(mean_rep)
                index_rows.append({
                    "seq_id": s_id,
                    "length": seq_len,
                    "is_long": is_long,
                    "embedding_idx": len(embeddings) - 1,
                })

    emb_array = np.vstack(embeddings).astype(np.float32)
    np.save(output_npy, emb_array)

    index_df = pl.DataFrame(index_rows)
    index_df.write_parquet(output_index_parquet)

    return output_npy, output_index_parquet


class ESM2LinearClassifier:
    """Standardized LogisticRegression classifier on frozen ESM-2 embeddings."""

    def __init__(self, C: float = 1.0, random_state: int = 42):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(
            C=C,
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        # Returns probability of positive class (label=1)
        return self.clf.predict_proba(X_scaled)[:, 1]


class ESM2MLPClassifier:
    """1-Hidden-Layer MLP on frozen ESM-2 embeddings."""

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.2, lr: float = 1e-3, epochs: int = 30):
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray, device: str = "cpu"):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        input_dim = X.shape[1]
        self.model = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        ).to(device)

        dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        pos_weight = torch.tensor([(len(y) - sum(y)) / (sum(y) + 1e-5)]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                logits = self.model(batch_x).squeeze(-1)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X: np.ndarray, device: str = "cpu") -> np.ndarray:
        import torch
        self.model.eval()
        with torch.no_grad():
            tensor_x = torch.from_numpy(X).float().to(device)
            logits = self.model(tensor_x).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs
