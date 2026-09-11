"""Train and validate Neural Boundary Refiner on cluster30 IS sequences."""

import random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from Bio import SeqIO

from deepise_ml.boundary.neural_refiner import (
    BoundaryRefinerNet,
    NeuralBoundaryRefiner,
    one_hot_encode_dna,
)


def generate_random_flank(length: int, gc_prob: float = 0.50) -> str:
    """Generate random DNA with specified GC content."""
    at_prob = (1.0 - gc_prob) / 2.0
    gc_half = gc_prob / 2.0
    bases = ["A", "C", "G", "T"]
    probs = [at_prob, gc_half, gc_half, at_prob]
    return "".join(random.choices(bases, weights=probs, k=length))


def build_boundary_training_dataset(
    is_fna_path: Path,
    split_parquet: Path,
    num_samples_per_element: int = 2,
    max_perturbation: int = 25,
    window_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract perturbed boundary windows from IS sequences strictly matching the split."""
    # 1. Load split IS names
    df_split = pl.read_parquet(split_parquet)
    valid_is_names = set(df_split["is_name"].unique().to_list())
    
    # 2. Index IS nucleotide records
    is_records = {}
    for r in SeqIO.parse(is_fna_path, "fasta"):
        # Header format: e.g. "IS-LL6_IS3_IS3" or "IS1"
        is_name = r.id.split("_")[0]
        if is_name in valid_is_names:
            is_records[is_name] = str(r.seq).upper()
            
    print(f"Matched {len(is_records)} IS elements in split {split_parquet.name}.")
    
    windows = []
    offsets = []
    junction_indices = []
    half_w = window_size // 2
    
    random.seed(42)
    np.random.seed(42)
    
    for is_name, is_seq in is_records.items():
        is_len = len(is_seq)
        if is_len < 200:
            continue
            
        gc_val = (is_seq.count("G") + is_seq.count("C")) / max(1, is_len)
        
        # Flanking DNA
        flank_left = generate_random_flank(300, gc_prob=gc_val)
        flank_right = generate_random_flank(300, gc_prob=gc_val)
        
        # Optional TSD insertion
        tsd_len = random.choice([0, 2, 4, 8, 9])
        if tsd_len > 0:
            tsd_motif = flank_left[-tsd_len:]
            flank_right = tsd_motif + flank_right[tsd_len:]
            
        contig = flank_left + is_seq + flank_right
        s_true = len(flank_left)
        e_true = s_true + is_len
        
        # Sample perturbations for 5' boundary
        for _ in range(num_samples_per_element):
            delta = random.randint(-max_perturbation, max_perturbation)
            s_cand = s_true + delta
            w_left = contig[s_cand - half_w : s_cand + half_w]
            if len(w_left) == window_size:
                target_offset = float(-delta)
                target_idx = half_w - delta
                windows.append(one_hot_encode_dna(w_left, window_size))
                offsets.append(target_offset)
                junction_indices.append(target_idx)
                
        # Sample perturbations for 3' boundary
        for _ in range(num_samples_per_element):
            delta = random.randint(-max_perturbation, max_perturbation)
            e_cand = e_true + delta
            w_right = contig[e_cand - half_w : e_cand + half_w]
            if len(w_right) == window_size:
                target_offset = float(-delta)
                target_idx = half_w - delta
                windows.append(one_hot_encode_dna(w_right, window_size))
                offsets.append(target_offset)
                junction_indices.append(target_idx)
                
    x_arr = np.stack(windows).astype(np.float32)
    y_offsets = np.array(offsets, dtype=np.float32)
    y_indices = np.array(junction_indices, dtype=np.int64)
    
    return x_arr, y_offsets, y_indices


def train_neural_boundary_refiner():
    print("Building Neural Boundary Refiner training and validation sets...")
    is_fna = Path("data/raw/isfinder/2026-09-10/IS.fna")
    train_parquet = Path("data/splits/cluster30/train.parquet")
    val_parquet = Path("data/splits/cluster30/validation.parquet")
    
    x_train, y_off_train, y_idx_train = build_boundary_training_dataset(is_fna, train_parquet, num_samples_per_element=2)
    x_val, y_off_val, y_idx_val = build_boundary_training_dataset(is_fna, val_parquet, num_samples_per_element=2)
    
    print(f"Train samples: {x_train.shape[0]}, Val samples: {x_val.shape[0]}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")
    
    model = BoundaryRefinerNet(in_channels=4, base_channels=32, window_size=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    
    criterion_ce = nn.CrossEntropyLoss(label_smoothing=0.05)
    criterion_huber = nn.HuberLoss(delta=1.0)
    
    train_dataset = TensorDataset(
        torch.tensor(x_train),
        torch.tensor(y_off_train),
        torch.tensor(y_idx_train),
    )
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    t_x_val = torch.tensor(x_val, dtype=torch.float32).to(device)
    t_y_off_val = torch.tensor(y_off_val, dtype=torch.float32).to(device)
    t_y_idx_val = torch.tensor(y_idx_val, dtype=torch.long).to(device)
    
    epochs = 15
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for bx, by_off, by_idx in train_loader:
            bx = bx.to(device)
            by_off = by_off.to(device)
            by_idx = by_idx.to(device)
            
            optimizer.zero_grad()
            pred_logits, pred_offsets = model(bx)
            
            loss_ce = criterion_ce(pred_logits, by_idx)
            loss_huber = criterion_huber(pred_offsets, by_off)
            loss = loss_ce + 0.5 * loss_huber
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            v_logits, v_offsets = model(t_x_val)
            v_loss = (criterion_ce(v_logits, t_y_idx_val) + 0.5 * criterion_huber(v_offsets, t_y_off_val)).item()
            
            v_pred_idx = v_logits.argmax(dim=-1)
            acc_exact = (v_pred_idx == t_y_idx_val).float().mean().item()
            acc_near3 = (torch.abs(v_pred_idx - t_y_idx_val) <= 3).float().mean().item()
            mean_abs_err = torch.abs(v_offsets - t_y_off_val).mean().item()
            
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {epoch_loss/len(train_loader):.4f} | Val Loss: {v_loss:.4f} | Exact (0bp): {acc_exact*100:.2f}% | Near (<=3bp): {acc_near3*100:.2f}% | Mean Err: {mean_abs_err:.2f}bp")
        
    save_path = Path("benchmark/models/neural_boundary_refiner.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "window_size": 128,
        },
        save_path,
    )
    print(f"Neural Boundary Refiner successfully saved to {save_path}")


if __name__ == "__main__":
    train_neural_boundary_refiner()
