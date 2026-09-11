"""Multi-task PLM Transposase and Family Classifier (Phase-3).

Predicts both:
1. Binary transposase probability P(Tpase)
2. 28-class IS family distribution P(Family | Tpase)
using frozen or fine-tuned ESM-2 protein language model representations.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class MultiTaskPLMFamilyModel(nn.Module):
    """Shared-trunk neural network with binary detection head and family classification head."""

    def __init__(
        self,
        input_dim: int = 480,
        hidden_dim: int = 256,
        num_families: int = 28,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.binary_head = nn.Linear(hidden_dim // 2, 1)
        self.family_head = nn.Linear(hidden_dim // 2, num_families)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.trunk(x)
        binary_logit = self.binary_head(feat).squeeze(-1)
        family_logits = self.family_head(feat)
        return binary_logit, family_logits


class PLMFamilyClassifier:
    """Trainer and inference engine for multi-task Tpase & Family classification."""

    def __init__(
        self,
        input_dim: int = 480,
        hidden_dim: int = 256,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.family_to_idx: Dict[str, int] = {}
        self.idx_to_family: Dict[int, str] = {}
        self.model: Optional[MultiTaskPLMFamilyModel] = None

    def fit(
        self,
        x_train: np.ndarray,
        y_train_bin: np.ndarray,
        y_train_fam: List[Optional[str]],
        x_val: np.ndarray,
        y_val_bin: np.ndarray,
        y_val_fam: List[Optional[str]],
        epochs: int = 20,
        batch_size: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> Dict[str, List[float]]:
        """Train the multi-task model on train embeddings with validation monitoring."""
        # 1. Build family vocabulary from train positives
        unique_fams = sorted(list(set(f for f, b in zip(y_train_fam, y_train_bin) if b == 1 and f)))
        self.family_to_idx = {fam: i for i, fam in enumerate(unique_fams)}
        self.idx_to_family = {i: fam for fam, i in self.family_to_idx.items()}
        num_families = len(self.family_to_idx)

        # 2. Encode target arrays
        fam_train_targets = np.array([self.family_to_idx.get(f, 0) for f in y_train_fam], dtype=np.int64)
        fam_val_targets = np.array([self.family_to_idx.get(f, 0) for f in y_val_fam], dtype=np.int64)

        # 3. Create PyTorch Datasets
        t_x_tr = torch.tensor(x_train, dtype=torch.float32)
        t_y_bin_tr = torch.tensor(y_train_bin, dtype=torch.float32)
        t_y_fam_tr = torch.tensor(fam_train_targets, dtype=torch.long)

        t_x_val = torch.tensor(x_val, dtype=torch.float32)
        t_y_bin_val = torch.tensor(y_val_bin, dtype=torch.float32)
        t_y_fam_val = torch.tensor(fam_val_targets, dtype=torch.long)

        train_loader = DataLoader(
            TensorDataset(t_x_tr, t_y_bin_tr, t_y_fam_tr),
            batch_size=batch_size,
            shuffle=True,
        )

        self.model = MultiTaskPLMFamilyModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_families=num_families,
        ).to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion_bce = nn.BCEWithLogitsLoss()
        criterion_ce = nn.CrossEntropyLoss(label_smoothing=0.05)

        history = {"train_loss": [], "val_bce": [], "val_fam_acc": []}

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            for bx, by_bin, by_fam in train_loader:
                bx = bx.to(self.device)
                by_bin = by_bin.to(self.device)
                by_fam = by_fam.to(self.device)

                optimizer.zero_grad()
                out_bin, out_fam = self.model(bx)

                loss_bin = criterion_bce(out_bin, by_bin)
                pos_mask = by_bin == 1
                if pos_mask.sum() > 0:
                    loss_fam = criterion_ce(out_fam[pos_mask], by_fam[pos_mask])
                    loss = loss_bin + 1.2 * loss_fam
                else:
                    loss = loss_bin

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_bx = t_x_val.to(self.device)
                val_by_bin = t_y_bin_val.to(self.device)
                val_by_fam = t_y_fam_val.to(self.device)

                val_out_bin, val_out_fam = self.model(val_bx)
                val_loss_bce = criterion_bce(val_out_bin, val_by_bin).item()

                pos_val = val_by_bin == 1
                if pos_val.sum() > 0:
                    pred_fam = val_out_fam[pos_val].argmax(dim=-1)
                    fam_acc = (pred_fam == val_by_fam[pos_val]).float().mean().item()
                else:
                    fam_acc = 0.0

            history["train_loss"].append(epoch_loss / len(train_loader))
            history["val_bce"].append(val_loss_bce)
            history["val_fam_acc"].append(fam_acc)

        return history

    def predict_proba(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (P(Tpase), P(Family | Tpase) distribution)."""
        if self.model is None:
            raise RuntimeError("Model not trained or loaded.")
        self.model.eval()
        t_x = torch.tensor(x, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            out_bin, out_fam = self.model(t_x)
            prob_bin = torch.sigmoid(out_bin).cpu().numpy()
            prob_fam = F.softmax(out_fam, dim=-1).cpu().numpy()
        return prob_bin, prob_fam

    def predict_family(self, x: np.ndarray) -> Tuple[List[str], np.ndarray]:
        """Return (Top-1 family names, Top-1 confidence)."""
        prob_bin, prob_fam = self.predict_proba(x)
        top1_idx = prob_fam.argmax(axis=-1)
        top1_conf = prob_fam.max(axis=-1)
        fams = [self.idx_to_family.get(int(i), "Unknown") for i in top1_idx]
        return fams, top1_conf

    def save(self, model_path: Path) -> None:
        """Save weights and vocabulary mapping."""
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "family_to_idx": self.family_to_idx,
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
            },
            model_path,
        )

    def load(self, model_path: Path) -> None:
        """Load weights and vocabulary mapping."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.family_to_idx = checkpoint["family_to_idx"]
        self.idx_to_family = {i: f for f, i in self.family_to_idx.items()}
        self.input_dim = checkpoint["input_dim"]
        self.hidden_dim = checkpoint["hidden_dim"]
        self.model = MultiTaskPLMFamilyModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_families=len(self.family_to_idx),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
