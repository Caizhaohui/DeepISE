"""Neural Boundary Refinement Module (Phase-3).

Refines coarse/fuzzy IS element boundaries using a 1D Dilated Residual Convolutional
Neural Network trained on local junction contexts (terminal inverted repeat transitions,
target site duplications, and cleavage motif signatures).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot_encode_dna(seq: str, length: int = 128) -> np.ndarray:
    """One-hot encode a DNA string into shape (4, length)."""
    seq = seq.upper()
    if len(seq) < length:
        seq = seq.ljust(length, "N")
    elif len(seq) > length:
        seq = seq[:length]

    mapping = {
        "A": [1.0, 0.0, 0.0, 0.0],
        "C": [0.0, 1.0, 0.0, 0.0],
        "G": [0.0, 0.0, 1.0, 0.0],
        "T": [0.0, 0.0, 0.0, 1.0],
    }
    encoded = np.zeros((4, length), dtype=np.float32)
    for i, char in enumerate(seq):
        vec = mapping.get(char, [0.25, 0.25, 0.25, 0.25])
        for c in range(4):
            encoded[c, i] = vec[c]
    return encoded


class ResidualBlock1D(nn.Module):
    """1D Convolutional Residual Block with dilation support."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + res)


class BoundaryRefinerNet(nn.Module):
    """1D Dilated Residual Network with joint classification and regression heads."""

    def __init__(self, in_channels: int = 4, base_channels: int = 32, window_size: int = 128):
        super().__init__()
        self.window_size = window_size
        self.base_channels = base_channels
        self.init_conv = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.GELU(),
        )

        self.res1 = ResidualBlock1D(base_channels, base_channels, kernel_size=5, dilation=1)
        self.res2 = ResidualBlock1D(base_channels, base_channels * 2, kernel_size=5, dilation=2)
        self.res3 = ResidualBlock1D(base_channels * 2, base_channels * 2, kernel_size=5, dilation=4)

        # Head 1: Junction coordinate distribution (per-nucleotide probability over window)
        self.junction_head = nn.Conv1d(base_channels * 2, 1, kernel_size=1)

        # Head 2: Offset regression head (predicts displacement in bp relative to center)
        self.offset_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.init_conv(x)
        feat = self.res1(feat)
        feat = self.res2(feat)
        feat = self.res3(feat)

        junction_logits = self.junction_head(feat).squeeze(1)  # (B, window_size)
        offset = self.offset_head(feat).squeeze(-1)            # (B,)
        return junction_logits, offset


class NeuralBoundaryRefiner:
    """Inference engine for deep learning boundary refinement."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        window_size: int = 128,
        device: Optional[str] = None,
    ):
        self.window_size = window_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BoundaryRefinerNet(window_size=window_size).to(self.device)
        self.model.eval()

        if model_path is not None and model_path.exists():
            self.load(model_path)

    def refine_boundary(
        self,
        contig_seq: str,
        cand_coord: int,
        is_five_prime: bool = True,
        max_shift: int = 30,
    ) -> Tuple[int, float, float]:
        """Refine a candidate boundary coordinate using local DNA context.

        Args:
            contig_seq: Entire contig nucleotide sequence.
            cand_coord: Candidate 0-indexed coordinate.
            is_five_prime: True for 5' boundary, False for 3' boundary.
            max_shift: Maximum allowable shift in bp.

        Returns:
            Tuple of (refined_coord, predicted_offset, confidence_score).
        """
        half_w = self.window_size // 2
        s_idx = max(0, cand_coord - half_w)
        e_idx = min(len(contig_seq), cand_coord + half_w)
        subseq = contig_seq[s_idx:e_idx]

        # Pad if at contig borders
        if len(subseq) < self.window_size:
            left_pad = "N" * max(0, half_w - cand_coord)
            right_pad = "N" * max(0, self.window_size - len(subseq) - len(left_pad))
            subseq = left_pad + subseq + right_pad

        one_hot = one_hot_encode_dna(subseq, self.window_size)
        t_x = torch.tensor(one_hot, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            j_logits, offset = self.model(t_x)
            j_probs = F.softmax(j_logits, dim=-1).cpu().numpy()[0]
            pred_offset = offset.cpu().item()

        # Combine peak junction index with regression offset
        peak_idx = int(np.argmax(j_probs))
        class_offset = peak_idx - half_w
        peak_prob = float(j_probs[peak_idx])

        # Soft consensus between classification peak and regression
        consensus_offset = 0.6 * class_offset + 0.4 * pred_offset
        bounded_offset = int(np.clip(round(consensus_offset), -max_shift, max_shift))

        refined_coord = max(0, min(len(contig_seq), cand_coord + bounded_offset))
        confidence = float(np.clip(peak_prob * 2.0, 0.0, 1.0))

        return refined_coord, float(bounded_offset), confidence

    def save(self, save_path: Path) -> None:
        """Save model checkpoint."""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "window_size": self.window_size,
                "base_channels": getattr(self.model, "base_channels", 32),
            },
            save_path,
        )

    def load(self, model_path: Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.window_size = checkpoint["window_size"]
        base_channels = checkpoint.get("base_channels", 32)
        self.model = BoundaryRefinerNet(
            in_channels=4,
            base_channels=base_channels,
            window_size=self.window_size,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
