"""Unit tests for Phase-3 Neural Boundary Refiner and Multi-Task PLM Family Model."""

from pathlib import Path
import numpy as np
import pytest
import torch

from deepise_ml.boundary.hybrid import HybridBoundaryEngine
from deepise_ml.boundary.neural_refiner import (
    BoundaryRefinerNet,
    NeuralBoundaryRefiner,
    one_hot_encode_dna,
)
from deepise_ml.models.plm_family import MultiTaskPLMFamilyModel, PLMFamilyClassifier


def test_one_hot_encode_dna():
    seq = "ACGTN"
    enc = one_hot_encode_dna(seq, length=8)
    assert enc.shape == (4, 8)
    # Check A
    assert np.allclose(enc[:, 0], [1.0, 0.0, 0.0, 0.0])
    # Check C
    assert np.allclose(enc[:, 1], [0.0, 1.0, 0.0, 0.0])
    # Check G
    assert np.allclose(enc[:, 2], [0.0, 0.0, 1.0, 0.0])
    # Check T
    assert np.allclose(enc[:, 3], [0.0, 0.0, 0.0, 1.0])
    # Check N (padded/unknown)
    assert np.allclose(enc[:, 4], [0.25, 0.25, 0.25, 0.25])


def test_boundary_refiner_net_forward():
    net = BoundaryRefinerNet(in_channels=4, base_channels=16, window_size=128)
    x = torch.randn(4, 4, 128)
    j_logits, offsets = net(x)
    assert j_logits.shape == (4, 128)
    assert offsets.shape == (4,)


def test_neural_boundary_refiner_inference(tmp_path):
    # Save a small dummy model
    dummy_model_path = tmp_path / "dummy_refiner.pt"
    net = BoundaryRefinerNet(in_channels=4, base_channels=16, window_size=128)
    torch.save({"model_state": net.state_dict(), "window_size": 128, "base_channels": 16}, dummy_model_path)

    refiner = NeuralBoundaryRefiner(model_path=dummy_model_path, window_size=128)
    contig = "A" * 500 + "GATC" * 50 + "T" * 500
    cand = 500

    ref_coord, offset, conf = refiner.refine_boundary(contig, cand, is_five_prime=True, max_shift=20)
    assert isinstance(ref_coord, int)
    assert isinstance(offset, float)
    assert 0.0 <= conf <= 1.0
    assert abs(ref_coord - cand) <= 20


def test_hybrid_boundary_engine_routing(tmp_path):
    engine = HybridBoundaryEngine(neural_model_path=None)
    contig = "A" * 300 + "GGTTCCAATT" + "C" * 800 + "AATTGGAACC" + "T" * 300
    pred = engine.predict_boundary(
        contig=contig,
        tpase_start=400,
        tpase_end=1000,
        family="IS3",
    )
    assert pred.predicted_start < pred.predicted_end
    assert pred.confidence_score > 0.0


def test_plm_family_model_forward():
    model = MultiTaskPLMFamilyModel(input_dim=480, hidden_dim=64, num_families=5)
    x = torch.randn(8, 480)
    bin_logit, fam_logits = model(x)
    assert bin_logit.shape == (8,)
    assert fam_logits.shape == (8, 5)


def test_plm_family_classifier_train_and_predict(tmp_path):
    # Synthetic small dataset
    np.random.seed(42)
    x_tr = np.random.randn(50, 480).astype(np.float32)
    y_tr_bin = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(50)], dtype=np.float32)
    y_tr_fam = ["IS1" if i % 4 == 0 else "IS3" if i % 2 == 0 else None for i in range(50)]

    x_val = np.random.randn(20, 480).astype(np.float32)
    y_val_bin = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(20)], dtype=np.float32)
    y_val_fam = ["IS1" if i % 4 == 0 else "IS3" if i % 2 == 0 else None for i in range(20)]

    clf = PLMFamilyClassifier(input_dim=480, hidden_dim=64)
    history = clf.fit(x_tr, y_tr_bin, y_tr_fam, x_val, y_val_bin, y_val_fam, epochs=3, batch_size=16)

    assert len(history["train_loss"]) == 3
    prob_bin, prob_fam = clf.predict_proba(x_val[:5])
    assert prob_bin.shape == (5,)
    assert prob_fam.shape[0] == 5
    assert prob_fam.shape[1] == len(clf.family_to_idx)

    fams, confs = clf.predict_family(x_val[:5])
    assert len(fams) == 5
    assert len(confs) == 5

    save_pt = tmp_path / "test_plm.pt"
    clf.save(save_pt)
    assert save_pt.exists()

    clf2 = PLMFamilyClassifier()
    clf2.load(save_pt)
    assert clf2.family_to_idx == clf.family_to_idx
