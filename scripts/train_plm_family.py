"""Train and evaluate MultiTaskPLMFamilyModel on ESM-2 embeddings."""

from pathlib import Path
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, classification_report, f1_score

from deepise_ml.models.plm_family import PLMFamilyClassifier


def train_and_eval_family_model():
    print("Loading datasets and embeddings...")
    emb_dir = Path("data/processed/embeddings")
    splits_dir = Path("data/splits/cluster30")
    
    # 1. Load train
    x_tr = np.load(emb_dir / "esm2_train.npy")
    df_tr_idx = pl.read_parquet(emb_dir / "esm2_train_index.parquet")
    df_tr_meta = pl.read_parquet(splits_dir / "train_combined.parquet")
    df_tr = df_tr_idx.join(df_tr_meta, on="seq_id")
    y_tr_bin = df_tr["label"].to_numpy().astype(np.float32)
    y_tr_fam = [f if b == 1 else None for f, b in zip(df_tr["family"].to_list(), y_tr_bin)]
    
    # 2. Load val
    x_val = np.load(emb_dir / "esm2_val.npy")
    df_val_idx = pl.read_parquet(emb_dir / "esm2_val_index.parquet")
    df_val_meta = pl.read_parquet(splits_dir / "validation_combined.parquet")
    df_val = df_val_idx.join(df_val_meta, on="seq_id")
    y_val_bin = df_val["label"].to_numpy().astype(np.float32)
    y_val_fam = [f if b == 1 else None for f, b in zip(df_val["family"].to_list(), y_val_bin)]
    
    # 3. Load test
    x_te = np.load(emb_dir / "esm2_test.npy")
    df_te_idx = pl.read_parquet(emb_dir / "esm2_test_index.parquet")
    df_te_meta = pl.read_parquet(splits_dir / "test_combined.parquet")
    df_te = df_te_idx.join(df_te_meta, on="seq_id")
    y_te_bin = df_te["label"].to_numpy().astype(np.float32)
    y_te_fam = [f if b == 1 else None for f, b in zip(df_te["family"].to_list(), y_te_bin)]
    
    print(f"Train: {x_tr.shape}, Val: {x_val.shape}, Test: {x_te.shape}")
    
    # 4. Train model
    clf = PLMFamilyClassifier(input_dim=480, hidden_dim=256)
    history = clf.fit(
        x_tr, y_tr_bin, y_tr_fam,
        x_val, y_val_bin, y_val_fam,
        epochs=25,
        batch_size=128,
        lr=1e-3,
    )
    
    print(f"Final Val BCE: {history['val_bce'][-1]:.4f}, Final Val Fam Acc: {history['val_fam_acc'][-1]*100:.2f}%")
    
    # 5. Evaluate on Test Set
    prob_bin, prob_fam = clf.predict_proba(x_te)
    test_auprc = average_precision_score(y_te_bin, prob_bin)
    print(f"Test Binary Transposase AUPRC: {test_auprc:.4f}")
    
    # Evaluate family classification on test positives
    pos_mask = y_te_bin == 1
    pos_te_fams = [f for f, m in zip(y_te_fam, pos_mask) if m]
    pos_prob_fam = prob_fam[pos_mask]
    pred_indices = pos_prob_fam.argmax(axis=-1)
    pred_fams = [clf.idx_to_family[i] for i in pred_indices]
    
    macro_f1 = f1_score(pos_te_fams, pred_fams, average="macro", zero_division=0)
    micro_acc = np.mean([p == t for p, t in zip(pred_fams, pos_te_fams)])
    
    # Top-3 Accuracy
    top3_indices = np.argsort(pos_prob_fam, axis=-1)[:, -3:]
    top3_acc = np.mean([clf.family_to_idx.get(t, -1) in top3 for t, top3 in zip(pos_te_fams, top3_indices)])
    
    print(f"Test Positive Samples: {len(pos_te_fams)}")
    print(f"Family Micro-Accuracy (Top-1): {micro_acc*100:.2f}%")
    print(f"Family Top-3 Accuracy: {top3_acc*100:.2f}%")
    print(f"Family Macro-F1: {macro_f1:.4f}")
    
    # Save model checkpoint
    model_save_path = Path("benchmark/models/plm_family_classifier.pt")
    clf.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Save test predictions table
    out_table = Path("benchmark/tables/phase3_plm_family_metrics.tsv")
    out_table.parent.mkdir(parents=True, exist_ok=True)
    
    rows = [
        {"metric": "Test Binary Transposase AUPRC", "value": f"{test_auprc:.4f}"},
        {"metric": "Test Positive Family Micro-Accuracy (Top-1)", "value": f"{micro_acc*100:.2f}%"},
        {"metric": "Test Positive Family Top-3 Accuracy", "value": f"{top3_acc*100:.2f}%"},
        {"metric": "Test Positive Family Macro-F1", "value": f"{macro_f1:.4f}"},
        {"metric": "Number of Evaluated Families", "value": f"{len(clf.family_to_idx)}"},
    ]
    pl.DataFrame(rows).write_csv(out_table, separator="\t")
    print(f"Saved metrics to {out_table}")


if __name__ == "__main__":
    train_and_eval_family_model()
