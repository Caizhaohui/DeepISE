# DeepISE Phase-1 Remote Homology Benchmark Report

## Overall Benchmark Performance on Homology-Disjoint Test Set

| Method | AUPRC | Recall@5%FDR | Recall@1%FDR | Precision@5%FDR | F1@5%FDR | MCC@5%FDR | Macro Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ESM2-LR | 0.9960 | **0.9878** | 0.9727 | 0.9519 | 0.9695 | 0.9594 | 0.9605 |
| ESM2-MLP | 0.9950 | **0.9849** | 0.9671 | 0.9650 | 0.9749 | 0.9665 | 0.9584 |
| Whole-pHMM | 0.9752 | **0.9690** | 0.9436 | 0.9904 | 0.9796 | 0.9729 | 0.9657 |
| BLASTP | 0.9786 | **0.9539** | 0.9163 | 0.9816 | 0.9676 | 0.9571 | 0.9572 |
| MMseqs2 | 0.9489 | **0.9323** | 0.9106 | 0.9970 | 0.9635 | 0.9529 | 0.9516 |

## Sequence Identity Stratified Recall (@ 5% FDR)

| Method | Identity Bin | Positive Count | Detected | Recall |
| :--- | :---: | :---: | :---: | :---: |
| BLASTP | <20% | 71 | 25 | 0.3521 |
| BLASTP | 20-30% | 221 | 220 | 0.9955 |
| BLASTP | 30-50% | 751 | 749 | 0.9973 |
| BLASTP | 50-70% | 10 | 10 | 1.0000 |
| BLASTP | >70% | 10 | 10 | 1.0000 |
| MMseqs2 | <20% | 71 | 0 | 0.0000 |
| MMseqs2 | 20-30% | 221 | 221 | 1.0000 |
| MMseqs2 | 30-50% | 751 | 750 | 0.9987 |
| MMseqs2 | 50-70% | 10 | 10 | 1.0000 |
| MMseqs2 | >70% | 10 | 10 | 1.0000 |
| Whole-pHMM | <20% | 71 | 39 | 0.5493 |
| Whole-pHMM | 20-30% | 221 | 220 | 0.9955 |
| Whole-pHMM | 30-50% | 751 | 751 | 1.0000 |
| Whole-pHMM | 50-70% | 10 | 10 | 1.0000 |
| Whole-pHMM | >70% | 10 | 10 | 1.0000 |
| ESM2-LR | <20% | 71 | 59 | 0.8310 |
| ESM2-LR | 20-30% | 221 | 220 | 0.9955 |
| ESM2-LR | 30-50% | 751 | 751 | 1.0000 |
| ESM2-LR | 50-70% | 10 | 10 | 1.0000 |
| ESM2-LR | >70% | 10 | 10 | 1.0000 |
| ESM2-MLP | <20% | 71 | 56 | 0.7887 |
| ESM2-MLP | 20-30% | 221 | 220 | 0.9955 |
| ESM2-MLP | 30-50% | 751 | 751 | 1.0000 |
| ESM2-MLP | 50-70% | 10 | 10 | 1.0000 |
| ESM2-MLP | >70% | 10 | 10 | 1.0000 |