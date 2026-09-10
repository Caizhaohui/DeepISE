# DeepISE Splitting Methodology & Homology Leakage Controls

## 1. The Core Methodological Rule

> **Random protein sequence splitting is strictly prohibited.**

In transposable element biology, mobile elements frequently duplicate across related strains with 95%–100% identity. A random split allows near-identical copies into both training and test sets, artificially inflating test metrics (e.g. AUROC > 0.99) while providing zero real generalization to remote or novel elements.

## 2. 30% Sequence Identity Cluster Partitioning

All deduplicated transposase sequences are clustered using **MMseqs2**:
- `--min-seq-id 0.30`
- `-c 0.80` (minimum 80% bidirectional alignment coverage)
- `--cov-mode 0`
- `--cluster-mode 2`
- `-s 7.5`

The atomic unit for dataset splitting is the **`cluster30_id`**, not the individual sequence. Every sequence belonging to cluster $C_i$ is placed strictly into **either** Train (70%), Validation (15%), or Test (15%).

## 3. Split Allocation Algorithm

A greedy multi-criteria allocator balances:
1. **Cluster Integrity**: Zero cluster overlap between splits ($C_{train} \cap C_{test} = \emptyset$).
2. **Target Capacities**: Approximately 70% Train, 15% Validation, 15% Test.
3. **IS Family Representation**: Preserves proportional distribution of IS families across splits.
4. **Reproducibility**: Deterministic seed (`seed = 42`).

## 4. All-vs-All Homology Leakage Audit

To empirically verify zero leakage, DeepISE executes an all-vs-all MMseqs2 search (`test vs train`).
Any test protein with:
$$\text{Identity} > 0.30 \quad \text{AND} \quad \max(\text{Coverage}_{query}, \text{Coverage}_{target}) \ge 0.80$$
is flagged as a **LEAKAGE VIOLATION**.
If violations $> 0$, the benchmark creation pipeline fails immediately.
