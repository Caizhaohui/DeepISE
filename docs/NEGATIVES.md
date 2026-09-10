# DeepISE Multi-Tiered Negative Dataset Specification

## 1. Rationale

Classifying a transposase versus a random ribosomal protein is biologically trivial for a modern protein language model. The true methodological challenge lies in distinguishing transposases from structurally and functionally related mobile, recombination, and nucleolytic enzymes (e.g. integrases, resolvases, RNase H-like folds).

## 2. Stratified Negative Levels

DeepISE samples non-transposase prokaryotic proteins from the curated **Swiss-Prot** database into four tiers:

| Negative Level | Fraction | Biological Purpose |
| :--- | :---: | :--- |
| **Level 1: Easy Negatives** | 20% | Housekeeping enzymes (ribosomal proteins, ATP synthase, RNA polymerase) for basic sanity checking. |
| **Level 2: Length-Matched** | 30% | Proteins matching the length distribution of positive transposases ($0.8L \le \text{length} \le 1.2L$) to prevent the model from learning a trivial length shortcut. |
| **Level 3: Taxonomy-Matched** | Integrated | Proteins originating from the same prokaryotic host taxa as positive IS elements. |
| **Level 4: Hard Negatives** | 50% | Enzymes sharing catalytic folds or mobile-element associations: integrases, resolvases, site-specific recombinases, invertases, DDE/HUH nucleases, RNase H, helicases, plasmid/phage mobility proteins. |

## 3. Ambiguous and Homology Filters

1. **Exclusion Filter**: Any protein with descriptions matching `transposase`, `insertion sequence`, or `mobile element` is discarded.
2. **Reverse-Homology Filter**: All candidate negatives are queried against known transposases using MMseqs2. Candidates with $E\text{-value} \le 10^{-5}$ and $\ge 50\%$ coverage are filtered out to prevent hidden positive contamination.
3. **Negative Homology Partitioning**: Negative proteins are also partitioned into 30% clusters to prevent negative-side data leakage between training and testing.
