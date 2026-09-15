# DeepISE Scientific Audit v1.1: Independent Hard-Negative Challenge Benchmark

> **Objective:** Evaluate specificity and false-positive rates on challenging cellular enzymes.
> **Source:** Curated Swiss-Prot Reviewed (Excluded from Training, Validation, and Test).
> **Homology Constraint:** Strict Reciprocal 30% Isolation ($\min(	ext{qcov}, 	ext{tcov}) < 80\%$ or $	ext{identity} < 30\%$ to train).

## 1. Challenge Dataset Composition

| Mechanistic Enzyme Class | Count | Description / Biological Relevance |
| :--- | :---: | :--- |
| **ruvc_nuclease** | 56 | Holliday junction resolvase RuvC (ancestor of Cas12/TnpB RuvC fold) |
| **plasmid_partition** | 46 | Plasmid segregation and partition ATPases (ParA/ParB) |
| **helicase** | 39 | DNA helicases (UvrD, RecQ, RecBCD) with motor ATPase domains |
| **rnase_h** | 26 | Ribonuclease H enzymes sharing the canonical catalytic RNase H fold with DDE transposases |
| **dna_repair_nuclease** | 22 | DNA repair endo/exonucleases (RecJ, UvrABC, MutS/L) |
| **serine_recombinase** | 6 | Resolvases / invertases (Gin, Cin) with catalytic serine residues |
| **phage_integrase** | 1 | Bacteriophage integrases |
| **Total Challenge Set** | **196** | Independent stress test benchmark |

