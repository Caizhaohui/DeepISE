"""Unit tests for negative sampling logic."""

import pytest
from deepise_ml.dataset.negatives import sample_negatives


def test_sample_negatives_distribution():
    hard_pool = [
        {"accession": f"H{i}", "entry_name": f"H_ENT_{i}", "description": "integrase", "organism": "E. coli", "taxonomy_id": "562", "protein_sequence": "M"*200, "protein_length": 200, "negative_type": "hard"}
        for i in range(50)
    ]
    easy_pool = [
        {"accession": f"E{i}", "entry_name": f"E_ENT_{i}", "description": "ribosomal protein", "organism": "E. coli", "taxonomy_id": "562", "protein_sequence": "A"*100, "protein_length": 100, "negative_type": "easy"}
        for i in range(30)
    ]
    general_pool = [
        {"accession": f"G{i}", "entry_name": f"G_ENT_{i}", "description": "metabolic enzyme", "organism": "E. coli", "taxonomy_id": "562", "protein_sequence": "L"*300, "protein_length": 300, "negative_type": "general"}
        for i in range(50)
    ]

    pos_lengths = [100, 200, 300] * 10
    target_count = 20

    records = sample_negatives(
        hard_pool=hard_pool,
        easy_pool=easy_pool,
        general_pool=general_pool,
        positive_lengths=pos_lengths,
        target_count=target_count,
        hard_ratio=0.50,
        matched_ratio=0.30,
        easy_ratio=0.20,
    )

    assert len(records) == target_count
    hard_count = sum(1 for r in records if r.negative_type == "hard")
    easy_count = sum(1 for r in records if r.negative_type == "easy")
    matched_count = sum(1 for r in records if r.negative_type == "length_matched")

    assert hard_count == 10  # 50% of 20
    assert matched_count == 6  # 30% of 20
    assert easy_count == 4  # 20% of 20
