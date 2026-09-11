"""Target Site Duplication (TSD) detection module."""

from typing import Optional, Tuple
from deepise_ml.boundary.schemas import TSDMatch


def find_candidate_tsd(
    contig: str,
    left_boundary: int,
    right_boundary: int,
    min_tsd_len: int = 2,
    max_tsd_len: int = 14,
    allow_mismatch: bool = True,
    micro_shift: int = 3,
) -> Tuple[Optional[TSDMatch], int, int]:
    """
    Search for direct repeats flanking candidate IS boundaries.
    
    Returns:
        (best_tsd, refined_left_boundary, refined_right_boundary)
    """
    contig_len = len(contig)
    best_tsd: Optional[TSDMatch] = None
    best_score = -1.0
    refined_left = left_boundary
    refined_right = right_boundary

    # Try small boundary micro-shifts to recover exact terminal cleavage sites
    for d_left in range(-micro_shift, micro_shift + 1):
        cur_l = left_boundary + d_left
        if cur_l - max_tsd_len < 0:
            continue

        for d_right in range(-micro_shift, micro_shift + 1):
            cur_r = right_boundary + d_right
            if cur_r + max_tsd_len > contig_len or cur_r <= cur_l:
                continue

            for k in range(min_tsd_len, max_tsd_len + 1):
                seq_left = contig[cur_l - k : cur_l].upper()
                seq_right = contig[cur_r : cur_r + k].upper()

                # Calculate mismatches
                mismatches = sum(1 for a, b in zip(seq_left, seq_right) if a != b)
                
                # Filter: exact match or at most 1 mismatch for longer repeats
                if mismatches == 0:
                    # Exact TSD
                    # Length bonus: 3-9 bp are standard biological TSD lengths
                    score = k * 2.5 - abs(d_left) * 1.5 - abs(d_right) * 1.5
                elif allow_mismatch and mismatches == 1 and k >= 6:
                    score = (k - 1) * 1.8 - abs(d_left) * 1.5 - abs(d_right) * 1.5
                else:
                    continue

                if score > best_score:
                    best_score = score
                    refined_left = cur_l
                    refined_right = cur_r
                    best_tsd = TSDMatch(
                        left_start=cur_l - k,
                        left_end=cur_l,
                        right_start=cur_r,
                        right_end=cur_r + k,
                        sequence=seq_left,
                        length=k,
                        mismatches=mismatches,
                        score=round(score, 2),
                    )

    return best_tsd, refined_left, refined_right
