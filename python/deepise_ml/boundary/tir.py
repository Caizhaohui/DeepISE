"""Terminal Inverted Repeat (TIR) search engine using fast seed-and-extend and dynamic programming."""

from typing import List, Optional, Tuple
from deepise_ml.boundary.schemas import TIRMatch

DNA_COMPLEMENT = str.maketrans("ACGTUacgtuNn", "TGCAATGCAANn")


def reverse_complement(seq: str) -> str:
    """Return reverse complement of a DNA sequence."""
    return seq.translate(DNA_COMPLEMENT)[::-1]


def align_semiglobal(s1: str, s2: str, match_score: int = 2, mismatch_pen: int = 2, gap_pen: int = 3) -> Tuple[int, int, int, int, float]:
    """
    Semi-global alignment between two short DNA sequences.
    Returns: (score, aligned_length, mismatches, gaps, identity)
    """
    n, m = len(s1), len(s2)
    if n == 0 or m == 0:
        return 0, 0, 0, 0, 0.0

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = -i * gap_pen
    for j in range(1, m + 1):
        dp[0][j] = -j * gap_pen

    for i in range(1, n + 1):
        c1 = s1[i - 1]
        for j in range(1, m + 1):
            c2 = s2[j - 1]
            diag = dp[i - 1][j - 1] + (match_score if c1 == c2 else -mismatch_pen)
            up = dp[i - 1][j] - gap_pen
            left = dp[i][j - 1] - gap_pen
            dp[i][j] = max(diag, up, left)

    i, j = n, m
    matches = 0
    mismatches = 0
    gaps = 0

    while i > 0 and j > 0:
        score = dp[i][j]
        diag = dp[i - 1][j - 1]
        c1 = s1[i - 1]
        c2 = s2[j - 1]
        step_match = match_score if c1 == c2 else -mismatch_pen

        if score == diag + step_match:
            if c1 == c2:
                matches += 1
            else:
                mismatches += 1
            i -= 1
            j -= 1
        elif score == dp[i - 1][j] - gap_pen:
            gaps += 1
            i -= 1
        else:
            gaps += 1
            j -= 1

    gaps += i + j
    identity = matches / max(1, matches + mismatches)
    total_score = dp[n][m]
    return total_score, matches + mismatches + gaps, mismatches, gaps, identity


def find_candidate_tirs(
    contig: str,
    tpase_start: int,
    tpase_end: int,
    upstream_search_len: int = 450,
    downstream_search_len: int = 450,
    min_tir_len: int = 8,
    max_tir_len: int = 50,
    min_identity: float = 0.65,
    seed_k: int = 6,
    top_k: int = 8,
) -> List[TIRMatch]:
    """
    Search for Terminal Inverted Repeats flanking the transposase ORF using fast seed-and-extend.
    """
    contig_len = len(contig)
    up_win_start = max(0, tpase_start - upstream_search_len)
    up_win_end = tpase_start
    down_win_start = tpase_end
    down_win_end = min(contig_len, tpase_end + downstream_search_len)

    up_seq = contig[up_win_start:up_win_end].upper()
    down_seq = contig[down_win_start:down_win_end].upper()

    if len(up_seq) < min_tir_len or len(down_seq) < min_tir_len:
        return []

    # Filter out homopolymer / low-complexity k-mers
    kmer_pos = {}
    for i in range(len(up_seq) - seed_k + 1):
        km = up_seq[i : i + seed_k]
        # Ignore homopolymers
        if len(set(km)) <= 1:
            continue
        if km not in kmer_pos:
            kmer_pos[km] = []
        if len(kmer_pos[km]) < 4:
            kmer_pos[km].append(i)

    down_len = len(down_seq)
    candidates: List[TIRMatch] = []
    visited_pairs = set()

    max_mismatches = 3

    for d_i in range(down_len - seed_k + 1):
        fwd_km = down_seq[d_i : d_i + seed_k]
        if len(set(fwd_km)) <= 1:
            continue
        rc_km = reverse_complement(fwd_km)
        if rc_km not in kmer_pos:
            continue

        for u_i in kmer_pos[rc_km]:
            # Seed match at up_seq[u_i : u_i + seed_k] and revcomp(down_seq[d_i : d_i + seed_k])
            # Extend left from seed
            l_ext = 0
            mismatches = 0
            while (u_i - l_ext - 1 >= 0) and (d_i + seed_k + l_ext < down_len) and (seed_k + l_ext < max_tir_len):
                c_up = up_seq[u_i - l_ext - 1]
                c_down = down_seq[d_i + seed_k + l_ext]
                if c_up == reverse_complement(c_down):
                    l_ext += 1
                elif mismatches < max_mismatches:
                    mismatches += 1
                    l_ext += 1
                else:
                    break

            # Extend right from seed
            r_ext = 0
            while (u_i + seed_k + r_ext < len(up_seq)) and (d_i - r_ext - 1 >= 0) and (seed_k + l_ext + r_ext < max_tir_len):
                c_up = up_seq[u_i + seed_k + r_ext]
                c_down = down_seq[d_i - r_ext - 1]
                if c_up == reverse_complement(c_down):
                    r_ext += 1
                elif mismatches < max_mismatches:
                    mismatches += 1
                    r_ext += 1
                else:
                    break

            total_len = seed_k + l_ext + r_ext
            if total_len < min_tir_len:
                continue

            left_c_start = up_win_start + u_i - l_ext
            left_c_end = left_c_start + total_len
            right_c_end = down_win_start + d_i + seed_k + l_ext
            right_c_start = right_c_end - total_len

            pair_key = (left_c_start, right_c_end)
            if pair_key in visited_pairs:
                continue
            visited_pairs.add(pair_key)

            matches = total_len - mismatches
            identity = matches / total_len
            if identity >= min_identity:
                left_cand = contig[left_c_start:left_c_end].upper()
                right_cand = contig[right_c_start:right_c_end].upper()
                score = matches * 2.0 - mismatches * 2.5

                tir = TIRMatch(
                    left_start=left_c_start,
                    left_end=left_c_end,
                    right_start=right_c_start,
                    right_end=right_c_end,
                    left_seq=left_cand,
                    right_seq=right_cand,
                    length=total_len,
                    mismatches=mismatches,
                    gaps=0,
                    identity=round(identity, 3),
                    score=float(score),
                )
                candidates.append(tir)

    candidates.sort(key=lambda x: (x.score, x.identity * x.length), reverse=True)

    unique_candidates: List[TIRMatch] = []
    for c in candidates:
        overlap = False
        for u in unique_candidates:
            if abs(c.left_start - u.left_start) <= 3 and abs(c.right_end - u.right_end) <= 3:
                overlap = True
                break
        if not overlap:
            unique_candidates.append(c)
        if len(unique_candidates) >= top_k:
            break

    return unique_candidates
