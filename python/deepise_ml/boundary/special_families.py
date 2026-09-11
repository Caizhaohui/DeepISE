"""Specialized non-canonical boundary detection modules for Plan A."""

import re
from typing import List, Optional, Tuple
from deepise_ml.boundary.schemas import StructureEvidence
from deepise_ml.boundary.tir import reverse_complement


def find_hairpins(
    seq: str,
    min_stem: int = 5,
    max_stem: int = 10,
    min_loop: int = 3,
    max_loop: int = 8,
    allow_mismatch: int = 1,
) -> List[Tuple[float, int, int, str, str, str]]:
    """
    Search for stem-loop / hairpin secondary structures.
    Returns list of (score, start_idx, end_idx, stem_left, loop, stem_right) sorted by score.
    """
    results = []
    n = len(seq)
    for s_len in range(min_stem, max_stem + 1):
        for i in range(n - 2 * s_len - min_loop + 1):
            stem_l = seq[i : i + s_len]
            stem_l_rc = reverse_complement(stem_l)
            for l_len in range(min_loop, max_loop + 1):
                j = i + s_len + l_len
                if j + s_len <= n:
                    stem_r = seq[j : j + s_len]
                    mismatches = sum(1 for a, b in zip(stem_r, stem_l_rc) if a != b)
                    if mismatches <= allow_mismatch:
                        gc = (stem_l.count("G") + stem_l.count("C")) / s_len
                        # Hairpin score favoring GC rich stable stems and compact loops
                        score = s_len * 2.5 - mismatches * 3.0 + gc * 2.0 - abs(l_len - 4) * 0.5
                        results.append((score, i, j + s_len, stem_l, seq[i + s_len : j], stem_r))

    results.sort(key=lambda x: x[0], reverse=True)
    return results


def detect_is200_is605_boundary(
    contig: str,
    tpase_start: int,
    tpase_end: int,
) -> Tuple[int, int, float, StructureEvidence]:
    """
    Detect boundaries of IS200/IS605 family elements via 3' terminal hairpin and 5' motif.
    HUH transposase mediated transposition, zero TSD, zero canonical TIR.
    """
    contig_len = len(contig)
    # Search 3' hairpin in window downstream of Tpase [tpase_end : min(contig_len, tpase_end + 300)]
    down_win_start = tpase_end
    down_win_end = min(contig_len, tpase_end + 300)
    down_seq = contig[down_win_start:down_win_end].upper()

    hairpins = find_hairpins(down_seq, min_stem=5, max_stem=9, min_loop=3, max_loop=7, allow_mismatch=1)
    
    if hairpins:
        best_hp = hairpins[0]
        hp_score, hp_rel_s, hp_rel_e, stem_l, loop, stem_r = best_hp
        # Biological boundary in IS200/IS605: typically 2-15 bp downstream of the 3' hairpin
        # Check for trailing poly-T or terminal cleavage motif (e.g. 5'-TTAA or G-tail)
        post_hp_seq = down_seq[hp_rel_e : min(len(down_seq), hp_rel_e + 20)]
        extra_offset = 6
        if len(post_hp_seq) >= 4:
            # Check for cleavage boundary
            for k in range(len(post_hp_seq) - 1):
                if post_hp_seq[k : k + 2] in ("TT", "TA", "AA"):
                    extra_offset = k + 2
                    break
        pred_end = down_win_start + hp_rel_e + extra_offset
        hairpin_str = f"{stem_l}-{loop}-{stem_r}"
    else:
        # Fallback: expected 3' distance for IS200/IS605 is ~60 bp
        pred_end = min(contig_len, tpase_end + 65)
        hp_score = 5.0
        hairpin_str = "unresolved"

    # Search 5' boundary: upstream window [max(0, tpase_start - 250) : tpase_start]
    up_win_start = max(0, tpase_start - 250)
    up_seq = contig[up_win_start:tpase_start].upper()
    
    # 5' end of IS200/IS605 typically starts with (T/C)T(A/T)(A/T) or contains a 5' subterminal hairpin
    motifs_5p = ["TTAA", "TTAT", "TTAC", "CTTA", "CTAT", "TGTC", "TTTG"]
    best_5p_offset = None
    for m in motifs_5p:
        matches = [m_iter.start() for m_iter in re.finditer(m, up_seq)]
        # Filter for realistic distance (typically 30-180 bp upstream of Tpase)
        valid_matches = [pos for pos in matches if 30 <= (tpase_start - (up_win_start + pos)) <= 200]
        if valid_matches:
            # Prefer the leftmost valid motif
            best_5p_offset = valid_matches[0]
            break

    if best_5p_offset is not None:
        pred_start = up_win_start + best_5p_offset
        motif_left = up_seq[best_5p_offset : best_5p_offset + 4]
        motif_score = 8.0
    else:
        # Fallback: typical 5' distance is ~65 bp
        pred_start = max(0, tpase_start - 65)
        motif_left = "predicted"
        motif_score = 4.0

    stability = float(round(hp_score + motif_score, 2))
    conf = min(0.95, max(0.40, stability / 25.0))

    evidence = StructureEvidence(
        feature_type="hairpin_stem_loop",
        motif_left=motif_left,
        motif_right=None,
        hairpin_left=None,
        hairpin_right=hairpin_str,
        stability_score=stability,
        notes="IS200/IS605 subterminal stem-loop hairpin identified at 3' non-coding region.",
    )
    return pred_start, pred_end, conf, evidence


def detect_is91_boundary(
    contig: str,
    tpase_start: int,
    tpase_end: int,
) -> Tuple[int, int, float, StructureEvidence]:
    """
    Detect boundaries of IS91 rolling-circle elements via oriIS and terIS motifs.
    """
    contig_len = len(contig)
    # 3' terIS search downstream of Tpase [tpase_end : min(contig_len, tpase_end + 300)]
    down_win_start = tpase_end
    down_win_end = min(contig_len, tpase_end + 300)
    down_seq = contig[down_win_start:down_win_end].upper()

    # Conserved terIS motifs: TTCCTAT(A/C/G)CGT or CCTAT(A/C/G)CGT or GAATTTCCTAT
    ter_patterns = [
        r"TTCCTAT[ACGT]CGT",
        r"CCTAT[ACGT]CGT",
        r"GAATTTCCTAT",
        r"TTTCAATTTCCT",
        r"TTCCTTTTATGTCG",
    ]
    best_ter_pos = None
    ter_motif_found = None
    for pat in ter_patterns:
        m = re.search(pat, down_seq)
        if m:
            best_ter_pos = down_win_start + m.end()
            ter_motif_found = m.group(0)
            break

    if best_ter_pos is not None:
        pred_end = best_ter_pos
        ter_score = 15.0
    else:
        # Fallback for IS91: average downstream distance is ~120 bp
        pred_end = min(contig_len, tpase_end + 120)
        ter_score = 5.0
        ter_motif_found = "predicted_ter"

    # 5' oriIS search upstream of Tpase [max(0, tpase_start - 250) : tpase_start]
    up_win_start = max(0, tpase_start - 250)
    up_seq = contig[up_win_start:tpase_start].upper()
    
    # Conserved oriIS 5' motifs (often GTAC, CTTG, GCCGC, GAACC, AAAC)
    ori_patterns = [r"GCCGCCT", r"GAACCACG", r"AAACAAATC", r"GTAC", r"CTTG"]
    best_ori_pos = None
    ori_motif_found = None
    for pat in ori_patterns:
        matches = [m.start() for m in re.finditer(pat, up_seq)]
        valid = [p for p in matches if 40 <= (tpase_start - (up_win_start + p)) <= 180]
        if valid:
            best_ori_pos = up_win_start + valid[0]
            ori_motif_found = pat
            break

    if best_ori_pos is not None:
        pred_start = best_ori_pos
        ori_score = 10.0
    else:
        pred_start = max(0, tpase_start - 90)
        ori_score = 5.0
        ori_motif_found = "predicted_ori"

    stability = float(ter_score + ori_score)
    conf = min(0.95, max(0.45, stability / 25.0))

    evidence = StructureEvidence(
        feature_type="is91_ori_ter",
        motif_left=ori_motif_found,
        motif_right=ter_motif_found,
        hairpin_left=None,
        hairpin_right=None,
        stability_score=stability,
        notes="IS91 rolling-circle oriIS / terIS termination sequence detected.",
    )
    return pred_start, pred_end, conf, evidence


def detect_is110_boundary(
    contig: str,
    tpase_start: int,
    tpase_end: int,
) -> Tuple[int, int, float, StructureEvidence]:
    """
    Detect boundaries of IS110 recombinase elements via subterminal core recombination motifs.
    IS110 has no TSD, characteristic ~80 bp 5' distance and ~280 bp 3' distance.
    """
    contig_len = len(contig)
    # Search 3' terminal motif in [tpase_end + 200 : min(contig_len, tpase_end + 380)]
    down_win_start = min(contig_len, tpase_end + 200)
    down_win_end = min(contig_len, tpase_end + 380)
    down_seq = contig[down_win_start:down_win_end].upper()

    # IS110 3' terminal consensus: (G/A)(G/A/T)C(C/A)TCCAT(A/T)(T/A)A or (T/C)CA(T/G)(A/T)(T/G)A
    recomb_3p = [
        r"[ATGC]{2}TCCATATA",
        r"[ATGC]{2}CCCTTATA",
        r"[ATGC]{2}CCAT[ACG]T[AT]",
        r"CCATATA",
        r"CCATGTA",
        r"CCATACA",
        r"CCATAGA",
    ]
    best_3p_end = None
    motif_3p_str = None
    for pat in recomb_3p:
        m = re.search(pat, down_seq)
        if m:
            best_3p_end = down_win_start + m.end()
            motif_3p_str = m.group(0)
            break

    if best_3p_end is not None:
        pred_end = best_3p_end
        score_3p = 12.0
    else:
        pred_end = min(contig_len, tpase_end + 280)
        score_3p = 5.0
        motif_3p_str = "predicted_3p"

    # Search 5' terminal motif in [max(0, tpase_start - 160) : max(0, tpase_start - 40)]
    up_win_start = max(0, tpase_start - 160)
    up_win_end = max(0, tpase_start - 40)
    up_seq = contig[up_win_start:up_win_end].upper()

    recomb_5p = [
        r"AATG[AG]A[AT]GGA",
        r"TGATGGAATGGA",
        r"AATGGTATGGA",
        r"ATGGA[GC]TGGA",
        r"TAATG[AG]",
        r"CAATG[AG]",
        r"GAATG[AG]",
    ]
    best_5p_start = None
    motif_5p_str = None
    for pat in recomb_5p:
        m = re.search(pat, up_seq)
        if m:
            best_5p_start = up_win_start + m.start()
            motif_5p_str = m.group(0)
            break

    if best_5p_start is not None:
        pred_start = best_5p_start
        score_5p = 10.0
    else:
        pred_start = max(0, tpase_start - 85)
        score_5p = 5.0
        motif_5p_str = "predicted_5p"

    stability = float(score_5p + score_3p)
    conf = min(0.95, max(0.40, stability / 25.0))

    evidence = StructureEvidence(
        feature_type="is110_recombination",
        motif_left=motif_5p_str,
        motif_right=motif_3p_str,
        hairpin_left=None,
        hairpin_right=None,
        stability_score=stability,
        notes="IS110 recombinase subterminal core recombination boundary motif detected.",
    )
    return pred_start, pred_end, conf, evidence
