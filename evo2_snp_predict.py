#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Zero-shot SNP deleteriousness prediction using Evo 2 (HG38)
Sequence fetched via Ensembl REST API

Input:
    chrom, pos (1-based, hg38), ref, alt
Output:
    delta_score = logP(ALT) - logP(REF)
"""

import requests
# from evo2.models import Evo2


# ======================
# Config
# ======================
WINDOW_SIZE = 8192
ENSEMBL_SERVER = "https://rest.ensembl.org"


# ======================
# Fetch sequence from API
# ======================
def fetch_sequence_api(chrom, start, end):
    """
    Fetch hg38 genomic sequence from Ensembl REST API
    chrom: "chr1" or "1"
    start/end: 1-based inclusive
    """
    chrom = chrom.replace("chr", "")
    region = f"{chrom}:{start}..{end}"

    url = f"{ENSEMBL_SERVER}/sequence/region/human/{region}"
    headers = {"Content-Type": "text/plain"}

    r = requests.get(url, headers=headers, timeout=30)
    if not r.ok:
        raise RuntimeError(f"API error: {r.status_code} {r.text}")

    return r.text.strip().upper()


# ======================
# Build sequences
# ======================
def build_sequences_api(chrom, pos, ref, alt):
    """
    pos: 1-based (hg38)
    """
    half = WINDOW_SIZE // 2
    start = max(1, pos - half)
    end = pos + half

    ref_seq = fetch_sequence_api(chrom, start, end)
    print(ref_seq)  # test API
    snp_idx = pos - start  # 0-based index in window

    if ref_seq[snp_idx] != ref.upper():
        raise ValueError(
            f"Reference mismatch: genome={ref_seq[snp_idx]}, input={ref}"
        )

    alt_seq = (
        ref_seq[:snp_idx]
        + alt.upper()
        + ref_seq[snp_idx + 1:]
    )

    return ref_seq, alt_seq


# ======================
# Evo2 scoring
# ======================
def score_snp_hg38_api(
    chrom,
    pos,
    ref,
    alt,
    model_name="evo2_1b_base",
):
    # model = Evo2(model_name)  test

    ref_seq, alt_seq = build_sequences_api(
        chrom, pos, ref, alt
    )

    ref_score = model.score_sequences([ref_seq])[0]
    alt_score = model.score_sequences([alt_seq])[0]

    delta = alt_score - ref_score

    return {
        "chrom": chrom,
        "pos_hg38": pos,
        "ref": ref,
        "alt": alt,
        "ref_score": ref_score,
        "alt_score": alt_score,
        "delta_score": delta,
    }


# ======================
# Example usage
# ======================
if __name__ == "__main__":

    # ======= MODIFY HERE =======
    chrom = "chr2"
    pos = 155827701      # hg38
    ref = "A"
    alt = "C"
    model_name = "evo2_1b_base"
    # ==========================

    result = score_snp_hg38_api(
        chrom=chrom,
        pos=pos,
        ref=ref,
        alt=alt,
        model_name=model_name,
    )

    print("\n=== Evo2 SNP effect prediction (hg38, API) ===")
    for k, v in result.items():
        print(f"{k}: {v}")

    print("\nInterpretation:")
    d = result["delta_score"]
    if d < -2:
        print("  Strongly deleterious")
    elif d < -0.5:
        print("  Moderately deleterious")
    elif d < 0:
        print("  Weak effect")
    else:
        print("  Likely neutral")
