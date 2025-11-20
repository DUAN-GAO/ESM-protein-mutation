#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import gzip
import time
import sys
import argparse
import requests


# -------------------------------
# Utility: Normalize chromosome
# -------------------------------
def _normalize_chrom(chrom):
    chrom = str(chrom)
    if chrom.startswith("chr"):
        chrom = chrom[3:]
    return chrom


# -------------------------------
# Compute delta score (your logic)
# -------------------------------
def compute_delta_score(ref_score, alt_score):
    """
    你提供的 compute_delta_score 逻辑（示例：alt - ref）
    如果你有真实逻辑，请把这里替换掉即可。
    """
    if ref_score is None or alt_score is None:
        return None
    return alt_score - ref_score


# -------------------------------
# Query VEP REST
# -------------------------------
def annotate_with_vep(variant_list, assembly="grch37", chunk_size=200):
    """
    使用 Ensembl VEP REST API 批量注释
    """
    if assembly == "grch38":
        endpoint = "https://rest.ensembl.org/vep/homo_sapiens/region"
    else:
        endpoint = "https://rest.ensembl.org/vep/homo_sapiens/region/GRCh37"

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    results = []

    for i in range(0, len(variant_list), chunk_size):
        chunk = variant_list[i:i + chunk_size]

        # --------------------------
        # FIXED: 正确 VEP region 格式
        # chrom start end ref/alt
        # --------------------------
        var_strs = []
        for v in chunk:
            chrom_clean = _normalize_chrom(v["chrom"])
            pos = v["pos"]
            ref = v["ref"]
            alt = v["alt"]

            # ***关键修复行***
            var_strs.append(f"{chrom_clean} {pos} {pos} {ref}/{alt}")

        payload = {"variants": var_strs}

        # retry until success
        for _ in range(3):
            r = requests.post(endpoint, headers=headers, data=json.dumps(payload))
            if r.status_code == 200:
                break
            time.sleep(1)

        if r.status_code != 200:
            print(f"[ERROR] VEP non-200 response: {r.status_code}", file=sys.stderr)
            print(r.text, file=sys.stderr)
            continue

        try:
            data = r.json()
        except Exception:
            print("[ERROR] VEP returned non-JSON.", file=sys.stderr)
            print(r.text, file=sys.stderr)
            continue

        for v, raw in zip(chunk, data):
            results.append({**v, "vep": raw})

    return results


# -------------------------------
# Load VCF (very simple parser)
# -------------------------------
def load_vcf(path):
    variants = []
    opener = gzip.open if path.endswith(".gz") else open

    with opener(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue

            fields = line.strip().split("\t")
            chrom, pos, vid, ref, alts = fields[0:5]

            for alt in alts.split(","):
                variants.append({
                    "chrom": chrom,
                    "pos": int(pos),
                    "ref": ref,
                    "alt": alt,
                })

    return variants


# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcf", required=True, help="Input VCF")
    parser.add_argument("--out", required=True, help="TSV output")
    parser.add_argument("--assembly", default="grch37", choices=["grch37", "grch38"])
    args = parser.parse_args()

    print("[INFO] Loading VCF...")
    variants = load_vcf(args.vcf)
    print(f"[INFO] Loaded {len(variants)} variants.")

    print("[INFO] Querying VEP...")
    annotated = annotate_with_vep(variants, assembly=args.assembly)

    print("[INFO] Writing output...")
    with open(args.out, "w") as w:
        w.write("chrom\tpos\tref\talt\tref_score\talt_score\tdelta_score\tvep_json\n")
        for v in annotated:
            vep = v.get("vep", {})

            # Demo: get scores (replace with your real logic)
            ref_score = vep.get("transcript_consequences", [{}])[0].get("sift_score")
            alt_score = vep.get("transcript_consequences", [{}])[0].get("polyphen_score")
            delta = compute_delta_score(ref_score, alt_score)

            w.write(
                f"{v['chrom']}\t{v['pos']}\t{v['ref']}\t{v['alt']}\t"
                f"{ref_score}\t{alt_score}\t{delta}\t"
                f"{json.dumps(vep)}\n"
            )

    print(f"[INFO] Done. Output → {args.out}")


if __name__ == "__main__":
    main()
