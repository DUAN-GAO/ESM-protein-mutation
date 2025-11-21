#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, time, requests, sys
from esm import pretrained
import torch
import json
import gzip


# ---------------- User's compute_delta_score (unchanged) ----------------
def compute_delta_score(wildtype_sequence, mutation_position, wildtype_aa, mutant_aa):
    print(f"[DEBUG] compute_delta_score: pos={mutation_position}, wt={wildtype_aa}, mut={mutant_aa}, seq_len={len(wildtype_sequence)}")

    model, alphabet = pretrained.esm1b_t33_650M_UR50S()
    model.eval()

    batch_converter = alphabet.get_batch_converter()
    seq = list(wildtype_sequence)
    seq[mutation_position] = alphabet.get_tok(alphabet.mask_idx)
    masked_sequence_str = "".join(seq)

    data = [("protein", masked_sequence_str)]
    _, _, batch_tokens = batch_converter(data)

    with torch.no_grad():
        outputs = model(batch_tokens, repr_layers=[])
        logits = outputs["logits"]

    mask_logits = logits[0, mutation_position+1, :]
    probabilities = torch.softmax(mask_logits, dim=0)

    wt_idx = alphabet.get_idx(wildtype_aa)
    mut_idx = alphabet.get_idx(mutant_aa)

    p_wild = probabilities[wt_idx].item()
    p_mutant = probabilities[mut_idx].item()

    epsilon = 1e-10
    delta_score = torch.log(torch.tensor(p_mutant + epsilon) / torch.tensor(p_wild + epsilon)).item()

    print(f"[DEBUG] delta_score={delta_score}, p_wt={p_wild}, p_mut={p_mutant}")
    return delta_score, p_wild, p_mutant


# ---------------- normalize chromosome ----------------
def _normalize_chrom(chrom):
    chrom = str(chrom)
    if chrom.startswith("chr"):
        chrom = chrom[3:]
    return chrom


# ---------------- simple VCF loader ----------------
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


# ---------------- VEP annotation ----------------
def annotate_with_vep(variant_list, assembly="grch37", chunk_size=200):

    if assembly == "grch38":
        endpoint = "https://rest.ensembl.org/vep/homo_sapiens/region"
    else:
        endpoint = "https://rest.ensembl.org/vep/homo_sapiens/region/GRCh37"

    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    results = []

    for i in range(0, len(variant_list), chunk_size):

        chunk = variant_list[i:i + chunk_size]
        var_strs = []

        for v in chunk:
            chrom_clean = _normalize_chrom(v["chrom"])
            pos = v["pos"]
            ref = v["ref"]
            alt = v["alt"]
            var_strs.append(f"{chrom_clean} {pos} {pos} {ref}/{alt}")

        payload = {"variants": var_strs}

        for _ in range(3):
            try:
                r = requests.post(endpoint, headers=headers, data=json.dumps(payload))
                if r.status_code == 200:
                    break
            except Exception as e:
                print(f"[DEBUG] VEP request exception: {e}", file=sys.stderr)
            time.sleep(1)

        if r.status_code != 200:
            print(f"[DEBUG] VEP non-200 response: {r.status_code}", file=sys.stderr)
            for v in chunk:
                results.append({**v, "vep": None})
            continue

        try:
            data = r.json()
        except Exception:
            print("[DEBUG] VEP returned non-JSON", file=sys.stderr)
            for v in chunk:
                results.append({**v, "vep": None})
            continue

        for v, raw in zip(chunk, data):
            results.append({**v, "vep": raw})

    return results


# ---------------- HGVS parsing ----------------
def three_to_one(aa):
    table = {
        'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Gln':'Q','Glu':'E','Gly':'G',
        'His':'H','Ile':'I','Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P','Ser':'S',
        'Thr':'T','Trp':'W','Tyr':'Y','Val':'V','Sec':'U','Pyl':'O'
    }
    if len(aa)==1:
        return aa
    return table.get(aa, aa[0])


def parse_hgvsp(hgvsp):
    if not hgvsp:
        return None
    if ":p." in hgvsp:
        p = hgvsp.split(":p.")[1]
    elif "p." in hgvsp:
        p = hgvsp.split("p.")[1]
    else:
        p = hgvsp

    import re
    m = re.match(r"([A-Za-z]{1,3}|[A-Za-z])(\d+)([A-Za-z]{1,3}|[A-Za-z])", p)
    if not m:
        return None

    ref, pos, alt = m.group(1), int(m.group(2)), m.group(3)
    return {"ref": three_to_one(ref), "pos": pos, "alt": three_to_one(alt)}


# ---------------- fetch protein ----------------
def fetch_protein_seq(ensp_id):
    url = f"http://grch37.rest.ensembl.org/sequence/id/{ensp_id}"

    try:
        r = requests.get(
            url,
            headers={"Accept": "text/x-fasta"},
            params={"type": "protein"},
            timeout=20
        )
        if r.status_code == 200:
            lines = r.text.strip().splitlines()
            seq = "".join([l.strip() for l in lines if not l.startswith(">")])
            return seq

    except Exception as e:
        print(f"[DEBUG] fetch_protein_seq failed: {e}", file=sys.stderr)

    return None


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcf", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--assembly", default="grch37", choices=["grch37","grch38"])
    args = parser.parse_args()

    print(f"[DEBUG] Loading VCF {args.vcf}...")
    variants = load_vcf(args.vcf)
    print(f"[DEBUG] {len(variants)} variants loaded")

    print("[DEBUG] Querying VEP...")
    annotated = annotate_with_vep(variants, assembly=args.assembly)
    print(f"[DEBUG] VEP annotation done, {len(annotated)} records")

    # ---------------- write output ----------------
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["chrom","pos","ref","alt","delta_score"])

        for v in annotated:
            vep = v.get("vep")
            delta_score = None

            # -----------------------------------------------------------------
            # no VEP
            # -----------------------------------------------------------------
            if not vep:
                print(f"[DEBUG] No VEP result for {v['chrom']}:{v['pos']}")
                w.writerow([v["chrom"], v["pos"], v["ref"], v["alt"], None])
                continue

            tc_list = vep.get("transcript_consequences") or []
            if not tc_list:
                print(f"[DEBUG] No transcript_consequences for {v['chrom']}:{v['pos']}")
                w.writerow([v["chrom"], v["pos"], v["ref"], v["alt"], None])
                continue

            # -----------------------------------------------------------------
            # new logic: choose a usable transcript
            # -----------------------------------------------------------------
            tc = None

            # 1) canonical + protein_id + hgvsp
            for t in tc_list:
                if t.get("canonical") == 1 and t.get("protein_id") and t.get("hgvsp"):
                    tc = t
                    break

            # 2) any transcript with protein_id + hgvsp
            if not tc:
                for t in tc_list:
                    if t.get("protein_id") and t.get("hgvsp"):
                        tc = t
                        break

            # 3) fallback: transcript with protein_id only (hgvsp may be missing)
            if not tc:
                for t in tc_list:
                    if t.get("protein_id"):
                        tc = t
                        break

            if not tc:
                print(f"[DEBUG] No usable transcript for {v['chrom']}:{v['pos']}")
                w.writerow([v["chrom"], v["pos"], v["ref"], v["alt"], None])
                continue

            protein_id = tc.get("protein_id")
            hgvsp = tc.get("hgvsp")

            if not hgvsp:
                print(f"[DEBUG] Missing hgvsp for transcript {protein_id}")
                w.writerow([v["chrom"], v["pos"], v["ref"], v["alt"], None])
                continue

            parsed = parse_hgvsp(hgvsp)
            if not parsed:
                print(f"[DEBUG] Failed to parse HGVS {hgvsp} for {protein_id}")
                w.writerow([v["chrom"], v["pos"], v["ref"], v["alt"], None])
                continue

            seq = fetch_protein_seq(protein_id)
            if not seq:
                print(f"[DEBUG] No protein sequence for {protein_id}")
                w.writerow([v["chrom"], v["pos"], v["ref"], v["alt"], None])
                continue

            aa_pos = parsed["pos"] - 1
            wt = parsed["ref"]
            mut = parsed["alt"]

            try:
                delta_score, _, _ = compute_delta_score(seq, aa_pos, wt, mut)
            except Exception as e:
                print(f"[DEBUG] compute_delta_score failed for {protein_id}: {e}")
                delta_score = None

            w.writerow([v["chrom"], v["pos"], v["ref"], v["alt"], delta_score])


if __name__ == "__main__":
    main()
