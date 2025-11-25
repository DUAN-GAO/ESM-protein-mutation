#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, time, requests, sys, math
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

    mask_logits = logits[0, mutation_position + 1, :]
    probabilities = torch.softmax(mask_logits, dim=0)

    wt_idx = alphabet.get_idx(wildtype_aa)
    mut_idx = alphabet.get_idx(mutant_aa)

    p_wild = probabilities[wt_idx].item()
    p_mutant = probabilities[mut_idx].item()

    epsilon = 1e-10
    delta_score = torch.log(torch.tensor(p_mutant + epsilon) / torch.tensor(p_wild + epsilon)).item()

    print(f"[DEBUG] delta_score={delta_score}, p_wt={p_wild}, p_mut={p_mutant}")
    return delta_score, p_wild, p_mutant

# ---------------- helpers ----------------
def _normalize_chrom(chrom):
    chrom = str(chrom)
    if chrom.lower().startswith("chr"):
        chrom = chrom[3:]
    return chrom

def open_vcf(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")

# ---------------- simple VCF loader ----------------
def load_vcf(path):
    variants = []
    with open_vcf(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            if len(fields) < 5:
                continue
            chrom, pos, vid, ref, alts = fields[0:5]
            for alt in alts.split(","):
                variants.append({
                    "chrom": chrom,
                    "pos": int(pos),
                    "ref": ref,
                    "alt": alt,
                })
    return variants

# ---------------- VEP annotation (HG38) ----------------
def annotate_with_vep(variant_list, assembly="grch38", chunk_size=200, pause=0.2):
    # use HG38 endpoint
    endpoint = "https://rest.ensembl.org/vep/homo_sapiens/region"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    results = []

    for i in range(0, len(variant_list), chunk_size):
        chunk = variant_list[i:i+chunk_size]
        var_strs = []
        for v in chunk:
            chrom_clean = _normalize_chrom(v["chrom"])
            pos = v["pos"]
            ref = v["ref"]
            alt = v["alt"]
            # HG38 region format expected: "1:12345 A/T"
            var_strs.append(f"{chrom_clean}:{pos} {ref}/{alt}")

        payload = {"variants": var_strs}

        # retry with exponential backoff for 429/other transient errors
        r = None
        for attempt in range(4):
            try:
                r = requests.post(endpoint, headers=headers, json=payload, timeout=30)
                if r.status_code == 200:
                    break
                # if rate limited, backoff more
                if r.status_code == 429:
                    wait = (2 ** attempt) * 1.0
                    print(f"[DEBUG] VEP 429 rate limit, sleeping {wait}s", file=sys.stderr)
                    time.sleep(wait)
                else:
                    # small pause then retry
                    time.sleep(0.5)
            except Exception as e:
                print(f"[DEBUG] VEP request exception: {e}", file=sys.stderr)
                time.sleep(0.5)

        if r is None:
            for v in chunk:
                results.append({**v, "vep": None})
            continue

        if r.status_code != 200:
            # print body for debugging
            body = None
            try:
                body = r.text
            except Exception:
                body = "<no body>"
            print(f"[DEBUG] VEP non-200 ({r.status_code}): {body}", file=sys.stderr)
            for v in chunk:
                results.append({**v, "vep": None})
            time.sleep(pause)
            continue

        try:
            data = r.json()
        except Exception as e:
            print(f"[DEBUG] VEP returned non-JSON: {e}", file=sys.stderr)
            print(f"[DEBUG] raw text: {r.text}", file=sys.stderr)
            for v in chunk:
                results.append({**v, "vep": None})
            time.sleep(pause)
            continue

        for v, raw in zip(chunk, data):
            results.append({**v, "vep": raw})

        time.sleep(pause)
    return results

# ---------------- HGVS parsing + fallback ----------------
def three_to_one(aa):
    table = {
        'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Gln':'Q','Glu':'E','Gly':'G',
        'His':'H','Ile':'I','Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P','Ser':'S',
        'Thr':'T','Trp':'W','Tyr':'Y','Val':'V','Sec':'U','Pyl':'O'
    }
    if aa is None:
        return None
    aa = str(aa)
    if len(aa) == 1:
        return aa
    # try mapping three-letter to one-letter
    return table.get(aa, aa[0] if len(aa) > 0 else None)

def parse_hgvsp(hgvsp):
    """Parse p.Ala123Val or similar. Return dict or None."""
    if not hgvsp:
        return None
    if ":p." in hgvsp:
        p = hgvsp.split(":p.")[1]
    elif "p." in hgvsp:
        p = hgvsp.split("p.")[1]
    else:
        p = hgvsp

    # accept stop '*' or Ter, and three-letter or one-letter aa
    import re
    # patterns: Trp50*  or W50*  or Ala50Val  or A50V
    m = re.match(r"([A-Za-z]{1,3}|[A-Za-z])(\d+)([A-Za-z]{1,3}|\*|[A-Za-z])", p)
    if not m:
        return None
    ref_raw, pos_raw, alt_raw = m.group(1), m.group(2), m.group(3)
    try:
        pos = int(pos_raw)
    except Exception:
        return None
    ref = three_to_one(ref_raw)
    alt = three_to_one(alt_raw) if alt_raw != "*" else "*"
    return {"ref": ref, "pos": pos, "alt": alt}

# ---------------- fetch protein ----------------
def fetch_protein_seq(ensp_id):
    # hg38 endpoint (rest.ensembl.org)
    url = f"https://rest.ensembl.org/sequence/id/{ensp_id}"
    try:
        r = requests.get(url, headers={"Accept": "text/x-fasta"}, params={"type": "protein"}, timeout=20)
        if r.status_code == 200:
            lines = r.text.strip().splitlines()
            seq = "".join([l.strip() for l in lines if not l.startswith(">")])
            return seq
        else:
            print(f"[DEBUG] fetch_protein_seq non-200 {r.status_code} for {ensp_id}: {r.text[:200]}", file=sys.stderr)
    except Exception as e:
        print(f"[DEBUG] fetch_protein_seq failed: {e}", file=sys.stderr)
    return None

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcf", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--assembly", default="grch38", choices=["grch38"])
    args = parser.parse_args()

    print(f"[DEBUG] Loading VCF {args.vcf}...")
    variants = load_vcf(args.vcf)
    print(f"[DEBUG] {len(variants)} variants loaded")

    print("[DEBUG] Querying VEP (hg38)...")
    annotated = annotate_with_vep(variants, assembly=args.assembly)
    print(f"[DEBUG] VEP annotation done, {len(annotated)} records")

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["chrom","pos","ref","alt","delta_score"])

        for v in annotated:
            vep = v.get("vep")
            delta_score = None

            if not vep:
                print(f"[DEBUG] No VEP result for {v['chrom']}:{v['pos']}", file=sys.stderr)
                w.writerow([v["chrom"], v["pos"], v["ref"], v["alt"], None])
                continue

            tc_list = vep.get("transcript_consequences") or []
            if not tc_list:
                print(f"[DEBUG] No transcript_consequences for {v['chrom']}:{v['pos']}", file=sys.stderr)
                # for debugging, dump short vep
                if "input" in vep:
                    print(f"[DEBUG] VEP input echoed: {vep.get('input')}", file=sys.stderr)
                w.writerow([v["chrom"], v["pos"], v["ref"], v["alt"], None])
                continue

            # Prefer protein_coding transcripts
            protein_tcs = [t for t in tc_list if (t.get("biotype") == "protein_coding" or t.get("gene_biotype") == "protein_coding")]

            # choose transcript: priority:
            # 1) canonical & protein_coding & protein_id & hgvsp
            # 2) any protein_coding with protein_id & hgvsp
            # 3) any protein_coding with protein_id (use amino_acids + protein_start fallback)
            # 4) any with protein_id & hgvsp
            # 5) any with protein_id (amino_acids fallback)
            tc = None

            def has_protein_and_hgvsp(t): return t.get("protein_id") and t.get("hgvsp")
            def has_protein(t): return t.get("protein_id")

            search_lists = [protein_tcs if protein_tcs else tc_list, tc_list]
            # first pass: want hgvsp
            for lst in search_lists:
                for t in lst:
                    if t.get("canonical") == 1 and has_protein_and_hgvsp(t):
                        tc = t; break
                if tc: break
            if not tc:
                for lst in search_lists:
                    for t in lst:
                        if has_protein_and_hgvsp(t):
                            tc = t; break
                    if tc: break
            if not tc:
                for lst in search_lists:
                    for t in lst:
                        if t.get("canonical") == 1 and has_protein(t):
                            tc = t; break
                    if tc: break
            if not tc:
                for lst in search_lists:
                    for t in lst:
                        if has_protein(t):
                            tc = t; break
                    if tc: break

            if not tc:
                print(f"[DEBUG] No usable transcript for {v['chrom']}:{v['pos']}", file=sys.stderr)
                # dump one tc for debugging
                if tc_list:
                    print(f"[DEBUG] example tc keys: {list(tc_list[0].keys())}", file=sys.stderr)
                w.writerow([v["chrom"], v["pos"], v["ref"], v["alt"], None])
                continue

            protein_id = tc.get("protein_id")
            hgvsp = tc.get("hgvsp")

            # primary path: parse hgvsp
            parsed = parse_hgvsp(hgvsp) if hgvsp else None

            # fallback: use amino_acids and protein_start
            if not parsed:
                amino = tc.get("amino_acids")  # "G/A" style
                pstart = tc.get("protein_start") or tc.get("protein_start")
                if amino and pstart:
                    # amino like "G/A" or "G/A"
                    parts = amino.split("/")
                    if len(parts) == 2:
                        wt_raw, mut_raw = parts[0], parts[1]
                        try:
                            pos = int(pstart)
                            parsed = {"ref": three_to_one(wt_raw), "pos": pos, "alt": three_to_one(mut_raw)}
                            print(f"[DEBUG] Fallback parsed from amino_acids for {protein_id}: {parsed}", file=sys.stderr)
                        except Exception:
                            parsed = None

            if not parsed:
                print(f"[DEBUG] Missing or unparsable protein change for transcript {protein_id} (hgvsp={hgvsp}, amino_acids={tc.get('amino_acids')}, protein_start={tc.get('protein_start')})", file=sys.stderr)
                w.writerow([v["chrom"], v["pos"], v["ref"], v["alt"], None])
                continue

            # fetch sequence
            seq = fetch_protein_seq(protein_id)
            if not seq:
                print(f"[DEBUG] No protein sequence for {protein_id}", file=sys.stderr)
                w.writerow([v["chrom"], v["pos"], v["ref"], v["alt"], None])
                continue

            aa_pos = parsed["pos"] - 1
            wt = parsed["ref"]
            mut = parsed["alt"]

            # bounds check
            if aa_pos < 0 or aa_pos >= len(seq):
                print(f"[DEBUG] aa_pos {aa_pos} out of range for {protein_id} (len={len(seq)})", file=sys.stderr)
                w.writerow([v["chrom"], v["pos"], v["ref"], v["alt"], None])
                continue

            try:
                delta_score, _, _ = compute_delta_score(seq, aa_pos, wt, mut)
            except Exception as e:
                print(f"[DEBUG] compute_delta_score failed for {protein_id}: {e}", file=sys.stderr)
                delta_score = None

            w.writerow([v["chrom"], v["pos"], v["ref"], v["alt"], delta_score])

if __name__ == "__main__":
    main()
