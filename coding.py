#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, time, requests, sys
from esm import pretrained
import torch
import gzip

# ---------------- compute_delta_score ----------------
def compute_delta_score(seq, aa_pos, wt, mut):
    model, alphabet = pretrained.esm1b_t33_650M_UR50S()
    model.eval()

    batch_converter = alphabet.get_batch_converter()
    masked_seq = list(seq)
    masked_seq[aa_pos] = alphabet.get_tok(alphabet.mask_idx)
    masked_seq_str = "".join(masked_seq)
    data = [("protein", masked_seq_str)]
    _, _, batch_tokens = batch_converter(data)

    with torch.no_grad():
        outputs = model(batch_tokens, repr_layers=[])
        logits = outputs["logits"]

    mask_logits = logits[0, aa_pos + 1, :]
    probs = torch.softmax(mask_logits, dim=0)
    p_wt = probs[alphabet.get_idx(wt)].item()
    p_mut = probs[alphabet.get_idx(mut)].item()

    epsilon = 1e-10
    delta_score = torch.log(torch.tensor(p_mut + epsilon) / torch.tensor(p_wt + epsilon)).item()
    return delta_score

# ---------------- helpers ----------------
def _normalize_chrom(chrom):
    chrom = str(chrom)
    if chrom.lower().startswith("chr"):
        chrom = chrom[3:]
    return chrom

def open_vcf(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")

def load_vcf(path):
    variants = []
    with open_vcf(path) as f:
        for line in f:
            if line.startswith("#"): continue
            fields = line.strip().split("\t")
            if len(fields) < 5: continue
            chrom, pos, _, ref, alts = fields[0:5]
            for alt in alts.split(","):
                variants.append({"chrom": chrom, "pos": int(pos), "ref": ref, "alt": alt})
    return variants

# ---------------- VEP annotation ----------------
def annotate_with_vep(variants, chunk_size=200, pause=0.2):
    endpoint = "https://rest.ensembl.org/vep/human/region"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    results = []

    for i in range(0, len(variants), chunk_size):
        chunk = variants[i:i+chunk_size]
        var_strs = [f"{_normalize_chrom(v['chrom'])} {v['pos']} . {v['ref']} {v['alt']}" for v in chunk]
        payload = {"variants": var_strs}

        r = None
        for attempt in range(4):
            try:
                r = requests.post(endpoint, headers=headers, json=payload, timeout=30)
                if r.status_code == 200: break
                elif r.status_code == 429: time.sleep((2**attempt)*1.0)
                else: time.sleep(0.5)
            except Exception as e:
                print(f"[DEBUG] VEP request exception: {e}", file=sys.stderr)
                time.sleep(0.5)

        if r is None or r.status_code != 200:
            for v in chunk: results.append({**v, "vep": None})
            continue

        try:
            data = r.json()
        except Exception:
            for v in chunk: results.append({**v, "vep": None})
            continue

        for v, raw in zip(chunk, data):
            results.append({**v, "vep": raw})

        time.sleep(pause)
    return results

# ---------------- HGVS parsing ----------------
def three_to_one(aa):
    table = {'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Gln':'Q','Glu':'E','Gly':'G',
             'His':'H','Ile':'I','Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P','Ser':'S',
             'Thr':'T','Trp':'W','Tyr':'Y','Val':'V','Sec':'U','Pyl':'O'}
    if aa is None: return None
    aa = str(aa)
    if len(aa)==1: return aa
    return table.get(aa, aa[0] if len(aa)>0 else None)

def parse_hgvsp(hgvsp):
    if not hgvsp: return None
    if ":p." in hgvsp: p = hgvsp.split(":p.")[1]
    elif "p." in hgvsp: p = hgvsp.split("p.")[1]
    else: p = hgvsp
    import re
    m = re.match(r"([A-Za-z]{1,3}|[A-Za-z])(\d+)([A-Za-z]{1,3}|\*|[A-Za-z])", p)
    if not m: return None
    ref, pos, alt = m.group(1), int(m.group(2)), m.group(3)
    return {"ref": three_to_one(ref), "pos": pos, "alt": three_to_one(alt) if alt!="*" else "*"}

# ---------------- fetch protein sequence ----------------
def fetch_protein_seq(ensp_id):
    url = f"https://rest.ensembl.org/sequence/id/{ensp_id}"
    try:
        r = requests.get(url, headers={"Accept": "text/x-fasta"}, params={"type":"protein"}, timeout=20)
        if r.status_code==200:
            lines = r.text.strip().splitlines()
            return "".join([l.strip() for l in lines if not l.startswith(">")])
    except Exception as e:
        print(f"[DEBUG] fetch_protein_seq failed: {e}", file=sys.stderr)
    return None

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcf", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    variants = load_vcf(args.vcf)
    print(f"[INFO] Loaded {len(variants)} variants from {args.vcf}")

    annotated = annotate_with_vep(variants)
    print(f"[INFO] VEP annotation done, {len(annotated)} records")

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["chrom","pos","ref","alt","delta_score"])

        for v in annotated:
            vep = v.get("vep")
            if not vep:
                w.writerow([v["chrom"], v["pos"], v["ref"], v["alt"], None])
                continue

            tc_list = vep.get("transcript_consequences") or []
            protein_tcs = [t for t in tc_list if t.get("biotype")=="protein_coding" or t.get("gene_biotype")=="protein_coding"]

            tc = None
            for t in protein_tcs:
                if t.get("canonical")==1 and t.get("protein_id") and t.get("hgvsp"):
                    tc = t; break
            if not tc:
                for t in protein_tcs:
                    if t.get("protein_id") and t.get("hgvsp"):
                        tc = t; break
            if not tc and protein_tcs:
                tc = protein_tcs[0]

            if not tc:
                w.writerow([v["chrom"], v["pos"], v["ref"], v["alt"], None])
                continue

            protein_id = tc.get("protein_id")
            parsed = parse_hgvsp(tc.get("hgvsp")) if tc.get("hgvsp") else None
            if not parsed and tc.get("amino_acids") and tc.get("protein_start"):
                aa = tc["amino_acids"].split("/")
                if len(aa)==2:
                    parsed = {"ref": three_to_one(aa[0]), "pos": int(tc["protein_start"]), "alt": three_to_one(aa[1])}

            if not parsed:
                w.writerow([v["chrom"], v["pos"], v["ref"], v["alt"], None])
                continue

            seq = fetch_protein_seq(protein_id)
            if not seq:
                w.writerow([v["chrom"], v["pos"], v["ref"], v["alt"], None])
                continue

            aa_pos = parsed["pos"]-1
            if aa_pos<0 or aa_pos>=len(seq):
                w.writerow([v["chrom"], v["pos"], v["ref"], v["alt"], None])
                continue

            try:
                delta = compute_delta_score(seq, aa_pos, parsed["ref"], parsed["alt"])
            except Exception as e:
                print(f"[DEBUG] delta_score failed: {e}", file=sys.stderr)
                delta = None

            w.writerow([v["chrom"], v["pos"], v["ref"], v["alt"], delta])

if __name__=="__main__":
    main()
