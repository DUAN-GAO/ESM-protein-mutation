#!/usr/bin/env python3
"""
VCF -> ESM delta score using user's original compute_delta_score function.

Input:
    --vcf input.vcf
Output:
    --out output.tsv
Notes:
    - HG19 / GRCh37 coordinates
    - Only SNPs
    - Online access to Ensembl GRCh37 REST API
    - compute_delta_score is embedded directly
"""

import argparse, csv, time, requests, sys
from collections import defaultdict
import torch
from esm import pretrained

# ---------------- User's compute_delta_score ----------------
def compute_delta_score(wildtype_sequence, mutation_position, wildtype_aa, mutant_aa):
    # 加载预训练模型 ESM-1b
    model, alphabet = pretrained.esm1b_t33_650M_UR50S()
    model.eval()

    # 构建批次转换器
    batch_converter = alphabet.get_batch_converter()

    # 1. 生成带掩码的序列
    seq = list(wildtype_sequence)
    seq[mutation_position] = alphabet.get_tok(alphabet.mask_idx)
    masked_sequence_str = "".join(seq)

    # 2. 转换为Token格式
    data = [("protein", masked_sequence_str)]
    _, _, batch_tokens = batch_converter(data)

    # 3. 输入模型
    with torch.no_grad():
        outputs = model(batch_tokens, repr_layers=[])
        logits = outputs["logits"]

    # 4. 提取突变位点logits并softmax成概率
    mask_logits = logits[0, mutation_position+1, :]
    probabilities = torch.softmax(mask_logits, dim=0)

    # 5. 获取野生型和突变型氨基酸概率
    wt_idx = alphabet.get_idx(wildtype_aa)
    mut_idx = alphabet.get_idx(mutant_aa)

    p_wild = probabilities[wt_idx].item()
    p_mutant = probabilities[mut_idx].item()

    # 6. 计算Δscore
    epsilon = 1e-10
    delta_score = torch.log(torch.tensor(p_mutant + epsilon) / torch.tensor(p_wild + epsilon)).item()

    return delta_score, p_wild, p_mutant

# ---------------- Ensembl GRCh37 REST ----------------
ENSEMBL_GRCH37 = "http://grch37.rest.ensembl.org"
VEP_ENDPOINT = "/vep/human/region"
HEADERS_JSON = {"Content-Type": "application/json", "Accept": "application/json"}
HEADERS_FASTA = {"Accept": "text/x-fasta"}

# ---------------- Utilities ----------------
def parse_vcf(vcf_path):
    with open(vcf_path, "r") as fh:
        for ln in fh:
            if ln.startswith("#"): continue
            cols = ln.strip().split("\t")
            if len(cols) < 5: continue
            chrom, pos, ref, alt_str = cols[0], int(cols[1]), cols[3], cols[4]
            for alt in alt_str.split(","):
                yield {"chrom": chrom, "pos": pos, "ref": ref, "alt": alt}

def chunked(iterable, n):
    it = list(iterable)
    for i in range(0, len(it), n):
        yield it[i:i+n]

# ---------------- VEP annotation ----------------
def annotate_with_vep(variants, batch_size=50, pause=0.2):
    out = []
    url = ENSEMBL_GRCH37 + VEP_ENDPOINT
    for chunk in chunked(variants, batch_size):
        var_strs = [f"{v['chrom']}:{v['pos']}:{v['ref']}/{v['alt']}" for v in chunk]
        payload = {"variants": var_strs}
        try:
            r = requests.post(url, json=payload, headers=HEADERS_JSON, timeout=30)
            if r.status_code == 200:
                res = r.json()
                for v, rj in zip(chunk, res):
                    out.append({"input": v, "vep": rj})
            else:
                for v in chunk:
                    out.append({"input": v, "vep": None})
        except Exception:
            for v in chunk:
                out.append({"input": v, "vep": None})
        time.sleep(pause)
    return out

def pick_transcript_consequence(vep_entry):
    tcs = vep_entry.get("transcript_consequences") or []
    if not tcs: return None
    for tc in tcs:
        if tc.get("canonical") == 1 and tc.get("protein_id"): return tc
    for tc in tcs:
        if tc.get("protein_id"): return tc
    return tcs[0]

def fetch_protein_seq(ensp_id):
    url = ENSEMBL_GRCH37 + f"/sequence/id/{ensp_id}"
    params = {"type": "protein"}
    try:
        r = requests.get(url, headers=HEADERS_FASTA, params=params, timeout=20)
        if r.status_code == 200:
            lines = r.text.strip().splitlines()
            seq = "".join([l.strip() for l in lines if not l.startswith(">")])
            return seq
    except Exception:
        return None

def three_to_one(aa):
    table = {'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Gln':'Q','Glu':'E','Gly':'G',
             'His':'H','Ile':'I','Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P','Ser':'S',
             'Thr':'T','Trp':'W','Tyr':'Y','Val':'V','Sec':'U','Pyl':'O'}
    if len(aa)==1: return aa
    return table.get(aa, aa[0])

def parse_hgvsp(hgvsp):
    if not hgvsp: return None
    if ":p." in hgvsp: p = hgvsp.split(":p.")[1]
    elif "p." in hgvsp: p = hgvsp.split("p.")[1]
    else: p = hgvsp
    import re
    m = re.match(r"([A-Za-z]{1,3}|[A-Za-z])(\d+)([A-Za-z]{1,3}|[A-Za-z])", p)
    if not m: return None
    ref, pos, alt = m.group(1), int(m.group(2)), m.group(3)
    return {"ref": three_to_one(ref), "pos": pos, "alt": three_to_one(alt)}

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcf", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    variants = list(parse_vcf(args.vcf))
    if not variants:
        print("No variants found.", file=sys.stderr); return

    annotated = annotate_with_vep(variants)
    output_rows = []

    for item in annotated:
        vin = item["input"]
        vep = item.get("vep")
        row = {"chrom": vin["chrom"], "pos": vin["pos"], "ref": vin["ref"], "alt": vin["alt"],
               "gene": None, "transcript": None, "protein": None,
               "aa_pos": None, "wt_aa": None, "mut_aa": None,
               "delta_score": None, "p_wt": None, "p_mut": None}
        if not vep: 
            output_rows.append(row)
            continue
        tc = pick_transcript_consequence(vep)
        if not tc or not tc.get("protein_id"): 
            output_rows.append(row)
            continue

        hgvsp = tc.get("hgvsp")
        parsed = parse_hgvsp(hgvsp)
        protein_id = tc["protein_id"]
        transcript_id = tc.get("transcript_id")
        gene_symbol = tc.get("gene_symbol") or tc.get("gene_id")
        seq = fetch_protein_seq(protein_id)
        if not seq or not parsed: 
            output_rows.append(row)
            continue

        aa_pos = parsed["pos"] - 1  # 0-based for your compute_delta_score
        wt = parsed["ref"]
        mut = parsed["alt"]
        delta_score, p_wt, p_mut = compute_delta_score(seq, aa_pos, wt, mut)

        row.update({"gene": gene_symbol, "transcript": transcript_id, "protein": protein_id,
                    "aa_pos": aa_pos, "wt_aa": wt, "mut_aa": mut,
                    "delta_score": delta_score, "p_wt": p_wt, "p_mut": p_mut})
        output_rows.append(row)

    # write TSV
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["chrom","pos","ref","alt","gene","transcript","protein","aa_pos","wt_aa","mut_aa","delta_score","p_wt","p_mut"])
        for r in output_rows:
            w.writerow([r.get(c,"") for c in ["chrom","pos","ref","alt","gene","transcript","protein","aa_pos","wt_aa","mut_aa","delta_score","p_wt","p_mut"]])

    print(f"Done. Output written to {args.out}", file=sys.stderr)

if __name__ == "__main__":
    main()
