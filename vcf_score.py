#!/usr/bin/env python3
import argparse, csv, time, requests, sys
from collections import defaultdict
import torch
from esm import pretrained

# ---------------- User's compute_delta_score (UNCHANGED) ----------------
def compute_delta_score(wildtype_sequence, mutation_position, wildtype_aa, mutant_aa):
    print(f"[DEBUG] Running compute_delta_score: aa_pos={mutation_position}, wt={wildtype_aa}, mut={mutant_aa}, seq_len={len(wildtype_sequence)}")

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

    print(f"[DEBUG] compute_delta_score DONE: delta={delta_score}, p_wt={p_wild}, p_mut={p_mutant}")

    return delta_score, p_wild, p_mutant

# ---------------- Ensembl GRCh37 REST ----------------
ENSEMBL_GRCH37 = "http://grch37.rest.ensembl.org"
VEP_ENDPOINT = "/vep/human/region"
HEADERS_JSON = {"Content-Type": "application/json", "Accept": "application/json"}
HEADERS_FASTA = {"Accept": "text/x-fasta"}

# ---------------- Utilities ----------------
def parse_vcf(vcf_path):
    print(f"[DEBUG] Parsing VCF: {vcf_path}")
    with open(vcf_path, "r") as fh:
        for ln in fh:
            if ln.startswith("#"): continue
            cols = ln.strip().split("\t")
            if len(cols) < 5: continue
            chrom, pos, ref, alt_str = cols[0], int(cols[1]), cols[3], cols[4]
            for alt in alt_str.split(","):
                print(f"[DEBUG] Parsed variant: {chrom}:{pos} {ref}>{alt}")
                yield {"chrom": chrom, "pos": pos, "ref": ref, "alt": alt}

def chunked(iterable, n):
    it = list(iterable)
    for i in range(0, len(it), n):
        yield it[i:i+n]

def _normalize_chrom(chrom):
    """Remove leading 'chr' or 'CHR' if present, as Ensembl expects '1' not 'chr1'."""
    if chrom.lower().startswith("chr"):
        return chrom[3:]
    return chrom

# ---------------- VEP annotation (fixed variant string format) ----------------
def annotate_with_vep(variants, batch_size=50, pause=0.2):
    out = []
    url = ENSEMBL_GRCH37 + VEP_ENDPOINT

    for chunk in chunked(variants, batch_size):
        # NOTE: use "chrom pos ref/alt" format (space-separated) and normalize chrom by removing "chr"
        var_strs = []
        for v in chunk:
            chrom_clean = _normalize_chrom(v['chrom'])
            var_strs.append(f"{chrom_clean} {v['pos']} {v['ref']}/{v['alt']}")
        print(f"[DEBUG] Sending VEP request (GRCh37) with variants: {var_strs}")

        payload = {"variants": var_strs}

        try:
            r = requests.post(url, json=payload, headers=HEADERS_JSON, timeout=30)
            print(f"[DEBUG] VEP status: {r.status_code}")
            if r.status_code == 200:
                res = r.json()
                # expected: list aligned to var_strs
                for v, rj in zip(chunk, res):
                    out.append({"input": v, "vep": rj})
            else:
                # Print server response body to help debug (VEP returns JSON explaining the 400)
                text = ""
                try:
                    text = r.text
                except Exception:
                    text = "<no response body>"
                print(f"[DEBUG] VEP non-200 response text: {text}", file=sys.stderr)
                for v in chunk:
                    out.append({"input": v, "vep": None, "vep_error": f"status_{r.status_code}", "vep_text": text})
        except Exception as e:
            print(f"[DEBUG] VEP request FAILED: {e}", file=sys.stderr)
            for v in chunk:
                out.append({"input": v, "vep": None, "vep_error": str(e)})
        time.sleep(pause)

    return out

def pick_transcript_consequence(vep_entry):
    tcs = vep_entry.get("transcript_consequences") or []
    print(f"[DEBUG] transcript_consequences count: {len(tcs)}")
    if not tcs: return None
    for tc in tcs:
        if tc.get("canonical") == 1 and tc.get("protein_id"):
            print("[DEBUG] canonical transcript selected")
            return tc
    for tc in tcs:
        if tc.get("protein_id"):
            print("[DEBUG] non-canonical transcript with protein selected")
            return tc
    return tcs[0] if tcs else None

def fetch_protein_seq(ensp_id):
    print(f"[DEBUG] Fetching protein seq: {ensp_id}")
    url = ENSEMBL_GRCH37 + f"/sequence/id/{ensp_id}"
    params = {"type": "protein"}
    try:
        r = requests.get(url, headers=HEADERS_FASTA, params=params, timeout=20)
        print(f"[DEBUG] FASTA status: {r.status_code}")
        if r.status_code == 200:
            lines = r.text.strip().splitlines()
            seq = "".join([l.strip() for l in lines if not l.startswith(">")])
            print(f"[DEBUG] Protein length: {len(seq)}")
            return seq
    except Exception as e:
        print(f"[DEBUG] FASTA fetch failed: {e}", file=sys.stderr)
        return None

def three_to_one(aa):
    table = {'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Gln':'Q','Glu':'E','Gly':'G',
             'His':'H','Ile':'I','Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P','Ser':'S',
             'Thr':'T','Trp':'W','Tyr':'Y','Val':'V','Sec':'U','Pyl':'O'}
    if len(aa)==1: return aa
    return table.get(aa, aa[0])

def parse_hgvsp(hgvsp):
    print(f"[DEBUG] Parsing hgvsp: {hgvsp}")
    if not hgvsp: return None
    if ":p." in hgvsp: p = hgvsp.split(":p.")[1]
    elif "p." in hgvsp: p = hgvsp.split("p.")[1]
    else: p = hgvsp

    import re
    m = re.match(r"([A-Za-z]{1,3}|[A-Za-z])(\d+)([A-Za-z]{1,3}|[A-Za-z])", p)
    if not m:
        print("[DEBUG] Could not parse HGVS p.", file=sys.stderr)
        return None

    ref, pos, alt = m.group(1), int(m.group(2)), m.group(3)
    print(f"[DEBUG] Parsed aa change: {ref}{pos}{alt}")
    return {"ref": three_to_one(ref), "pos": pos, "alt": three_to_one(alt)}

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcf", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    print("[DEBUG] Starting...")

    variants = list(parse_vcf(args.vcf))
    print(f"[DEBUG] Total variants parsed: {len(variants)}")

    annotated = annotate_with_vep(variants)
    output_rows = []

    for item in annotated:
        vin = item["input"]
        vep = item.get("vep")
        vep_error = item.get("vep_error")
        vep_text = item.get("vep_text")

        print(f"\n[DEBUG] Processing variant: {vin}")

        row = {"chrom": vin["chrom"], "pos": vin["pos"], "ref": vin["ref"], "alt": vin["alt"],
               "gene": None, "transcript": None, "protein": None,
               "aa_pos": None, "wt_aa": None, "mut_aa": None,
               "delta_score": None, "p_wt": None, "p_mut": None, "vep_error": vep_error or ""}

        if not vep:
            print("[DEBUG] No VEP annotation (see vep_error)", file=sys.stderr)
            if vep_text:
                print(f"[DEBUG] VEP text: {vep_text}", file=sys.stderr)
            output_rows.append(row)
            continue

        tc = pick_transcript_consequence(vep)
        print(f"[DEBUG] Picked transcript: {tc}")

        if not tc or not tc.get("protein_id"):
            print("[DEBUG] Missing protein_id")
            output_rows.append(row)
            continue

        hgvsp = tc.get("hgvsp")
        parsed = parse_hgvsp(hgvsp)
        protein_id = tc["protein_id"]
        transcript_id = tc.get("transcript_id")
        gene_symbol = tc.get("gene_symbol") or tc.get("gene_id")

        seq = fetch_protein_seq(protein_id)

        if not seq or not parsed:
            print("[DEBUG] Sequence fetch failed OR HGVS parse failed")
            output_rows.append(row)
            continue

        aa_pos = parsed["pos"] - 1
        wt = parsed["ref"]
        mut = parsed["alt"]

        print(f"[DEBUG] Calling compute_delta_score: pos={aa_pos}, wt={wt}, mut={mut}")

        try:
            delta_score, p_wt, p_mut = compute_delta_score(seq, aa_pos, wt, mut)
        except Exception as e:
            print(f"[DEBUG] compute_delta_score raised exception: {e}", file=sys.stderr)
            row["vep_error"] = f"compute_error:{e}"
            output_rows.append(row)
            continue

        row.update({"gene": gene_symbol, "transcript": transcript_id, "protein": protein_id,
                    "aa_pos": aa_pos, "wt_aa": wt, "mut_aa": mut,
                    "delta_score": delta_score, "p_wt": p_wt, "p_mut": p_mut})

        output_rows.append(row)

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["chrom","pos","ref","alt","gene","transcript","protein","aa_pos","wt_aa","mut_aa","delta_score","p_wt","p_mut","vep_error"])
        for r in output_rows:
            w.writerow([r.get(c,"") for c in ["chrom","pos","ref","alt","gene","transcript","protein","aa_pos","wt_aa","mut_aa","delta_score","p_wt","p_mut","vep_error"]])

    print(f"[DEBUG] Done. Output written to {args.out}")

if __name__ == "__main__":
    main()
