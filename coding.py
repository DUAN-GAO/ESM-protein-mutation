import torch
import argparse
import time
import requests
from esm import pretrained


def parse_vcf_csq_format(csq_field):
    """
    从 VCF CSQ 字段中识别蛋白突变信息，例如:
    NP_001009931.1:p.Arg1442Gln → R1442Q
    """
    fields = csq_field.split("|")
    protein_change = fields[10]  # 如: NP_001009931.1:p.Arg1442Gln
    uniprot_id = fields[8]       # Transcript 例如 NM_001009931.3

    if ":p." not in protein_change:
        raise ValueError("No protein change format found in VCF CSQ")

    aa_change = protein_change.split(":p.")[1]  # Arg1442Gln
    wt = aa_change[0]                          # R
    mut = aa_change[-1]                        # Q
    pos = int("".join([c for c in aa_change if c.isdigit()])) - 1  # 转 0-based

    return pos, wt, mut, uniprot_id, protein_change


def fetch_uniprot_sequence(uniprot_id):
    """
    通过 UniProt 获取蛋白序列
    """
    api_url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    print(f"[API] Fetching protein from UniProt: {api_url}")

    r = requests.get(api_url)
    if r.status_code != 200:
        raise ValueError(f"Failed to fetch UniProt FASTA for {uniprot_id}")

    seq = "".join([l.strip() for l in r.text.split("\n") if not l.startswith(">")])

    print(f"[OK] Protein length: {len(seq)} aa")
    return seq


def compute_delta_score(wildtype_sequence, mutation_position, wildtype_aa, mutant_aa):
    """
    运行 ESM-1b 模型用于打分
    """
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

    wt_index = alphabet.get_idx(wildtype_aa)
    mut_index = alphabet.get_idx(mutant_aa)

    p_wild = probabilities[wt_index].item()
    p_mut = probabilities[mut_index].item()

    epsilon = 1e-10
    delta_score = torch.log(torch.tensor(p_mut + epsilon) / torch.tensor(p_wild + epsilon)).item()

    return delta_score, p_wild, p_mut


def extract_vcf_info(vcf_line):
    """
    解析 VCF 格式，自动识别 CSQ 字段
    """
    columns = vcf_line.split("\t")
    info_field = columns[7]

    csq = None
    for sub in info_field.split(";"):
        if sub.startswith("CSQ="):
            csq = sub.replace("CSQ=", "")
            break

    if not csq:
        raise ValueError("No CSQ annotation found in VCF INFO field")

    return parse_vcf_csq_format(csq)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ESM score directly from VCF mutation line")
    parser.add_argument("--vcf", type=str, required=True, help="Single-line VCF record containing CSQ annotation")

    args = parser.parse_args()

    print("[STEP 1] Parsing VCF...")
    pos, wt, mut, uniprot_id, protein_change = extract_vcf_info(args.vcf)

    print(f"[INFO] Protein mutation parsed: {protein_change}")
    print(f"[INFO] WT={wt} MUT={mut} POS={pos+1}")

    print("\n[STEP 2] Fetching protein sequence...")
    sequence = fetch_uniprot_sequence(uniprot_id)

    print("\n[STEP 3] Running ESM inference...")
    start = time.time()
    delta, p_wt, p_mut = compute_delta_score(sequence, pos, wt, mut)
    end = time.time()

    print("\n===== ESM PREDICTION RESULT =====")
    print(f"Protein: {uniprot_id}")
    print(f"Mutation: {wt}{pos+1}{mut}")
    print(f"P(wild-type)  = {p_wt:.6f}")
    print(f"P(mutant)     = {p_mut:.6f}")
    print(f"Δscore (log(mut/wt)) = {delta:.4f}")
    print(f"Runtime: {end - start:.2f} sec")
