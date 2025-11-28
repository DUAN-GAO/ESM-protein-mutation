import torch
import argparse
import time
import requests
import csv
from esm import pretrained


def parse_vcf_csq_format(csq_field):
    """
    从 VCF CSQ 字段中识别蛋白突变信息，例如:
    NP_001009931.1:p.Arg1442Gln → R1442Q
    如果未包含蛋白突变信息（同义突变等），返回 None
    """
    fields = csq_field.split("|")
    if len(fields) < 11:
        return None

    protein_change = fields[10]  # 例如: NP_001009931.1:p.Arg1442Gln
    uniprot_id = fields[0]       # 修正为真实 UniProt 列

    if ":p." not in protein_change:
        return None  # 同义突变或缺失信息

    aa_change = protein_change.split(":p.")[1]  # Arg1442Gln
    wt = aa_change[0]                           # 原氨基酸
    mut = aa_change[-1]                         # 突变氨基酸
    pos_digits = "".join([c for c in aa_change if c.isdigit()])
    if not pos_digits:
        return None
    pos = int(pos_digits) - 1  # 转 0-based index

    return pos, wt, mut, uniprot_id, protein_change


def fetch_uniprot_sequence(uniprot_id):
    """
    通过 UniProt REST API 获取蛋白序列
    """
    api_url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    print(f"[API] Fetching protein from UniProt: {api_url}")

    try:
        r = requests.get(api_url, timeout=10)
    except Exception as e:
        print(f"[WARN] 请求 UniProt 失败: {e}")
        return None

    if r.status_code != 200:
        print(f"[WARN] UniProt 访问失败: 状态码 {r.status_code}")
        return None

    seq = "".join([l.strip() for l in r.text.split("\n") if not l.startswith(">")])
    if len(seq) < 50:
        print(f"[WARN] 获取的蛋白长度异常 ({len(seq)} aa)")
        return None

    print(f"[OK] Protein length: {len(seq)} aa")
    return seq


def compute_delta_score(wildtype_sequence, mutation_position, wildtype_aa, mutant_aa):
    """
    计算 ESM Δscore
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
    mask_logits = logits[0, mutation_position+1, :]
    probabilities = torch.softmax(mask_logits, dim=0)
    wt_idx = alphabet.get_idx(wildtype_aa)
    mut_idx = alphabet.get_idx(mutant_aa)
    p_wild = probabilities[wt_idx].item()
    p_mutant = probabilities[mut_idx].item()
    epsilon = 1e-10
    delta_score = torch.log(torch.tensor(p_mutant + epsilon) / torch.tensor(p_wild + epsilon)).item()
    return delta_score, p_wild, p_mutant


def extract_vcf_info(vcf_line):
    """
    解析 VCF 格式，识别 CSQ 字段
    """
    columns = vcf_line.split("\t")
    if len(columns) < 8:
        return None

    info_field = columns[7]
    csq = None
    for sub in info_field.split(";"):
        if sub.startswith("CSQ="):
            csq = sub.replace("CSQ=", "")
            break

    if not csq:
        return None

    return parse_vcf_csq_format(csq)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESM protein mutation scoring from VCF")
    parser.add_argument("--vcf", type=str, required=True, help="VCF file path with annotation")
    parser.add_argument("--out", type=str, default="esm_results.csv", help="Output CSV file")
    args = parser.parse_args()

    results = []

    with open(args.vcf) as f:
        for line in f:
            if line.startswith("#"):
                continue  # 跳过头部
            line = line.strip()
            info = extract_vcf_info(line)
            if not info:
                print(f"[INFO] 未检测到蛋白突变信息，可能为同义突变，跳过：{line}")
                continue

            pos, wt, mut, uniprot_id, protein_change = info
            print(f"\n[INFO] Mutation parsed: {protein_change} (WT={wt} MUT={mut} POS={pos+1})")

            sequence = fetch_uniprot_sequence(uniprot_id)
            if not sequence:
                print(f"[WARN] 无法获取蛋白序列，跳过：{protein_change}")
                continue

            start = time.time()
            delta, p_wt, p_mut = compute_delta_score(sequence, pos, wt, mut)
            end = time.time()

            print(f"[OK] Δscore={delta:.4f}, Runtime={end-start:.2f} sec")
            results.append({
                "Protein_ID": uniprot_id,
                "Mutation": f"{wt}{pos+1}{mut}",
                "P_WT": p_wt,
                "P_Mut": p_mut,
                "Delta": delta,
                "Runtime_sec": round(end-start, 2)
            })

    # 输出 CSV
    if results:
        keys = results[0].keys()
        with open(args.out, "w", newline="") as out_csv:
            writer = csv.DictWriter(out_csv, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[INFO] 所有结果已保存到 {args.out}")
    else:
        print("\n[INFO] 没有有效突变结果，CSV 未生成。")
