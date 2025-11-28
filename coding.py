import torch
import argparse
import time
import requests
import csv
from esm import pretrained


def parse_vcf_csq_format(csq_field):
    """
    修正 CSQ 解析，支持 VEP 格式：
    CSQ 列通常是用 | 分隔的字段，蛋白质变化在 12 列
    返回 pos, wt, mut, uniprot_id, protein_change
    """
    fields = csq_field.split("|")
    if len(fields) < 12:
        return None

    protein_change = fields[11]  # NP_057208.3:p.Val9Leu
    if not protein_change or ":p." not in protein_change:
        return None  # 同义突变或缺失信息

    uniprot_id = protein_change.split(":")[0]

    aa_change = protein_change.split(":p.")[1]  # Val9Leu
    wt = aa_change[0]
    mut = aa_change[-1]
    pos_digits = "".join([c for c in aa_change if c.isdigit()])
    if not pos_digits:
        return None
    pos = int(pos_digits) - 1

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

    return seq


def compute_delta_score(wildtype_sequence, mutation_position, wildtype_aa, mutant_aa):
    """
    计算 Δscore
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
    wt_idx = alphabet.get_idx(wildtype_aa)
    mut_idx = alphabet.get_idx(mutant_aa)
    p_wild = probabilities[wt_idx].item()
    p_mutant = probabilities[mut_idx].item()
    epsilon = 1e-10
    delta_score = torch.log(torch.tensor(p_mutant + epsilon) / torch.tensor(p_wild + epsilon)).item()
    return delta_score, p_wild, p_mutant


def extract_vcf_lines(vcf_file):
    """
    读取 VCF 文件中所有非注释行
    """
    with open(vcf_file) as f:
        for line in f:
            if not line.startswith("#"):
                yield line.strip()


def extract_csq_info(info_field):
    """
    解析 VCF INFO 字段，获取 CSQ 注释
    """
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
    parser.add_argument("--outcsv", type=str, default="esm_results.csv", help="Output CSV file")
    args = parser.parse_args()

    results = []

    print("[STEP 1] Reading VCF and parsing mutations...")
    for line in extract_vcf_lines(args.vcf):
        columns = line.split("\t")
        if len(columns) < 8:
            continue
        info_field = columns[7]
        csq_info = extract_csq_info(info_field)
        if not csq_info:
            print(f"[INFO] 未检测到蛋白突变信息，可能为同义突变，跳过：{line[:50]}...")
            continue

        pos, wt, mut, uniprot_id, protein_change = csq_info
        print(f"[INFO] Mutation parsed: {protein_change} ({wt}{pos+1}{mut})")

        sequence = fetch_uniprot_sequence(uniprot_id)
        if not sequence:
            print(f"[WARN] 无法获取蛋白序列 {uniprot_id}，跳过...")
            continue

        print(f"[STEP 2] ESM scoring for {protein_change}...")
        start = time.time()
        delta, p_wt, p_mut = compute_delta_score(sequence, pos, wt, mut)
        end = time.time()

        results.append({
            "Protein_ID": uniprot_id,
            "Mutation": f"{wt}{pos+1}{mut}",
            "P_Wild": p_wt,
            "P_Mutant": p_mut,
            "Delta_score": delta,
            "Runtime_sec": round(end - start, 2)
        })

    if results:
        with open(args.outcsv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[OK] 计算完成，结果已保存到 {args.outcsv}")
    else:
        print("\n[INFO] 没有有效突变结果，CSV 未生成。")
