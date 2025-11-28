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

    # 安全检查
    if len(fields) < 11:
        raise ValueError(f"CSQ字段格式异常: {csq_field}")

    protein_change = fields[10]  # 例如: NP_001009931.1:p.Arg1442Gln
    uniprot_id = fields[0]  # 修正为真实 UniProt 列

    if ":p." not in protein_change:
        raise ValueError("未从 CSQ 提取到蛋白突变信息 (缺少 :p.)")

    aa_change = protein_change.split(":p.")[1]  # Arg1442Gln
    wt = aa_change[0]                           # 原氨基酸
    mut = aa_change[-1]                         # 突变氨基酸

    pos = int("".join([c for c in aa_change if c.isdigit()])) - 1  # 转 0-based index

    return pos, wt, mut, uniprot_id, protein_change


def fetch_uniprot_sequence(uniprot_id):
    """
    通过 UniProt REST API 获取蛋白序列
    """

    api_url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    print(f"[API] Fetching protein from UniProt: {api_url}")

    r = requests.get(api_url)

    if r.status_code != 200:
        raise ValueError(f"[ERROR] UniProt 访问失败: {uniprot_id}")

    seq = "".join([l.strip() for l in r.text.split("\n") if not l.startswith(">")])

    if len(seq) < 50:
        raise ValueError("[ERROR] 获取的蛋白长度异常，请检查ID是否正确")

    print(f"[OK] Protein length: {len(seq)} aa")
    return seq


def compute_delta_score(wildtype_sequence, mutation_position, wildtype_aa, mutant_aa):
    """
    使用 ESM 计算突变影响打分
    """

    print("[MODEL] Loading ESM-1b model...")
    model, alphabet = pretrained.esm1b_t33_650M_UR50S()
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    seq = list(wildtype_sequence)

    # 用 MASK 替换突变位点
    seq[mutation_position] = alphabet.get_tok(alphabet.mask_idx)
    masked_sequence_str = "".join(seq)

    data = [("protein", masked_sequence_str)]
    _, _, batch_tokens = batch_converter(data)

    print("[MODEL] Running inference...")
    with torch.no_grad():
        outputs = model(batch_tokens, repr_layers=[])
        logits = outputs["logits"]

    mask_logits = logits[0, mutation_position + 1, :]
    probabilities = torch.softmax(mask_logits, dim=0)

    wt_index = alphabet.get_idx(wildtype_aa)
    mut_index = alphabet.get_idx(mutant_aa)

    p_wild = probabilities[wt_index].item()
    p_mut = probabilities[mut_index].item()

    # log( mutant / WT )
    delta_score = torch.log(torch.tensor(p_mut + 1e-10) / torch.tensor(p_wild + 1e-10)).item()

    return delta_score, p_wild, p_mut


def extract_vcf_line(vcf_file):
    """
    读取 VCF 文件中的第一条变异记录
    """
    
    with open(vcf_file) as f:
        for line in f:
            if not line.startswith("#"):
                return line.strip()

    raise ValueError("VCF中未找到有效突变条目")


def extract_vcf_info(vcf_line):
    """
    解析 VCF 格式，识别 CSQ 字段
    """
    columns = vcf_line.split("\t")
    if len(columns) < 8:
        raise ValueError("VCF 格式错误: INFO 字段缺失")

    info_field = columns[7]

    csq = None
    for sub in info_field.split(";"):
        if sub.startswith("CSQ="):
            csq = sub.replace("CSQ=", "")
            break

    if not csq:
        raise ValueError("未检测到 CSQ 注释字段，请先使用 VEP 注释")

    return parse_vcf_csq_format(csq)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESM protein mutation scoring from VCF")
    parser.add_argument("--vcf", type=str, required=True, help="VCF file path with annotation")

    args = parser.parse_args()

    print("[STEP 1] Reading VCF file...")
    line = extract_vcf_line(args.vcf)

    print("[STEP 2] Parsing CSQ...")
    pos, wt, mut, uniprot_id, protein_change = extract_vcf_info(line)

    print(f"[INFO] Mutation parsed: {protein_change}")
    print(f"[INFO] WT={wt} MUT={mut} POS={pos + 1}")

    print("\n[STEP 3] Fetching UniProt sequence...")
    sequence = fetch_uniprot_sequence(uniprot_id)

    print("\n[STEP 4] ESM Deep Mutation Scoring...")
    start = time.time()
    delta, p_wt, p_mut = compute_delta_score(sequence, pos, wt, mut)
    end = time.time()

    print("\n===== ESM PREDICTION RESULT =====")
    print(f"Protein ID: {uniprot_id}")
    print(f"Mutation: {wt}{pos + 1}{mut}")
    print(f"P(Wild-type)     = {p_wt:.6f}")
    print(f"P(Mutant)        = {p_mut:.6f}")
    print(f"Δscore (log ratio) = {delta:.4f}")
    print(f"Runtime: {end - start:.2f} sec\n")
