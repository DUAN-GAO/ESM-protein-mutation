import torch
import argparse
import time
import requests
from esm import pretrained
import urllib.parse
import csv

def parse_vcf_csq_format(csq_field):
    """
    从 VCF CSQ 字段中识别蛋白突变信息，例如:
    NP_001009931.1:p.Arg1442Gln → R1442Q
    如果未包含蛋白突变信息（同义突变等），返回 None
    """
    fields = csq_field.split("|")
    if len(fields) < 11:
        return None

    protein_change = fields[10]  # NP_001009931.1:p.Arg1442Gln
    refseq_id = fields[0]        # RefSeq 列

    if ":p." not in protein_change:
        return None

    aa_change = protein_change.split(":p.")[1]
    aa_change = urllib.parse.unquote(aa_change)  # 处理 %3D 等 URL 编码

    # 同义突变末尾是 "=" 或 "-"，跳过
    if aa_change.endswith("=") or aa_change == "-":
        return None

    wt = aa_change[0]
    mut = aa_change[-1]
    pos_digits = "".join([c for c in aa_change if c.isdigit()])
    if not pos_digits:
        return None
    pos = int(pos_digits) - 1  # 0-based index

    return pos, wt, mut, refseq_id, protein_change


def refseq_to_uniprot(refseq_id):
    """
    使用 UniProt mapping API 将 RefSeq ID 转为 UniProt accession
    """
    url = "https://rest.uniprot.org/idmapping/run"
    data = {"from": "RefSeq_Protein", "to": "UniProtKB", "ids": refseq_id}
    try:
        r = requests.post(url, data=data, timeout=10)
        if r.status_code != 200:
            print(f"[WARN] Mapping失败: {r.status_code}")
            return None
        job_id = r.json()["jobId"]
    except Exception as e:
        print(f"[WARN] Mapping请求异常: {e}")
        return None

    status_url = f"https://rest.uniprot.org/idmapping/status/{job_id}"
    while True:
        s = requests.get(status_url, timeout=10).json()
        if s.get("jobStatus") == "FINISHED":
            break
        time.sleep(1)

    result_url = f"https://rest.uniprot.org/idmapping/uniprotkb/results/{job_id}"
    r2 = requests.get(result_url, timeout=10)
    res = r2.json()
    if res.get("results"):
        return res["results"][0]["to"]
    return None


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
    使用 ESM 模型计算 Δscore
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
    with open(vcf_file) as f:
        for line in f:
            if not line.startswith("#"):
                yield line.strip()


def extract_vcf_info(vcf_line):
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
    args = parser.parse_args()

    print("[STEP 1] Reading VCF and parsing mutations...")
    results = []
    for line in extract_vcf_lines(args.vcf):
        info = extract_vcf_info(line)
        if not info:
            print(f"[INFO] 未检测到蛋白突变信息，可能为同义突变，跳过：{line}")
            continue

        pos, wt, mut, refseq_id, protein_change = info
        print(f"[INFO] Mutation parsed: {protein_change} ({wt}{pos+1}{mut})")

        # RefSeq → UniProt
        uniprot_id = refseq_to_uniprot(refseq_id)
        if not uniprot_id:
            print(f"[WARN] 无法映射 {refseq_id} 到 UniProt ID，跳过...")
            continue

        sequence = fetch_uniprot_sequence(uniprot_id)
        if not sequence:
            print(f"[WARN] 无法获取蛋白序列 {uniprot_id}，跳过...")
            continue

        print("[STEP 2] ESM Deep Mutation Scoring...")
        start = time.time()
        delta, p_wt, p_mut = compute_delta_score(sequence, pos, wt, mut)
        end = time.time()

        results.append({
            "Protein_ID": uniprot_id,
            "Mutation": f"{wt}{pos+1}{mut}",
            "P_WT": p_wt,
            "P_Mut": p_mut,
            "Delta": delta,
            "Runtime_s": end - start
        })
        print(f"[OK] Δscore={delta:.4f}, runtime={end-start:.2f}s\n")

    if not results:
        print("[INFO] 没有有效突变结果，CSV 未生成。")
    else:
        out_csv = "esm_results.csv"
        keys = results[0].keys()
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"[DONE] 结果已保存到 {out_csv}")
