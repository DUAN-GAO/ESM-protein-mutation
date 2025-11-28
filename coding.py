#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import argparse
import time
import requests
import re
import urllib.parse
from esm import pretrained
import csv

# ---------------------------
# VCF CSQ 解析函数
# ---------------------------
def parse_vcf_csq_format(csq_field):
    """
    从 VCF CSQ 字段中识别蛋白突变信息，例如:
    NP_057208.3:p.Val9Leu -> V9L
    返回: pos(0-based), wt, mut, refseq_id, protein_change
    """
    fields = csq_field.split("|")
    if len(fields) < 11:
        return None

    protein_change = fields[10]  # NP_057208.3:p.Val9Leu
    refseq_id = fields[0]

    if not protein_change or ":p." not in protein_change:
        return None

    aa_change = protein_change.split(":p.")[1]
    aa_change = urllib.parse.unquote(aa_change)  # 处理 %3D 等 URL 编码

    # 跳过同义突变或缺失
    if aa_change in ("=", "-", "") or aa_change.endswith("="):
        return None

    # 正则匹配三字母或单字母形式
    m = re.match(r'^([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$', aa_change)
    if m:
        wt, pos_str, mut = m.groups()
        # 三字母 -> 单字母
        aa3to1 = {
            'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Glu':'E','Gln':'Q',
            'Gly':'G','His':'H','Ile':'I','Leu':'L','Lys':'K','Met':'M','Phe':'F',
            'Pro':'P','Ser':'S','Thr':'T','Trp':'W','Tyr':'Y','Val':'V','Sec':'U','Pyl':'O'
        }
        wt = aa3to1.get(wt, wt)
        mut = aa3to1.get(mut, mut)
    else:
        m2 = re.match(r'^([A-Z])(\d+)([A-Z])$', aa_change)
        if not m2:
            return None
        wt, pos_str, mut = m2.groups()

    pos = int(pos_str) - 1  # 0-based
    return pos, wt, mut, refseq_id, protein_change

# ---------------------------
# RefSeq -> UniProt
# ---------------------------
def refseq_to_uniprot(refseq_id):
    url = "https://rest.uniprot.org/idmapping/run"
    data = {"from": "RefSeq_Protein", "to": "UniProtKB", "ids": refseq_id}
    try:
        r = requests.post(url, data=data)
        if r.status_code != 200:
            print(f"[WARN] Mapping失败: {r.status_code}")
            return None
        job_id = r.json()["jobId"]
    except Exception as e:
        print(f"[WARN] Mapping请求异常: {e}")
        return None

    # 查询状态
    status_url = f"https://rest.uniprot.org/idmapping/status/{job_id}"
    while True:
        s = requests.get(status_url).json()
        if s.get("jobStatus") == "FINISHED":
            break
        time.sleep(1)

    # 获取结果
    result_url = f"https://rest.uniprot.org/idmapping/uniprotkb/results/{job_id}"
    r2 = requests.get(result_url)
    res = r2.json()
    if res.get("results"):
        return res["results"][0]["to"]
    return None

# ---------------------------
# 获取 UniProt 序列
# ---------------------------
def fetch_uniprot_sequence(uniprot_id):
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

# ---------------------------
# ESM 打分
# ---------------------------
def compute_delta_score(wildtype_sequence, mutation_position, wildtype_aa, mutant_aa):
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

# ---------------------------
# 读取 VCF
# ---------------------------
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

# ---------------------------
# 主程序
# ---------------------------
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

        # RefSeq -> UniProt
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
