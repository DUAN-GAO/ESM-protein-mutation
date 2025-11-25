#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import subprocess
import gzip
import csv

def extract_rsids_from_vcf(vcf_path):
    """从 VCF 文件中提取 rsID"""
    opener = gzip.open if vcf_path.endswith(".gz") else open
    rsids = []
    with opener(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            info = fields[7] if len(fields) > 7 else ""
            if "CSQ=" in info:
                csq_entries = info.split("CSQ=")[1].split(",")
                for entry in csq_entries:
                    cols = entry.split("|")
                    for col in cols:
                        col = col.strip()
                        if col.startswith("rs") and col[2:].isdigit():
                            rsids.append(col)
    return list(set(rsids))  # 去重

def main(vcf_path, output_csv):
    rsids = extract_rsids_from_vcf(vcf_path)
    print(f"[INFO] 共找到 {len(rsids)} 个 rsID")

    results = []
    for rsid in rsids:
        print(f"[RUN] 处理 {rsid} ...")
        try:
            # 调用你的 main.py
            result = subprocess.run(
                ["python", "main.py", "--rsid", rsid],
                capture_output=True, text=True, check=True
            )
            # 假设 main.py 输出 chrom, pos, ref, alt, delta_score 逗号分隔
            output_line = result.stdout.strip()
            if output_line:
                chrom, pos, ref, alt, delta = output_line.split(",")
                results.append({
                    "rsid": rsid,
                    "chrom": chrom,
                    "pos": pos,
                    "ref": ref,
                    "alt": alt,
                    "delta_score": delta
                })
        except subprocess.CalledProcessError as e:
            print(f"[WARN] {rsid} 处理失败: {e.stderr.strip()}")

    # 保存 CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rsid", "chrom", "pos", "ref", "alt", "delta_score"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"[FINISHED] 结果已保存到 {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcf", required=True, help="输入 VCF 文件")
    parser.add_argument("--out", default="results.csv", help="输出 CSV 文件")
    args = parser.parse_args()
    main(args.vcf, args.out)
