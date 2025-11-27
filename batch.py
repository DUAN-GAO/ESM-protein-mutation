# ---------- batch.py ----------

import argparse
import gzip
import csv
from main import score_variant  # 直接调用 main.py 函数


def parse_vcf_line(line):
    fields = line.strip().split("\t")
    chrom = fields[0].replace("chr", "")
    pos = int(fields[1])
    rsid = fields[2].split("&")[0]
    ref = fields[3]
    alt = fields[4]
    return rsid, chrom, pos, ref, alt


def load_vcf(vcf_path):
    variants = []
    opener = gzip.open if vcf_path.endswith(".gz") else open
    with opener(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            try:
                variants.append(parse_vcf_line(line))
            except Exception as e:
                print(f"[WARN] 跳过行: {line.strip()} → {e}")
    return variants


def main(vcf_path, output_csv="results.csv"):
    variants = load_vcf(vcf_path)
    print(f"[INFO] 共读取 {len(variants)} 条变异记录")

    results = []

    for rsid, chrom, pos, ref, alt in variants:
        res = score_variant(chrom, pos, ref, alt, rsid)
        results.append(res)

    keys = sorted({k for r in results for k in r.keys()})

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    print(f"[DONE] 结果已保存：{output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量 VCF AlphaGenome 评分")
    parser.add_argument("--vcf", required=True, help="VCF 文件路径 (.vcf 或 .vcf.gz)")
    parser.add_argument("--out", default="results.csv", help="输出 CSV 路径")
    args = parser.parse_args()

    main(args.vcf, args.out)
