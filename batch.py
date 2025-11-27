import argparse
import gzip
import csv
from main import score_single_variant  # 你的 main.py 文件中函数

def parse_vcf_line(line):
    """
    从 VCF 行提取 chrom, pos, rsid, ref, alt
    """
    fields = line.strip().split("\t")
    chrom = fields[0]
    pos = int(fields[1])
    rsid = fields[2].split("&")[0]  # 清理附加 ID
    ref = fields[3]
    alt = fields[4]
    return rsid, chrom, pos, ref, alt

def load_vcf(vcf_path):
    """
    读取 VCF 文件，返回 [(rsid, chrom, pos, ref, alt), ...]
    """
    variants = []
    opener = gzip.open if vcf_path.endswith(".gz") else open
    with opener(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            try:
                variant = parse_vcf_line(line)
                variants.append(variant)
            except Exception as e:
                print(f"[WARN] 解析行失败: {line.strip()} -> {e}")
    return variants

def main(vcf_path, output_csv="results.csv"):
    variants = load_vcf(vcf_path)
    print(f"[INFO] 共找到 {len(variants)} 个变异")

    results = []
    for rsid, chrom, pos, ref, alt in variants:
        print(f"[RUN] 处理 {rsid} ...")
        try:
            delta = score_single_variant(rsid, chrom, pos, ref, alt)
            results.append({"rsid": rsid, "chrom": chrom, "pos": pos, "ref": ref, "alt": alt, "delta_score": delta})
        except Exception as e:
            print(f"[WARN] {rsid} 处理失败: {e}")

    # 写入 CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rsid", "chrom", "pos", "ref", "alt", "delta_score"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"[DONE] 结果已保存到 {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量处理 VCF 并调用 AlphaGenome scoring")
    parser.add_argument("--vcf", required=True, help="输入 VCF 文件路径 (.vcf 或 .vcf.gz)")
    parser.add_argument("--out", default="results.csv", help="输出 CSV 文件")
    args = parser.parse_args()

    main(args.vcf, args.out)
