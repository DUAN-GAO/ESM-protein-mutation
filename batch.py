import argparse
import gzip
import csv
from main import score_variant, dna_client, API_KEY  # 使用上面修改后的 score_variant

def parse_vcf_line(line):
    fields = line.strip().split("\t")
    chrom = fields[0]
    pos = int(fields[1])
    ref = fields[3]
    alt = fields[4]
    return chrom, pos, ref, alt

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
                print(f"[WARN] 解析行失败: {line.strip()} -> {e}")
    return variants

def main(vcf_path, output_csv="results.csv"):
    variants = load_vcf(vcf_path)
    print(f"[INFO] 共找到 {len(variants)} 个变异")

    # ---------------- 只创建一次 dna_model ----------------
    dna_model = dna_client.create(API_KEY)

    results = []
    for chrom, pos, ref, alt in variants:
        print(f"[RUN] 处理 {chrom}:{pos} ...")
        try:
            tidy_df = score_variant(dna_model, chrom, pos, ref, alt)
            # 使用 raw_score 替换 nonzero_mean
            delta_scalar = float(abs(tidy_df["raw_score"]).max())
            results.append({
                "chrom": chrom,
                "pos": pos,
                "ref": ref,
                "alt": alt,
                "delta_score": delta_scalar
            })
        except Exception as e:
            print(f"[WARN] {chrom}:{pos} 处理失败: {e}")

    # 写入 CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["chrom", "pos", "ref", "alt", "delta_score"])
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
