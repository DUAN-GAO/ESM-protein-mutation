import argparse
import gzip
import csv
from main import score_variant, API_KEY
from alphagenome.models import dna_client

def parse_vcf_line(line):
    """从 VCF 行提取 chrom, pos, rsid, ref, alt"""
    fields = line.strip().split("\t")
    chrom = fields[0]
    pos = int(fields[1])
    rsid = fields[2].split("&")[0] if fields[2] != "." else f"{chrom}:{pos}"
    ref = fields[3]
    alt = fields[4]
    return rsid, chrom, pos, ref, alt

def load_vcf(vcf_path):
    """读取 VCF 文件，返回 SNV 列表"""
    variants = []
    opener = gzip.open if vcf_path.endswith(".gz") else open
    with opener(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            try:
                rsid, chrom, pos, ref, alt = parse_vcf_line(line)
                # 只保留 SNV
                if len(ref) == 1 and len(alt) == 1:
                    variants.append((rsid, chrom, pos, ref, alt))
                else:
                    print(f"[SKIP] 非 SNV 变异 {chrom}:{pos} {ref}>{alt}")
            except Exception as e:
                print(f"[WARN] 解析行失败: {line.strip()} -> {e}")
    return variants

def main(vcf_path, output_csv="results.csv"):
    variants = load_vcf(vcf_path)
    print(f"[INFO] 共找到 {len(variants)} 个 SNV 变异")

    # ---------------- 创建一次 dna_model ----------------
    dna_model = dna_client.create(API_KEY)

    results = []
    for rsid, chrom, pos, ref, alt in variants:
        print(f"[RUN] 处理 {rsid} ...")
        try:
            delta_df = score_variant(dna_model, rsid, chrom, pos, ref, alt)
            if delta_df is None:
                continue
            delta_scalar = float(delta_df["nonzero_mean"].mean())
            results.append({
                "rsid": rsid,
                "chrom": chrom,
                "pos": pos,
                "ref": ref,
                "alt": alt,
                "delta_score": delta_scalar
            })
            print(f"[OK] {rsid} 单一 Δ = {delta_scalar}")
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
