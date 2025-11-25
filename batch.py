import csv
import argparse
from main import score_variant  # 假设你把 main.py 的核心逻辑封装成函数 score_variant(chrom, pos, ref, alt, rsid)

def parse_vcf(vcf_path):
    variants = []
    with open(vcf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            chrom, pos, _, ref, alt, _, _, info = fields[:8]

            # 从 CSQ 提取 rsID
            rsid = None
            if "CSQ=" in info:
                csq_entries = info.split("CSQ=")[1].split(",")
                for e in csq_entries:
                    parts = e.split("|")
                    for p in parts:
                        if p.startswith("rs"):
                            rsid = p
                            break
                    if rsid:
                        break
            if rsid is None:
                continue

            variants.append((rsid, chrom, int(pos), ref, alt))
    return variants

def main(vcf_path, out_csv):
    variants = parse_vcf(vcf_path)
    results = []

    for rsid, chrom, pos, ref, alt in variants:
        print(f"[RUN] {rsid} {chrom}:{pos} {ref}>{alt}")
        try:
            delta = score_variant(chrom, pos, ref, alt, rsid)  # 直接调用 main.py 核心函数
            results.append([rsid, chrom, pos, ref, alt, delta])
        except Exception as e:
            print(f"[WARN] {rsid} failed: {e}")

    # 写 CSV
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rsid", "chrom", "pos", "ref", "alt", "delta"])
        writer.writerows(results)
    print(f"[DONE] 结果已保存到 {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcf", required=True)
    parser.add_argument("--out", default="results.csv")
    args = parser.parse_args()
    main(args.vcf, args.out)
