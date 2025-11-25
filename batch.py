import argparse
import csv
from main import score_single_variant


def parse_vcf_and_score(vcf_path, out_csv):
    rows = []

    with open(vcf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue

            fields = line.strip().split("\t")
            chrom, pos, _, ref, alt = fields[:5]

            info = fields[7]
            rsid = None

            # 在 INFO 字段中查找 rsID
            tokens = info.split(";")
            for t in tokens:
                if t.startswith("RSID="):
                    rsid = t.replace("RSID=", "")
                    break

            if rsid is None:
                continue

            print(f"[RUN] {rsid} {chrom}:{pos} {ref}>{alt}")

            try:
                delta = score_single_variant(rsid, chrom, pos, ref, alt)
                rows.append([rsid, chrom, pos, ref, alt, delta])
            except Exception as e:
                print(f"[WARN] {rsid} failed: {e}")
                continue

    # 保存结果
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rsid", "chrom", "pos", "ref", "alt", "delta"])
        writer.writerows(rows)

    print(f"[DONE] 结果已保存到 {out_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcf", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    parse_vcf_and_score(args.vcf, args.out)


if __name__ == "__main__":
    main()
