from alphagenome.data import genome
from alphagenome.models import dna_client, variant_scorers

API_KEY = "AIzaSyD5Kht8QzCPkHeJ456_Tf_eBWirtKhmaRU"

# ---------------- 核心函数 ----------------
def score_variant(dna_model, chrom, pos, ref, alt):
    """
    输入 dna_model 和变异位点信息，返回 tidy_scores DataFrame
    """
    print(f"[INFO] Scoring {chrom}:{pos} {ref}>{alt}")

    # 创建 Variant 对象
    variant = genome.Variant(
        chromosome=chrom,
        position=int(pos),
        reference_bases=ref,
        alternate_bases=alt,
    )

    # 最小窗口 16kb
    interval = variant.reference_interval.resize(16384)

    scorer = variant_scorers.CenterMaskScorer(
        width=501,
        aggregation_type=variant_scorers.AggregationType.DIFF_MEAN,
        requested_output=dna_client.OutputType.RNA_SEQ,
    )

    result = dna_model.score_variant(
        interval=interval,
        variant=variant,
        variant_scorers=[scorer],
        organism=dna_client.Organism.HOMO_SAPIENS,
    )

    # 使用 tidy_scores 生成最终 Δ-score
    tidy_df = variant_scorers.tidy_scores([result[0]], match_gene_strand=True)

    # 计算 delta_scalar 使用 raw_score
    delta_scalar = float(abs(tidy_df["raw_score"]).mean())
    print(f"[OK] 单一 Δ = {delta_scalar}")

    return tidy_df  # 返回 tidy_scores DataFrame

# ---------------- 命令行调用 ----------------
if __name__ == "__main__":
    import argparse, csv

    parser = argparse.ArgumentParser()
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--pos", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--alt", required=True)
    parser.add_argument("--out", default=None, help="可选输出 CSV 文件")
    args = parser.parse_args()

    dna_model = dna_client.create(API_KEY)
    tidy_df = score_variant(dna_model, args.chrom, args.pos, args.ref, args.alt)
    delta_scalar = float(abs(tidy_df["raw_score"]).mean())

    # 输出 TXT 或 CSV
    if args.out:
        with open(args.out, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([args.chrom, args.pos, args.ref, args.alt, delta_scalar])
    else:
        with open(f"{args.chrom}_{args.pos}.txt", "w") as f:
            f.write(f"{args.chrom},{args.pos},{args.ref},{args.alt},{delta_scalar}\n")
        print(f"[SAVE] {args.chrom}_{args.pos}.txt written.")
