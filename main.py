from alphagenome.data import genome
from alphagenome.models import dna_client, variant_scorers

API_KEY = "AIzaSyD5Kht8QzCPkHeJ456_Tf_eBWirtKhmaRU"

# ---------------- 核心函数 ----------------
def score_variant(rsid, chrom, pos, ref, alt):
    """
    输入 rsID 和变异位点信息，返回 delta_score (唯一数值)
    """
    print(f"[INFO] Scoring {rsid}: {chrom}:{pos} {ref}>{alt}")

    # 初始化模型
    dna_model = dna_client.create(API_KEY)

    # 创建 Variant 对象
    variant = genome.Variant(
        chromosome=chrom,
        position=int(pos),
        reference_bases=ref,
        alternate_bases=alt,
    )

    # AlphaGenome 规定窗口必须 >= 16384bp
    interval = variant.reference_interval.resize(16384)

    scorer = variant_scorers.CenterMaskScorer(
        width=None,
        aggregation_type=variant_scorers.AggregationType.DIFF_SUM_LOG2,
        requested_output=dna_client.OutputType.RNA_SEQ,
    )

    result = dna_model.score_variant(
        interval=interval,
        variant=variant,
        variant_scorers=[scorer],
        organism=dna_client.Organism.HOMO_SAPIENS,
    )

    # ---- ⬇⬇⬇ 修改部分：矩阵 → 单一评分 ----
    var_df = result[0].var  # DataFrame

    # 计算唯一评分：abs(delta).mean()
    delta_scalar = float(abs(var_df["delta"]).mean())
    # ---- ⬆⬆⬆ 修改结束 ----

    print(f"[OK] {rsid} 单一 Δ = {delta_scalar}")
    return delta_scalar


# ---------------- 命令行调用 ----------------
if __name__ == "__main__":
    import argparse, csv

    parser = argparse.ArgumentParser()
    parser.add_argument("--rsid", required=True)
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--pos", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--alt", required=True)
    parser.add_argument("--out", default=None, help="可选输出 CSV 文件")
    args = parser.parse_args()

    delta = score_variant(args.rsid, args.chrom, args.pos, args.ref, args.alt)

    # 输出 TXT 或 CSV
    if args.out:
        with open(args.out, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([args.rsid, args.chrom, args.pos, args.ref, args.alt, delta])
    else:
        with open(f"{args.rsid}.txt", "w") as f:
            f.write(f"{args.rsid},{args.chrom},{args.pos},{args.ref},{args.alt},{delta}\n")
        print(f"[SAVE] {args.rsid}.txt written.")
