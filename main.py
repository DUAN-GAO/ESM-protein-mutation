from alphagenome.data import genome
from alphagenome.models import dna_client, variant_scorers

API_KEY = "AIzaSyD5Kht8QzCPkHeJ456_Tf_eBWirtKhmaRU"

# ---------------- 核心函数 ----------------
def score_variant(dna_model, rsid, chrom, pos, ref, alt):
    """
    输入 dna_model 和 rsID + 变异位点信息，返回 DataFrame (包含 nonzero_mean)
    """
    print(f"[INFO] Scoring {rsid}: {chrom}:{pos} {ref}>{alt}")

    # 创建 Variant 对象
    variant = genome.Variant(
        chromosome=chrom,
        position=int(pos),
        reference_bases=ref,
        alternate_bases=alt,
    )

    # AlphaGenome 最小窗口 16384bp
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

    var_df = result[0].var  # DataFrame
    delta_scalar = float(abs(var_df["nonzero_mean"]).mean())

    print(f"[OK] {rsid} 单一 Δ = {delta_scalar}")
    return var_df  # 返回完整 DataFrame，batch.py 中取 nonzero_mean.mean()

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

    dna_model = dna_client.create(API_KEY)
    var_df = score_variant(dna_model, args.rsid, args.chrom, args.pos, args.ref, args.alt)
    delta_scalar = float(abs(var_df["nonzero_mean"]).mean())

    # 输出 TXT 或 CSV
    if args.out:
        with open(args.out, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([args.rsid, args.chrom, args.pos, args.ref, args.alt, delta_scalar])
    else:
        with open(f"{args.rsid}.txt", "w") as f:
            f.write(f"{args.rsid},{args.chrom},{args.pos},{args.ref},{args.alt},{delta_scalar}\n")
        print(f"[SAVE] {args.rsid}.txt written.")
