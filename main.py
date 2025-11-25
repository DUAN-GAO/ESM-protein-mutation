import argparse
from alphagenome.data import genome
from alphagenome.models import dna_client, variant_scorers


API_KEY = "AIzaSyD5Kht8QzCPkHeJ456_Tf_eBWirtKhmaRU"


def score_single_variant(rsid, chrom, pos, ref, alt):
    print(f"[INFO] Running {rsid}: {chrom}:{pos} {ref}>{alt}")

    # 初始化模型
    dna_model = dna_client.create(API_KEY)

    # 创建 Variant 对象
    variant = genome.Variant(
        chromosome=chrom,
        position=int(pos),
        reference_bases=ref,
        alternate_bases=alt,
    )

    # 自动匹配 AlphaGenome 支持的最小窗口 16384bp
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

    delta = result[0].var
    print(f"[OK] {rsid} Δ = {delta}")

    return delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rsid", required=True)
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--pos", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--alt", required=True)
    args = parser.parse_args()

    delta = score_single_variant(
        rsid=args.rsid,
        chrom=args.chrom,
        pos=args.pos,
        ref=args.ref,
        alt=args.alt
    )

    # 输出 CSV 格式
    with open(f"{args.rsid}.txt", "w") as f:
        f.write(f"{args.rsid},{args.chrom},{args.pos},{args.ref},{args.alt},{delta}\n")

    print(f"[SAVE] {args.rsid}.txt written.")


if __name__ == "__main__":
    main()
