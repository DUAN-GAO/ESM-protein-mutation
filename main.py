from alphagenome.data import genome
from alphagenome.models import dna_client, variant_scorers

API_KEY = "AIzaSyD5Kht8QzCPkHeJ456_Tf_eBWirtKhmaRU"

def score_variant(dna_model, rsid, chrom, pos, ref, alt):
    """
    输入 rsID 和 SNV 变异位点信息，返回 delta DataFrame
    """
    print(f"[INFO] Scoring {rsid}: {chrom}:{pos} {ref}>{alt}")

    # 只处理 SNV
    if len(ref) != 1 or len(alt) != 1:
        print(f"[SKIP] 非 SNV 变异 {chrom}:{pos} {ref}>{alt}")
        return None

    # 创建 Variant 对象
    variant = genome.Variant(
        chromosome=chrom,
        position=int(pos),
        reference_bases=ref,
        alternate_bases=alt,
    )

    # 使用变异长度的窗口，最小 16384
    window_size = max(16384, len(ref) * 2)
    interval = variant.reference_interval.resize(window_size)

    scorer = variant_scorers.CenterMaskScorer(
        width=None,  # 可尝试改为固定宽度，如512
        aggregation_type=variant_scorers.AggregationType.DIFF_SUM_LOG2,
        requested_output=dna_client.OutputType.RNA_SEQ,
    )

    result = dna_model.score_variant(
        interval=interval,
        variant=variant,
        variant_scorers=[scorer],
        organism=dna_client.Organism.HOMO_SAPIENS,
    )

    delta_df = result.to_dataframe()  # 确保返回 DataFrame
    return delta_df
