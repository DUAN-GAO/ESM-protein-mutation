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

    # 窗口大小
    window_size = max(16384, len(ref) * 2)
    interval = variant.reference_interval.resize(window_size)

    scorer = variant_scorers.CenterMaskScorer(
        width=None,  # 可改为固定宽度，如 512
        aggregation_type=variant_scorers.AggregationType.DIFF_SUM_LOG2,
        requested_output=dna_client.OutputType.RNA_SEQ,
    )

    # score_variant 返回的是 list，每个 scorer 一个元素
    result_list = dna_model.score_variant(
        interval=interval,
        variant=variant,
        variant_scorers=[scorer],
        organism=dna_client.Organism.HOMO_SAPIENS,
    )

    # 取第一个 scorer 的结果
    result = result_list[0]

    # 变成 DataFrame
    delta_df = result.to_dataframe()
    return delta_df
