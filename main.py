import argparse
import requests
import pandas as pd
from alphagenome.data import genome
from alphagenome.models import dna_client
from alphagenome.models import variant_scorers


# 固定 API Key
API_KEY = "AIzaSyD5Kht8QzCPkHeJ456_Tf_eBWirtKhmaRU"


def rsid_to_variant_info(rsid):
    """
    使用 NCBI Variation API 获取 hg38 坐标：
    输入 rsID，返回:
        chromosome='chr1'
        position=1234567
        reference_bases='A'
        alternate_bases='G'
    """

    numeric_id = rsid.replace("rs", "")
    url = f"https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/{numeric_id}"

    r = requests.get(url, timeout=10)

    if r.status_code != 200:
        raise ValueError(f"{rsid} NCBI API 查询失败: HTTP {r.status_code}")

    data = r.json()

    # 解析 primary_assembly（GRCh38）
    placements = data.get("primary_snapshot_data", {}).get("placements_with_allele", [])

    # 找 hg38 + 有 REF/ALT 的 SNV
    for p in placements:
        if not p.get("is_ptlp"):  # 只取主组学坐标
            continue

        for allele in p.get("alleles", []):
            spdi = allele.get("allele", {}).get("spdi")
            if spdi:
                # 只处理 SNV
                if len(spdi["deleted_sequence"]) == 1 and len(spdi["inserted_sequence"]) == 1:
                    chrom = spdi["seq_id"].replace("NC_", "").replace(".11", "")
                    chrom = chrom.lstrip("0")  # 例如 "000001" → "1"

                    ref = spdi["deleted_sequence"]
                    alt = spdi["inserted_sequence"]
                    pos = int(spdi["position"]) + 1  # SPDI 坐标从 0 开始 → 转为 1-based

                    return f"chr{chrom}", pos, ref, alt

    raise ValueError(f"{rsid} 未找到 hg38 SNV 位点")


def main(rsid):
    # 初始化 AlphaGenome 模型
    print('API reached...')
    dna_model = dna_client.create(API_KEY)

    # rsID → hg38 variant
    chrom, pos, ref, alt = rsid_to_variant_info(rsid)
    print(f"[INFO] {rsid} -> {chrom}:{pos} {ref}>{alt}")

    variant = genome.Variant(
        chromosome=chrom,
        position=pos,
        reference_bases=ref,
        alternate_bases=alt,
    )

    # 定义预测区间（2048 bp）
    sequence_length = 2048
    interval = variant.reference_interval.resize(sequence_length)

    # 定义 scorer（CenterMaskScorer）
    scorer = variant_scorers.CenterMaskScorer(
        width=None,
        aggregation_type=variant_scorers.AggregationType.DIFF_SUM_LOG2,
        requested_output=dna_client.OutputType.RNA_SEQ,
    )

    # 打分
    score_result = dna_model.score_variant(
        interval=interval,
        variant=variant,
        variant_scorers=[scorer],
        organism=dna_client.Organism.HOMO_SAPIENS,
    )

    # 保存到 rsID.txt
    output_file = f"{rsid}.txt"
    with open(output_file, "w") as f:
        f.write(str(score_result[0].var))

    print(f"[RESULT] 已保存 score_result[0].var 到 {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query rsID and score variant with AlphaGenome.")
    parser.add_argument("--rsid", type=str, required=True, help="dbSNP rsID, e.g. rs5934683")

    args = parser.parse_args()
    main(args.rsid)
