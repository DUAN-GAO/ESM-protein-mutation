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
    使用 NCBI Variation API 获取 hg38 坐标（canonical chr1..22,X,Y,M）
    自动将 NC_000011.10 → chr11
    """

    numeric_id = rsid.replace("rs", "")
    url = f"https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/{numeric_id}"

    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        raise ValueError(f"{rsid} NCBI API 查询失败: HTTP {r.status_code}")

    data = r.json()

    placements = data.get("primary_snapshot_data", {}).get("placements_with_allele", [])

    for p in placements:

        # 只选主参考序列：is_ptlp = True
        if not p.get("is_ptlp"):
            continue

        for allele in p.get("alleles", []):
            spdi = allele.get("allele", {}).get("spdi")
            if not spdi:
                continue

            # 只处理 SNV
            if len(spdi["deleted_sequence"]) != 1 or len(spdi["inserted_sequence"]) != 1:
                continue

            seq_id = spdi["seq_id"]  # 如 NC_000011.10

            # ----------- canonical chromosome mapping -----------
            if seq_id.startswith("NC_"):
                # Example: NC_000011.10 → 000011.10 → 11
                numeric_part = seq_id.replace("NC_", "")
                chrom_num = numeric_part.split(".")[0]     # 000011
                chrom_num = chrom_num.lstrip("0")          # → 11

                if chrom_num == "":
                    chrom_num = "0"

                chrom = f"chr{chrom_num}"
            else:
                # Fallback
                chrom = seq_id

            ref = spdi["deleted_sequence"]
            alt = spdi["inserted_sequence"]
            pos = int(spdi["position"]) + 1  # SPDI 是 0-based

            return chrom, pos, ref, alt

    raise ValueError(f"{rsid} 未找到 hg38 canonical SNV 位点")


def main(rsid):
    print("API reached...")

    # 创建 AlphaGenome 模型
    dna_model = dna_client.create(API_KEY)

    # 获取 hg38 定位
    chrom, pos, ref, alt = rsid_to_variant_info(rsid)
    print(f"[INFO] {rsid} -> {chrom}:{pos} {ref}>{alt}")

    variant = genome.Variant(
        chromosome=chrom,
        position=pos,
        reference_bases=ref,
        alternate_bases=alt,
    )

    # AlphaGenome 支持长度 16384+
    sequence_length = 16384
    interval = variant.reference_interval.resize(sequence_length)

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

    # 保存
    output_file = f"{rsid}.txt"
    with open(output_file, "w") as f:
        f.write(str(score_result[0].var))

    print(f"[RESULT] 已保存 score_result[0].var 到 {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query rsID and score variant with AlphaGenome.")
    parser.add_argument("--rsid", type=str, required=True, help="dbSNP rsID, e.g. rs5934683")

    args = parser.parse_args()
    main(args.rsid)
