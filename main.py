import argparse
import requests
from alphagenome.data import genome
from alphagenome.models import dna_client, variant_scorers


API_KEY = "AIzaSyD5Kht8QzCPkHeJ456_Tf_eBWirtKhmaRU"


def rsid_to_variant_info(rsid):

    numeric_id = rsid.replace("rs", "")
    url = f"https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/{numeric_id}"
    r = requests.get(url, timeout=10)

    if r.status_code != 200:
        raise ValueError(f"{rsid} API 查询失败: HTTP {r.status_code}")

    data = r.json()
    placements = data.get("primary_snapshot_data", {}).get("placements_with_allele", [])

    for p in placements:
        if not p.get("is_ptlp"):
            continue

        for allele in p.get("alleles", []):
            spdi = allele.get("allele", {}).get("spdi")
            if not spdi:
                continue

            if len(spdi["deleted_sequence"]) != 1 or len(spdi["inserted_sequence"]) != 1:
                continue

            seq_id = spdi["seq_id"]  # like NC_000011.10
            num = seq_id.replace("NC_", "").split(".")[0].lstrip("0")
            num = num if num else "0"
            chrom = f"chr{num}"

            ref = spdi["deleted_sequence"]
            alt = spdi["inserted_sequence"]
            pos = spdi["position"] + 1

            return chrom, pos, ref, alt

    raise ValueError(f"{rsid} 未找到 hg38 canonical SNV")


def main(rsid):
    print("API reached...")

    dna_model = dna_client.create(API_KEY)

    chrom, pos, ref, alt = rsid_to_variant_info(rsid)
    print(f"[INFO] {rsid} -> {chrom}:{pos} {ref}>{alt}")

    variant = genome.Variant(
        chromosome=chrom,
        position=pos,
        reference_bases=ref,
        alternate_bases=alt,
    )

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

    delta = result[0].var  # 数值

    # ----------- 写 CSV 格式 -----------
    with open(f"{rsid}.txt", "w") as f:
        f.write(f"{rsid},{chrom},{pos},{ref},{alt},{delta}\n")

    print(f"[OK] {rsid} 已保存 CSV 格式输出")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rsid", type=str, required=True)
    args = parser.parse_args()
    main(args.rsid)
