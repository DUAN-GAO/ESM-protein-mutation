import gzip
import pandas as pd
import myvariant
from alphagenome.data import genome
from alphagenome.models import dna_client
from alphagenome.models import variant_scorers

API_KEY = "AIzaSyD5Kht8QzCPkHeJ456_Tf_eBWirtKhmaRU"

def extract_rsids_from_vcf(vcf_path):
    opener = gzip.open if vcf_path.endswith(".gz") else open
    rsids = []
    with opener(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            info = fields[7] if len(fields) > 7 else ""
            if "CSQ=" in info:
                csq_entries = info.split("CSQ=")[1].split(",")
                for entry in csq_entries:
                    cols = entry.split("|")
                    for col in cols:
                        col = col.strip()
                        if col.startswith("rs") and col[2:].isdigit():
                            rsids.append(col)
    return list(set(rsids))

def rsid_to_variant_info(rsid):
    mv = myvariant.MyVariantInfo()
    out = mv.getvariant(rsid, fields="dbsnp.hg38")
    if not out or "dbsnp" not in out:
        return None
    dbsnp = out["dbsnp"]
    hg38 = dbsnp.get("hg38", {})
    chrom = dbsnp.get("chrom")
    pos = hg38.get("start")
    ref = dbsnp.get("ref")
    alt = dbsnp.get("alt")
    if None in [chrom, pos, ref, alt]:
        return None
    return f"chr{chrom}", pos, ref, alt

def score_rsid(dna_model, rsid):
    info = rsid_to_variant_info(rsid)
    if not info:
        print(f"[WARN] {rsid} 信息不完整或未找到 hg38 坐标")
        return None
    chrom, pos, ref, alt = info
    variant = genome.Variant(chromosome=chrom, position=pos, reference_bases=ref, alternate_bases=alt)
    interval = variant.reference_interval.resize(2048)
    scorer = variant_scorers.CenterMaskScorer(
        width=None,
        aggregation_type=variant_scorers.AggregationType.DIFF_SUM_LOG2,
        requested_output=dna_client.OutputType.RNA_SEQ,
    )
    try:
        score_result = dna_model.score_variant(
            interval=interval,
            variant=variant,
            variant_scorers=[scorer],
            organism=dna_client.Organism.HOMO_SAPIENS,
        )
        return chrom, pos, ref, alt, score_result[0].var
    except Exception as e:
        print(f"[ERROR] {rsid} 打分失败: {e}")
        return None

def main(vcf_path, out_csv="results.csv"):
    dna_model = dna_client.create(API_KEY)
    rsids = extract_rsids_from_vcf(vcf_path)
    results = []
    for rsid in rsids:
        print(f"[RUN] 处理 {rsid}")
        res = score_rsid(dna_model, rsid)
        if res:
            chrom, pos, ref, alt, delta = res
            results.append({"rsid": rsid, "chrom": chrom, "pos": pos, "ref": ref, "alt": alt, "delta_score": delta})
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"[FINISHED] 已保存 {len(results)} 条结果到 {out_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcf", required=True)
    parser.add_argument("--out", default="results.csv")
    args = parser.parse_args()
    main(args.vcf, args.out)
