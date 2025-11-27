# ---------- main.py ----------

from alphagenome.data import genome
from alphagenome.models import dna_client, variant_scorers

API_KEY = "AIzaSyD5Kht8QzCPkHeJ456_Tf_eBWirtKhmaRU"

# 初始化一次，避免重复加载模型（提高性能）
dna_model = dna_client.create(API_KEY)

scorer = variant_scorers.CenterMaskScorer(
    width=None,
    aggregation_type=variant_scorers.AggregationType.DIFF_SUM_LOG2,
    requested_output=dna_client.OutputType.RNA_SEQ,
)

def score_variant(chrom, pos, ref, alt, rsid=None):
    """
    输入位点信息，输出评分结果(dict结构)
    """
    tag = rsid if rsid else f"{chrom}:{pos} {ref}>{alt}"
    print(f"[INFO] Scoring {tag}")

    try:
        variant = genome.Variant(
            chromosome=chrom,
            position=int(pos),
            reference_bases=ref,
            alternate_bases=alt,
        )

        interval = variant.reference_interval.resize(16384)

        result = dna_model.score_variant(
            interval=interval,
            variant=variant,
            variant_scorers=[scorer],
            organism=dna_client.Organism.HOMO_SAPIENS,
        )

        df = result[0].var
        delta_value = float(abs(df["delta"]).mean())

        print(f"[OK] {tag} → Δ={delta_value}")

        return {
            "rsid": rsid if rsid else "NA",
            "chrom": chrom,
            "pos": pos,
            "ref": ref,
            "alt": alt,
            "delta_score": delta_value
        }

    except Exception as e:
        print(f"[ERROR] {tag} 失败: {e}")
        return {
            "rsid": rsid if rsid else "NA",
            "chrom": chrom,
            "pos": pos,
            "ref": ref,
            "alt": alt,
            "delta_score": None,
            "error": str(e)
        }
