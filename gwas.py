import pandas as pd
import requests
import time

# -----------------------------
# 输入文件
# -----------------------------
input_file = "gwas-catalog-download-associations-v1.0-full.tsv"
output_vcf = "cancer_snps.vcf"

# -----------------------------
# 读取 GWAS association 文件
# -----------------------------
print("[INFO] Loading GWAS associations file...")
df = pd.read_csv(input_file, sep="\t", low_memory=False)
print(f"[INFO] Total SNP records: {len(df)}")

# -----------------------------
# 过滤 Cancer 相关
# -----------------------------
df_cancer = df[df["DISEASE/TRAIT"].str.contains("cancer", case=False, na=False)]
print(f"[INFO] Cancer-related SNPs found: {len(df_cancer)}")

# -----------------------------
# 处理 CHR_POS，过滤无效位点
# -----------------------------
def parse_pos(pos):
    """只保留单个位点，忽略范围（例如 '190188403 x 190248283'）"""
    try:
        if isinstance(pos, str) and (' ' in pos or 'x' in pos):
            return None
        return int(pos)
    except:
        return None

df_cancer["POS"] = df_cancer["CHR_POS"].apply(parse_pos)
df_cancer = df_cancer.dropna(subset=["POS"])
df_cancer["POS"] = df_cancer["POS"].astype(int)

# -----------------------------
# 过滤 REF 为 N 或缺失的位点
# -----------------------------
def parse_ref(snp_allele):
    """从 STRONGEST SNP-RISK ALLELE 获取风险等位基因"""
    if pd.isna(snp_allele):
        return None
    # 通常格式 rsID-风险等位基因
    if "-" in snp_allele:
        ref = snp_allele.split("-")[1]
        if ref.upper() == "N" or ref == "":
            return None
        return ref.upper()
    return None

df_cancer["REF"] = df_cancer["STRONGEST SNP-RISK ALLELE"].apply(parse_ref)
df_cancer = df_cancer.dropna(subset=["REF"])

# -----------------------------
# dbSNP API 获取 ALT
# -----------------------------
def fetch_alt_from_dbsnp(rsid):
    """调用 NCBI dbSNP REST API 获取 ALT 等位基因"""
    url = f"https://api.ncbi.nlm.nih.gov/variation/v0/beta/refsnp/{rsid.replace('rs','')}"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return "."
        data = r.json()
        # 获取参考等位基因和其它等位基因
        primary_snapshot = data.get("primary_snapshot_data", {})
        alleles = primary_snapshot.get("allele_annotations", [])
        if alleles:
            # 有些条目可能复杂，这里简单取 REF 已知，ALT 取第一个非 REF
            alt_alleles = []
            for allele in alleles:
                seq = allele.get("allele", {}).get("spdi", {}).get("inserted_sequence")
                if seq and seq.upper() != "N":
                    alt_alleles.append(seq.upper())
            if alt_alleles:
                return ",".join(alt_alleles)
        return "."
    except Exception as e:
        return "."

# 逐条获取 ALT（大文件可能慢）
alts = []
print("[INFO] Fetching ALT from dbSNP (this may take a while)...")
for idx, row in df_cancer.iterrows():
    rsid = row["SNP_ID_CURRENT"]
    alt = fetch_alt_from_dbsnp(rsid)
    alts.append(alt)
    time.sleep(0.1)  # 避免请求过快
df_cancer["ALT"] = alts

# -----------------------------
# 生成 VCF
# -----------------------------
vcf_columns = ["CHR_ID", "POS", "SNP_ID_CURRENT", "REF", "ALT"]
df_vcf = df_cancer[vcf_columns].copy()
df_vcf = df_vcf.rename(columns={"CHR_ID": "#CHROM", "SNP_ID_CURRENT": "ID"})

# 填充 VCF 固定列
df_vcf["QUAL"] = "."
df_vcf["FILTER"] = "PASS"
df_vcf["INFO"] = "GWAS=YES"

# 输出 VCF
with open(output_vcf, "w") as f:
    # 写 VCF header
    f.write("##fileformat=VCFv4.2\n")
    f.write("##source=GWAS_catalog_cancer\n")
    f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
    for _, row in df_vcf.iterrows():
        f.write(
            f"{row['#CHROM']}\t{row['POS']}\t{row['ID']}\t{row['REF']}\t{row['ALT']}\t{row['QUAL']}\t{row['FILTER']}\t{row['INFO']}\n"
        )

print(f"[DONE] VCF saved to {output_vcf}")
