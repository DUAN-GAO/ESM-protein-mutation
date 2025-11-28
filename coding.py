import re
import urllib.parse

def parse_vcf_csq_format(csq_field):
    """
    从 VCF CSQ 字段中识别蛋白突变信息，例如:
    NP_057208.3:p.Val9Leu → V9L
    返回: pos(0-based), wt, mut, refseq_id, protein_change
    """
    fields = csq_field.split("|")
    if len(fields) < 11:
        return None

    protein_change = fields[10]  # NP_057208.3:p.Val9Leu
    refseq_id = fields[0]

    if not protein_change or ":p." not in protein_change:
        return None

    aa_change = protein_change.split(":p.")[1]
    aa_change = urllib.parse.unquote(aa_change)  # 处理 %3D 等 URL 编码

    # 跳过同义突变或缺失
    if aa_change in ("=", "-", "") or aa_change.endswith("="):
        return None

    # 使用正则解析标准 HGVS 蛋白突变格式
    m = re.match(r'^([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$', aa_change)
    if not m:
        # 有些可能是单字母形式，例如 V9L
        m2 = re.match(r'^([A-Z])(\d+)([A-Z])$', aa_change)
        if not m2:
            return None
        wt, pos_str, mut = m2.groups()
    else:
        wt, pos_str, mut = m.groups()
        # 将三字母 aa 转为单字母
        aa3to1 = {
            'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Glu':'E','Gln':'Q',
            'Gly':'G','His':'H','Ile':'I','Leu':'L','Lys':'K','Met':'M','Phe':'F',
            'Pro':'P','Ser':'S','Thr':'T','Trp':'W','Tyr':'Y','Val':'V','Sec':'U','Pyl':'O'
        }
        wt = aa3to1.get(wt, wt)
        mut = aa3to1.get(mut, mut)

    pos = int(pos_str) - 1  # 0-based
    return pos, wt, mut, refseq_id, protein_change