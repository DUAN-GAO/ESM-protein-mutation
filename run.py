#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gzip

def open_vcf(path, mode="rt"):
    """自动识别是否为 gz 文件"""
    if path.endswith(".gz"):
        return gzip.open(path, mode)
    else:
        return open(path, mode)

def is_protein_coding(info_field):
    """判断 CSQ 是否包含 protein_coding 相关注释"""
    if "CSQ=" not in info_field:
        return False

    raw = info_field.split("CSQ=")[1]
    entries = raw.split(",")

    # protein_coding 判定关键字
    keywords = ["protein_coding", "missense", "synonymous", "stop_gained", "start_lost"]

    for entry in entries:
        if any(k in entry for k in keywords):
            return True

    return False


def split_vcf(vcf_path, out_pc="protein_coding.vcf", out_nonpc="non_protein_coding.vcf"):
    f_out_pc = open(out_pc, "w")
    f_out_non = open(out_nonpc, "w")

    with open_vcf(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                # 头部写入两个文件
                f_out_pc.write(line)
                f_out_non.write(line)
                continue

            parts = line.strip().split("\t")
            if len(parts) < 8:
                f_out_non.write(line)
                continue

            info = parts[7]

            if is_protein_coding(info):
                f_out_pc.write(line)
            else:
                f_out_non.write(line)

    f_out_pc.close()
    f_out_non.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="按 protein-coding 注释拆分 VCF 文件")
    parser.add_argument("--vcf", required=True, help="输入 VCF 文件")
    parser.add_argument("--out_pc", default="protein_coding.vcf", help="输出 protein_coding VCF")
    parser.add_argument("--out_nonpc", default="non_protein_coding.vcf", help="输出 non-protein-coding VCF")
    args = parser.parse_args()

    split_vcf(args.vcf, args.out_pc, args.out_nonpc)
