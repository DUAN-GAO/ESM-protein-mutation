#!/usr/bin/env python3
import sys
import gzip

PROTEIN_ALTERING = {
    "missense_variant",
    "synonymous_variant",
    "stop_gained",
    "stop_lost",
    "frameshift_variant",
    "inframe_insertion",
    "inframe_deletion",
    "start_lost",
}

def parse_csq(csq_string):
    csq_entries = csq_string.split(",")
    consequences = []
    for entry in csq_entries:
        fields = entry.split("|")
        if len(fields) > 1:
            consequences.append(fields[1])
    return consequences


def is_protein_affecting(csq_string):
    consequences = parse_csq(csq_string)
    return any(cons in PROTEIN_ALTERING for cons in consequences)


def open_file(path, mode="rt"):
    if path.endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)


def main(input_vcf, out_protein, out_nonprotein):
    fin = open_file(input_vcf, "rt")
    fout_pro = open_file(out_protein, "wt")
    fout_non = open_file(out_nonprotein, "wt")

    for line in fin:
        if line.startswith("#"):
            fout_pro.write(line)
            fout_non.write(line)
            continue

        fields = line.strip().split("\t")
        info = fields[7]

        csq_string = None
        for item in info.split(";"):
            if item.startswith("CSQ="):
                csq_string = item[4:]
                break

        if csq_string is None:
            fout_non.write(line)
            continue

        if is_protein_affecting(csq_string):
            fout_pro.write(line)
        else:
            fout_non.write(line)

    fin.close()
    fout_pro.close()
    fout_non.close()


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python split_by_exonic_status.py input.vcf protein.vcf nonprotein.vcf")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3])
