import torch
import argparse
from esm import pretrained
import time


def compute_delta_score(wildtype_sequence, mutation_position, wildtype_aa, mutant_aa):
    # 加载预训练模型 ESM-1b
    model, alphabet = pretrained.esm1b_t33_650M_UR50S()
    model.eval()

    # 构建批次转换器
    batch_converter = alphabet.get_batch_converter()

    # 1. 生成带掩码的序列
    seq = list(wildtype_sequence)
    seq[mutation_position] = alphabet.get_tok(alphabet.mask_idx)
    masked_sequence_str = "".join(seq)

    # 2. 转换为Token格式
    data = [("protein", masked_sequence_str)]
    _, _, batch_tokens = batch_converter(data)

    # 3. 输入模型
    with torch.no_grad():
        outputs = model(batch_tokens, repr_layers=[])
        logits = outputs["logits"]

    # 4. 提取突变位点logits并softmax成概率
    mask_logits = logits[0, mutation_position+1, :]
    probabilities = torch.softmax(mask_logits, dim=0)

    # 5. 获取野生型和突变型氨基酸概率
    wt_idx = alphabet.get_idx(wildtype_aa)
    mut_idx = alphabet.get_idx(mutant_aa)

    p_wild = probabilities[wt_idx].item()
    p_mutant = probabilities[mut_idx].item()

    # 6. 计算Δscore
    epsilon = 1e-10
    delta_score = torch.log(torch.tensor(p_mutant + epsilon) / torch.tensor(p_wild + epsilon)).item()

    return delta_score, p_wild, p_mutant


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute ESM mutation delta score")
    parser.add_argument("--seq", type=str, required=True,
                        help="Wild-type protein sequence (string or FASTA file path)")
    parser.add_argument("--pos", type=int, required=True,
                        help="Mutation position (0-based index)")
    parser.add_argument("--wt", type=str, required=True,
                        help="Wild-type amino acid (single letter)")
    parser.add_argument("--mut", type=str, required=True,
                        help="Mutant amino acid (single letter)")

    args = parser.parse_args()

    # 如果输入是fasta文件，读取序列
    sequence = args.seq
    if sequence.endswith(".fasta") or sequence.endswith(".fa"):
        with open(sequence, "r") as f:
            lines = f.readlines()
            sequence = "".join([l.strip() for l in lines if not l.startswith(">")])
    start_time = time.time()

    delta, p_wt, p_mut = compute_delta_score(sequence, args.pos, args.wt, args.mut)
    
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.2f} seconds")
    print(f"Wild type amino acid {args.wt} likelihood: {p_wt:.6f}")
    print(f"Mutant type amino acid {args.mut} likelihood: {p_mut:.6f}")
    print(f"Δscore (log likelihood ratio): {delta:.4f}")
