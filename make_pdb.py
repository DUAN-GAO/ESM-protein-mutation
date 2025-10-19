#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal ESMFold inference script (no JAX dependency)
Usage:
  python make_pdb.py --sequence "GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE"
"""

import argparse
import re
import os
import gc
import hashlib
import numpy as np
import torch
from scipy.special import softmax


def get_hash(x):
    return hashlib.sha1(x.encode()).hexdigest()


def parse_output(output):
    """Parse ESMFold output to extract metrics"""
    pae = (output["aligned_confidence_probs"][0] * np.arange(64)).mean(-1) * 31
    plddt = output["plddt"][0, :, 1]
    bins = np.append(0, np.linspace(2.3125, 21.6875, 63))
    sm_contacts = softmax(output["distogram_logits"], -1)[0]
    sm_contacts = sm_contacts[..., bins < 8].sum(-1)
    xyz = output["positions"][-1, 0, :, 1]
    mask = output["atom37_atom_exists"][0, :, 1] == 1

    return {
        "pae": pae[mask, :][:, mask],
        "plddt": plddt[mask],
        "sm_contacts": sm_contacts[mask, :][:, mask],
        "xyz": xyz[mask],
    }


def recursive_to_numpy(obj):
    """Recursively convert torch tensors to numpy arrays"""
    if torch.is_tensor(obj):
        return obj.cpu().numpy()
    elif isinstance(obj, dict):
        return {k: recursive_to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        cls = type(obj)
        return cls(recursive_to_numpy(v) for v in obj)
    else:
        return obj


def main():
    parser = argparse.ArgumentParser(description="Run ESMFold protein structure prediction")
    parser.add_argument("--sequence", type=str, required=True, help="Protein sequence (A-Z)")
    args = parser.parse_args()

    # Clean sequence
    sequence = re.sub("[^A-Z]", "", args.sequence.upper())
    if len(sequence) == 0:
        raise ValueError("Invalid or empty sequence. Must contain letters A-Z.")

    # Generate job name and output directory
    jobname = "esmfold_job_" + get_hash(sequence)[:6]
    os.makedirs(jobname, exist_ok=True)

    print(f"🧬 Sequence length: {len(sequence)}")
    print(f"📂 Output folder: {jobname}")

    # Default settings
    model_path = "esmfold.model"  # default model path
    num_recycles = 3
    chain_linker = 25

    # Load model
    print("📦 Loading ESMFold model...")
    model = torch.load(model_path) #, weights_only=False deleted
    model.eval().cuda().requires_grad_(False)

    # Optimize chunk size
    # model.set_chunk_size(64 if len(sequence) > 700 else 128) 
    torch.cuda.empty_cache()
    model.eval().cuda().requires_grad_(False)
    model.set_chunk_size(16)   # 原来 64/128
    num_recycles = 1 

    torch.cuda.empty_cache()
    print("🚀 Running inference...")

    output = model.infer(
        sequence,
        num_recycles=num_recycles,
        chain_linker="X" * chain_linker,
        residue_index_offset=512,
    )

    pdb_str = model.output_to_pdb(output)[0]
    # Convert all torch tensors in output to numpy
    output = recursive_to_numpy(output)

    ptm = output["ptm"][0]
    plddt = output["plddt"][0, ..., 1].mean()
    O = parse_output(output)

    print(f"✅ Done! PTM={ptm:.3f}, pLDDT={plddt:.3f}")

    prefix = f"{jobname}/ptm{ptm:.3f}_r{num_recycles}"
    np.savetxt(f"{prefix}.pae.txt", O["pae"], "%.3f")
    with open(f"{prefix}.pdb", "w") as f:
        f.write(pdb_str)

    print(f"📁 Results saved to:\n  {prefix}.pdb\n  {prefix}.pae.txt")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
