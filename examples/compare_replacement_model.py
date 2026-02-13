#!/usr/bin/env python3
"""Example: compare original vs. transcoder-replaced Qwen3-0.6B outputs.

Loads the model and its transcoders, runs ``compare_models``, and prints
per-position KL divergence, cosine similarity, and top-1 agreement.

Usage:
    uv run python examples/compare_replacement_model.py
"""

from __future__ import annotations

import torch

from llm_circuits.circuits.replacement_model import compare_models
from llm_circuits.instrumentation.chat import prepare_messages
from llm_circuits.models.qwen3 import load_qwen3
from llm_circuits.settings import default_device, default_dtype
from llm_circuits.transcoders.circuit_tracer_loader import load_transcoder


def main() -> None:
    device = default_device()
    dtype = default_dtype()
    dtype_str = "bf16" if dtype == torch.bfloat16 else "fp32"
    prompt = "The capital of France is"

    print(f"Device: {device}  Dtype: {dtype}")

    # --- Load model -----------------------------------------------------------
    print(f"Loading Qwen3-0.6B ({dtype_str}) ...")
    model, tokenizer = load_qwen3("0.6b", dtype_str=dtype_str, device_map=device)
    model.eval()

    # --- Load transcoders -----------------------------------------------------
    print("Loading transcoders ...")
    loaded = load_transcoder("qwen3-0.6b", device=device, dtype=dtype)
    tc = loaded.transcoder
    print(f"  Type: {type(tc).__name__}  Repo: {loaded.repo_id}")

    # --- Tokenize via chat template -------------------------------------------
    messages, n_bos_tokens = prepare_messages(prompt, "qwen3")
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True,
    ).to(device)
    tokens = [tokenizer.decode(t) for t in input_ids[0]]

    # --- Compare --------------------------------------------------------------
    print(f"\nPrompt: {prompt!r}")
    print(f"Tokens: {tokens}\n")

    with torch.no_grad():
        result = compare_models(model, tc, input_ids, n_bos_tokens=n_bos_tokens)

    # --- Print per-position metrics -------------------------------------------
    orig_preds = result.original_logits.argmax(dim=-1)
    repl_preds = result.replacement_logits.argmax(dim=-1)

    print(
        f"{'Pos':>3}  {'Token':>12}  {'Orig pred':>12}  {'Repl pred':>12}"
        f"  {'KL div':>10}  {'Cos sim':>10}  {'Top-1':>6}"
    )
    print("-" * 78)
    for i, tok in enumerate(tokens):
        orig_tok = tokenizer.decode(orig_preds[i].item())
        repl_tok = tokenizer.decode(repl_preds[i].item())
        kl = result.kl_divergence[i].item()
        cos = result.cosine_similarity[i].item()
        agree = "yes" if result.top1_agreement[i].item() else "NO"
        print(
            f"{i:3d}  {tok:>12s}  {orig_tok:>12s}  {repl_tok:>12s}"
            f"  {kl:10.4f}  {cos:10.4f}  {agree:>6s}"
        )

    # --- Summary --------------------------------------------------------------
    mean_kl = result.kl_divergence.mean().item()
    mean_cos = result.cosine_similarity.mean().item()
    pct_agree = result.top1_agreement.float().mean().item() * 100

    print(f"\nMean KL divergence:   {mean_kl:.4f}")
    print(f"Mean cosine sim:      {mean_cos:.4f}")
    print(f"Top-1 agreement:      {pct_agree:.1f}%")

    # --- Per-layer reconstruction error ---------------------------------------
    if result.reconstruction_errors:
        print(f"\n{'Layer':>5}  {'Mean L2 error':>14}")
        print("-" * 22)
        for layer_idx in sorted(result.reconstruction_errors):
            err = result.reconstruction_errors[layer_idx].mean().item()
            print(f"{layer_idx:5d}  {err:14.4f}")


if __name__ == "__main__":
    main()
