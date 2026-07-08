"""Smoke-test the device wiring end to end (TPU/XLA, CUDA, MPS, or CPU).

Run locally (resolves to cuda/mps/cpu):

    uv run python examples/tpu_smoke_test.py

Run on the TPU VM (launch.sh injects LLM_CIRCUITS_DEVICE=tpu -> "xla"):

    gcp/launch.sh examples/tpu_smoke_test.py

Two stages:
1. Tensor stage — allocate + matmul on the resolved device.
2. Model stage — load Qwen3-0.6B and generate a few tokens (skip with
   ``--tensor-only`` to avoid the model download).
"""

from __future__ import annotations

import argparse
import time

import torch

from llm_circuits.settings import default_device, default_dtype

MODEL_SIZE = "0.6b"
MAX_NEW_TOKENS = 8
PROMPT = "The capital of France is"


def tensor_stage(device: str, dtype: torch.dtype) -> None:
    t0 = time.perf_counter()
    x = torch.randn(512, 512, device=device, dtype=dtype)
    y = x @ x.T
    if device == "xla":
        import torch_xla

        torch_xla.sync()  # materialise the lazy XLA graph
    mean = float(y.float().mean())
    print(
        f"  matmul ok: {tuple(y.shape)} on {y.device}, mean={mean:.4f} "
        f"({time.perf_counter() - t0:.2f}s)"
    )


def model_stage(device: str, dtype_str: str) -> None:
    from llm_circuits.models.qwen3 import load_qwen3

    t0 = time.perf_counter()
    model, tokenizer = load_qwen3(MODEL_SIZE, dtype_str=dtype_str, device_map=device)
    model.eval()
    print(f"  model loaded on {next(model.parameters()).device} ({time.perf_counter() - t0:.1f}s)")

    inputs = tokenizer(PROMPT, return_tensors="pt").to(next(model.parameters()).device)
    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"  generate ok ({time.perf_counter() - t0:.1f}s): {text!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tensor-only", action="store_true", help="skip the model load/generate stage"
    )
    args = parser.parse_args()

    device = default_device()
    dtype = default_dtype()
    print(f"resolved device: {device} | dtype: {dtype}")

    print("[1/2] tensor stage")
    tensor_stage(device, dtype)

    if args.tensor_only:
        print("[2/2] model stage skipped (--tensor-only)")
    else:
        dtype_str = "bf16" if dtype == torch.bfloat16 else "fp32"
        print(f"[2/2] model stage (Qwen3-{MODEL_SIZE.upper()}, {dtype_str})")
        model_stage(device, dtype_str)

    print("SMOKE_TEST_OK")


if __name__ == "__main__":
    main()
