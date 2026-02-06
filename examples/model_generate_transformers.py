#!/usr/bin/env python3
"""Example: generate text with a Qwen3 model via Transformers.

Usage:
    uv run python examples/model_generate_transformers.py
"""

from llm_circuits.models.qwen3 import load_qwen3


def main() -> None:
    size = "0.6b"
    prompt = "The capital of France is"

    print(f"Loading Qwen3-{size} ...")
    model, tokenizer = load_qwen3(size, dtype_str="bf16", device_map="auto")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=64)
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print(f"Prompt: {prompt}")
    print(f"Output: {text}")


if __name__ == "__main__":
    main()
