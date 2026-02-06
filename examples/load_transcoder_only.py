#!/usr/bin/env python3
"""Example: load transcoders from the registry and print config info.

Usage:
    uv run python examples/load_transcoder_only.py
"""

from llm_circuits.transcoders.circuit_tracer_loader import load_transcoder
from llm_circuits.transcoders.registry import get_spec, list_specs


def main() -> None:
    # Show all available specs
    print("Available model specs:")
    for spec in list_specs():
        print(f"  {spec.size:>5s}  model={spec.hf_model_id}  transcoder={spec.transcoder_repo}")

    # Load the smallest transcoder
    spec = get_spec("0.6b")
    print(f"\nLoading transcoder for {spec.size} from {spec.transcoder_repo} ...")

    result = load_transcoder(spec, device="cpu")
    print(f"  Type:    {type(result.transcoder).__name__}")
    print(f"  Repo:    {result.repo_id}")
    print(f"  Config:  {result.config}")


if __name__ == "__main__":
    main()
