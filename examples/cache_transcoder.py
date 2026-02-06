#!/usr/bin/env python3
"""Example: cache transcoder weights locally.

Usage:
    uv run python examples/cache_transcoder.py
"""

from llm_circuits.transcoders.circuit_tracer_loader import cache_transcoder


def main() -> None:
    repo = "mwhanna/qwen3-0.6b-transcoders-lowl0"
    cache_dir = ".cache/transcoders"

    print(f"Caching {repo} -> {cache_dir}")
    cache_transcoder(repo, cache_dir)
    print("Done.")


if __name__ == "__main__":
    main()
