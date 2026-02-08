#!/usr/bin/env python3
"""Example: cache transcoder weights locally.

Usage:
    uv run python examples/cache_transcoder.py
"""

from llm_circuits.transcoders.circuit_tracer_loader import cache_transcoder


def main() -> None:
    repo = "mwhanna/qwen3-0.6b-transcoders-lowl0"

    # cache_dir defaults to settings.transcoder_cache_dir() (<project>/.cache/transcoders)
    print(f"Caching {repo} ...")
    cache_transcoder(repo)
    print("Done.")


if __name__ == "__main__":
    main()
