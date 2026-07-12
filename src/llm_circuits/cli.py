"""Typer CLI for llm-circuits."""

from __future__ import annotations

import sys

import typer
from rich import print as rprint
from rich.table import Table

app = typer.Typer(
    name="llm-circuits",
    help="Mechanistic-interpretability research toolkit for LLM circuits.",
    no_args_is_help=True,
)


@app.command()
def info():
    """Print environment info and available model specs."""
    import torch
    import transformers

    from llm_circuits import __version__
    from llm_circuits.transcoders.registry import list_registry

    rprint(f"[bold]llm-circuits[/bold] v{__version__}")
    rprint(f"  Python        {sys.version.split()[0]}")
    rprint(f"  PyTorch       {torch.__version__}")
    rprint(f"  Transformers  {transformers.__version__}")

    entries = list_registry()
    table = Table(title="Available model specs")
    table.add_column("Key", style="cyan")
    table.add_column("Family")
    table.add_column("Size")
    table.add_column("HF Model ID")
    table.add_column("Transcoder Repo")
    table.add_column("Transcoder Type")
    for key, s in entries:
        table.add_row(key, s.family, s.size, s.hf_model_id, s.transcoder_repo, s.transcoder_type)
    rprint(table)


@app.command("transcoder-inspect")
def transcoder_inspect(
    size: str | None = typer.Option(None, help="Registry key, e.g. 'qwen3-0.6b' or 'qwen3-4b'"),
    repo: str | None = typer.Option(None, help="Direct HF transcoder repo id"),
):
    """Inspect a transcoder's config (lightweight -- does not download weights)."""
    import yaml
    from huggingface_hub import hf_hub_download

    from llm_circuits.transcoders.registry import get_spec

    if size is None and repo is None:
        rprint("[red]Provide --size or --repo[/red]")
        raise typer.Exit(1)

    repo_id = repo if repo else get_spec(size).transcoder_repo  # type: ignore[arg-type]

    rprint(f"Inspecting config for [bold]{repo_id}[/bold] ...")

    try:
        config_path = hf_hub_download(repo_id=repo_id, filename="config.yaml")
    except Exception:
        try:
            config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
        except Exception as exc:
            rprint(f"[red]Could not download config: {exc}[/red]")
            raise typer.Exit(1) from exc

    if config_path.endswith(".yaml") or config_path.endswith(".yml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        import json

        with open(config_path) as f:
            config = json.load(f)

    table = Table(title=f"Config: {repo_id}")
    table.add_column("Key", style="cyan")
    table.add_column("Value")
    for k, v in config.items() if isinstance(config, dict) else [("value", config)]:
        table.add_row(str(k), str(v))
    rprint(table)


@app.command("transcoder-load")
def transcoder_load(
    size: str | None = typer.Option(None, help="Registry key, e.g. 'qwen3-0.6b' or 'qwen3-4b'"),
    repo: str | None = typer.Option(None, help="Direct HF transcoder repo id"),
    device: str | None = typer.Option(None, help="Device, e.g. 'cpu', 'cuda'"),
    cache_dir: str | None = typer.Option(None, help="Local cache directory for transcoders"),
):
    """Load transcoders (downloads weights) and print basic info."""
    from llm_circuits.transcoders.circuit_tracer_loader import load_transcoder

    if size is None and repo is None:
        rprint("[red]Provide --size or --repo[/red]")
        raise typer.Exit(1)

    spec_or_repo = size if size else repo  # type: ignore[assignment]
    result = load_transcoder(spec_or_repo, device=device, cache_dir=cache_dir)  # type: ignore[arg-type]

    rprint(f"[green]Loaded transcoders from {result.repo_id}[/green]")
    rprint(f"  Type: {type(result.transcoder).__name__}")

    attrs = [a for a in dir(result.transcoder) if not a.startswith("_")]
    rprint(f"  Attributes: {attrs[:20]}")

    if result.config:
        rprint("  Config keys:", list(result.config.keys())[:20])


@app.command("transcoder-cache")
def transcoder_cache(
    repo: str = typer.Option(..., help="HF transcoder repo id"),
    cache_dir: str | None = typer.Option(
        None, help="Local directory to cache into (default: .cache/transcoders)"
    ),
    dtype: str | None = typer.Option(
        None,
        help="Dtype of the cached safetensors: 'bf16', 'fp16', or 'fp32' (default: "
        "circuit-tracer's fp32). The mwhanna Qwen3 repos store bf16, so '--dtype bf16' "
        "halves the on-disk cache losslessly (CLAUDE.md perf notes).",
    ),
):
    """Cache transcoder weights to a local directory."""
    import torch

    from llm_circuits.transcoders.circuit_tracer_loader import cache_transcoder

    torch_dtype = None
    if dtype is not None:
        dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        if dtype not in dtypes:
            rprint(f"[red]Unknown --dtype {dtype!r}; use one of {sorted(dtypes)}[/red]")
            raise typer.Exit(1)
        torch_dtype = dtypes[dtype]

    cache_transcoder(repo, cache_dir, dtype=torch_dtype)
    from llm_circuits.settings import transcoder_cache_dir

    resolved = cache_dir if cache_dir is not None else str(transcoder_cache_dir())
    rprint(f"[green]Cached {repo} -> {resolved}[/green]")


@app.command()
def generate(
    key: str = typer.Option(
        "qwen3-0.6b",
        "--key",
        "--size",
        help="Registry key (e.g. 'qwen3-0.6b', 'qwen3-4b') or bare size for Qwen3 (e.g. '0.6b')",
    ),
    prompt: str = typer.Option("Hello, world!", help="Prompt text"),
    max_new_tokens: int = typer.Option(64, help="Max tokens to generate"),
    cache_dir: str | None = typer.Option(None, help="Local cache directory for model weights"),
):
    """Generate text with a model using Transformers (no circuit-tracer)."""
    from llm_circuits.models.hf_loader import load_model_and_tokenizer
    from llm_circuits.transcoders.registry import get_spec

    spec = get_spec(key)
    model, tokenizer = load_model_and_tokenizer(spec.hf_model_id, cache_dir=cache_dir)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    rprint(f"[bold]Prompt:[/bold]  {prompt}")
    rprint(f"[bold]Output:[/bold]  {text}")


@app.command()
def serve(
    port: int = typer.Option(8000, help="Port to bind"),
    host: str = typer.Option("127.0.0.1", help="Bind address (keep localhost; forward the port)"),
    mock: bool = typer.Option(False, "--mock", help="CPU mock engine (no model; for UI dev)"),
):
    """Launch the interactive attribution-graph + steering UI (FastAPI).

    Loads a Qwen3 model + transcoders on demand from the browser and does live
    graph building, re-pruning, steering, and end-layer sweeps. Run on a GPU
    node (VS Code auto-forwards the port), or with --mock anywhere.
    """
    import os

    try:
        import uvicorn
    except ImportError as exc:  # serve extras not installed
        rprint("[red]serve dependencies missing — install with: uv sync --group serve[/red]")
        raise typer.Exit(1) from exc

    if mock:
        os.environ["LLM_CIRCUITS_SERVE_MOCK"] = "1"
    rprint(f"[bold]llm-circuits serve[/bold] on http://{host}:{port}  (mock={mock})")
    uvicorn.run("llm_circuits.serve.app:app", host=host, port=port)
