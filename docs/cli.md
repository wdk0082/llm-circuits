# CLI reference

Installing the package exposes the `llm-circuits` command (a
[Typer](https://typer.tiangolo.com/) app). Run everything through `uv`:

```bash
uv run llm-circuits --help
```

Registry keys are family-prefixed (`qwen3-0.6b`, `qwen3-4b`, …); bare Qwen3
sizes (`0.6b`) still resolve for backward compatibility.

## `info`

Print the environment (versions, device) and the table of available model
specs.

```bash
uv run llm-circuits info
```

## `transcoder-inspect`

Inspect a transcoder's config **without downloading the weights** — it fetches
only `config.yaml`/`config.json` from the repo.

```bash
uv run llm-circuits transcoder-inspect --size qwen3-0.6b
uv run llm-circuits transcoder-inspect --repo mwhanna/qwen3-4b-transcoders
```

Option | Meaning
------ | -------
`--size` | registry key (`qwen3-0.6b`, …)
`--repo` | a direct HF transcoder repo id (overrides `--size`)

## `transcoder-load`

Download (or use cached) transcoders and print basic info — the type, its public
attributes, and config keys. Useful for a quick connectivity/sanity check.

```bash
uv run llm-circuits transcoder-load --size qwen3-0.6b
uv run llm-circuits transcoder-load --repo mwhanna/qwen3-4b-transcoders --device cpu
```

Option | Meaning
------ | -------
`--size` / `--repo` | registry key or direct repo id
`--device` | `cpu`, `cuda`, …
`--cache-dir` | local cache directory for the transcoders

## `transcoder-cache`

Cache transcoder weights to a local directory as safetensors.

```bash
uv run llm-circuits transcoder-cache \
    --repo mwhanna/qwen3-0.6b-transcoders-lowl0 \
    --cache-dir .cache/transcoders \
    --dtype bf16
```

Option | Meaning
------ | -------
`--repo` | HF transcoder repo id (required)
`--cache-dir` | destination (default `.cache/transcoders`)
`--dtype` | `bf16` / `fp16` / `fp32`

```{admonition} Cache in bf16
:class: tip
The mwhanna Qwen3 repos store bf16 weights. Passing `--dtype bf16` halves the
on-disk cache versus circuit-tracer's fp32 default with **no** precision loss
(fp32 caching just upcasts the bf16 source).
```

## `generate`

Generate text with a model using plain Transformers (no circuit-tracer). Handy
for confirming a model loads and behaves.

```bash
uv run llm-circuits generate --key qwen3-0.6b --prompt "The capital of France is"
```

Option | Meaning
------ | -------
`--key` / `--size` | registry key or bare Qwen3 size
`--prompt` | the prompt text
`--max-new-tokens` | tokens to generate (default 64)
`--cache-dir` | local cache directory for model weights

## `serve`

Launch the interactive attribution-graph + steering UI (FastAPI). It loads a
Qwen3 model and transcoders on demand from the browser and does **live** graph
building, re-pruning, steering, and end-layer sweeps — the interactive version
of the {doc}`demo <guide/demo>`.

```bash
uv run --group serve llm-circuits serve          # GPU node; http://localhost:8000
uv run --group serve llm-circuits serve --mock   # CPU mock engine, no model (UI dev)
```

Option | Meaning
------ | -------
`--port` | port to bind (default 8000)
`--host` | bind address (default `127.0.0.1` — keep localhost and forward the port)
`--mock` | CPU mock engine, no model loaded

```{admonition} The --group serve is needed at run time
:class: note
Plain `uv run llm-circuits serve` re-syncs the environment **without** the serve
extras and the import fails. On the HPC, `bash hpc/run_interactive_server.sh`
runs it inside a GPU allocation; VS Code auto-forwards the port.
```
