# Installation

`llm-circuits` targets **Python 3.11** and is managed with
[`uv`](https://docs.astral.sh/uv/). All commands below run through `uv` so they
use the locked, reproducible environment.

## Install

```bash
git clone https://github.com/wdk0082/llm-circuits.git
cd llm-circuits
uv sync --all-groups      # or: make install
```

`uv sync` creates a `.venv/` and installs the package (editable) with its
dependencies, including
[`circuit-tracer`](https://github.com/safety-research/circuit-tracer) pinned as a
git dependency. `--all-groups` also pulls the optional groups:

Group | What it adds | When you need it
----- | ------------ | ----------------
`dev` | ruff, pytest, mypy, pre-commit | development, CI
`notebook` | jupyterlab, ipykernel, datasets, mplfonts | running the reproduction notebooks
`serve` | fastapi, uvicorn, httpx | the interactive UI (`llm-circuits serve`)
`docs` | sphinx, furo, myst-nb | building this documentation

To install just what you need, pass individual groups, e.g.
`uv sync --group notebook`.

```{admonition} Run everything through uv
:class: note
The examples in these docs write `uv run …`. Plain `python` or a bare
`llm-circuits` will miss the project environment. When a command needs an
optional group at *run* time — the UI, for instance — pass it explicitly:
`uv run --group serve llm-circuits serve`.
```

## Configuration

Copy the example environment file and fill in your values:

```bash
cp .env.example .env
```

- **`HF_TOKEN`** — required to download gated models and the transcoder repos
  from the Hugging Face Hub.
- **`LLM_CIRCUITS_DEVICE`** — overrides device auto-detection. By default
  {func}`~llm_circuits.settings.default_device` picks `cuda`, then `mps`, then
  `cpu`; the value `tpu` resolves to a torch_xla device (TPU VM only). dtype
  follows from the device via {func}`~llm_circuits.settings.default_dtype`
  (bfloat16 on CUDA/CPU/TPU, float32 on MPS).

## Hardware

The attribution and intervention code runs on the model and its transcoders, so
a **GPU is strongly recommended** for anything beyond the smallest size. The two
worked examples default to **Qwen3-4B** with eagerly-loaded transcoders
(~57 GB of transcoder weights on top of the ~8 GB model; graph-build peak
~68 GiB). On a smaller card, use a smaller `--size` or the `--lazy-decoder`
escape hatch (much slower — see the note in {doc}`guide/demo`).

## Verify the install

```bash
uv run llm-circuits info                              # device + model registry
uv run llm-circuits transcoder-inspect --size qwen3-0.6b   # no large downloads
uv run pytest                                         # the test suite
```

## Building the documentation

```bash
make docs            # -> docs/_build/html/index.html
# equivalently:
uv run --group docs sphinx-build -b html docs docs/_build/html
```

The build imports the package for autodoc and copies
`notebooks/multilingual.ipynb` into the docs tree to render it (notebooks are
**not** executed during the build — the stored outputs are shown). Use
`make docs-clean` to remove `docs/_build/` and the generated notebook copy.
