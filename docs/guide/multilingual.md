# The multilingual reproduction

`notebooks/multilingual.ipynb` is the paper-exact reproduction of **multilingual
antonym circuits on Qwen3-4B**. Where the {doc}`demo <demo>` shows the machinery
on one prompt, this notebook uses it for a full research result: it builds
attribution graphs for the antonym task in English, French, and Chinese, groups
features into **supernodes**, and runs feature **swaps** to test whether the
circuit is language-agnostic.

```{admonition} The rendered notebook is pre-run
:class: note
The page linked below shows the **stored outputs** from the reviewed run — the
docs build never executes the notebook. To re-run it yourself you need a GPU
(Qwen3-4B with eagerly-loaded transcoders — see {doc}`../installation`) and the
reviewed graph dumps in `artifacts/supernode_inputs/4b/`. Launch it with
`uv run --group notebook jupyter lab notebooks/multilingual.ipynb`.
```

## What the notebook covers

The notebook is organised to mirror the paper's figures:

- **0 · Behavior** — the antonym task in EN/FR/ZH and the swap targets, in both
  chat and raw formats.
- **A · Attribution graphs** — loaded from the reviewed dumps.
- **B · Supernodes** — the two reviewed feature groupings (the paper's node
  classes).
- **C · Operation swap** — antonym → synonym.
- **D · Operand swap** — small → hot.
- **E · Language swap** — X → Y (raw pages).
- **F · Cross-lingual feature overlap** by layer.
- **G · Which language is the mechanistic default?**
- **H · Scale comparison** — 4b vs. 8b feature overlap.
- **Summary** — the verdicts against the paper.

The reproduction methodology — semantic supernode grouping, donor injection at
absolute activations, and coupled strength ladders — is task-specific and lives
in `notebooks/multilingual_helper.py` alongside the notebook, not in the package.
The package supplies the primitives it calls: transcoder loading, the local
replacement model, attribution and pruning, and the constrained-patching
interventions documented in the {doc}`API reference <../api/index>`.

## Read it

```{toctree}
:maxdepth: 2

/notebooks/multilingual
```

```{admonition} Verdicts and the write-up
:class: tip
The Summary cell records how each experiment landed against the paper. The full
narrative, including the operand/operation-swap variants, is in the repo's
`DEVLOG.md` and `DEVLOG_EXTRA.md`, and the companion notebooks
`multilingual_extra_operand_swap.ipynb` / `multilingual_extra_operation_swap.ipynb`.
```
