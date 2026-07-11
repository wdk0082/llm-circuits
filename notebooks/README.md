# Notebooks — the paper reproductions

The biology-paper reproductions live here, **two files per behavior** (the executable
notebook + its task-specific helper module):

| Behavior | Notebook | Helper |
|---|---|---|
| Addition (biology.html §Addition + methods supplement) | `addition.ipynb` | `addition_helper.py` |
| Multilingual circuits (biology.html §Multilingual) | `multilingual.ipynb` | `multilingual_helper.py` |

Paper reference values are digested in `../notes/biology_digest.md`; per-experiment
verdicts vs the paper are in each notebook's Summary and in `../DEVLOG.md`. Artifacts
(explorer HTMLs, figures, result JSONs) land in `../artifacts/paper_{addition,multilingual}/<size>/`.

**Artifact naming.** Files with no prefix are the chat-format main sections; a `raw_*`
prefix (multilingual only) is section G's rerun of the *same* experiment in the paper's
raw open-quote completion format; a `*_constrained` suffix is the same experiment under
the paper's **constrained-patching** protocol (the chosen patch end layer ℓ is stored in
the JSON and printed in the cell) — the un-suffixed twin is the fully-propagating
robustness variant. Each notebook's "What actually runs" cell (right under the title)
tables the exact model inputs, formats, supernode-selection rules, and file inventory.

Helpers are deliberately kept out of `src/llm_circuits/` — they are task-specific (prompt
formats, operand-grid probes, swap-sweep protocols) and don't generalise.

Execute on a GPU (A100-80GB ≈ 14–18 min per notebook with the eager-decoder load and warm
caches — the constrained-patching ℓ-sweeps add ~1–2 min; a cold node first pays the
~45 GB feature-label download inside the first graph build, ~20 min extra. Install deps
with `uv sync --group notebook`):

```bash
cd notebooks
uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 addition.ipynb        # or multilingual.ipynb
```

or on Slurm: `sbatch hpc/run_addition_notebook.sbatch` / `hpc/run_multilingual_notebook.sbatch`.
