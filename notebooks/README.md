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
