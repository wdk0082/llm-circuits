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

**Supernode pipeline (v2).** Selection lives outside the notebooks:
`build_supernode_inputs.py` (GPU, once per config) persists pruned graphs + operand
grids; `build_supernodes.py` (CPU) scans them and emits reviewable supernode files
under `notebooks/supernodes/` (evidence per member, `approved:false`); a human review
gate approves them; the notebooks load ONLY the approved file (`load_supernodes`
refuses anything else) and load the persisted graphs. Both notebooks run this pipeline
(constrained patching only). Addition's file is the grid-evidenced scan (delegated
review); multilingual's is **export-driven** — groups hand-adjusted on the explorer
review pages, "Export groups" JSONs ingested verbatim via `--from-exports`
(user-reviewed 2026-07-11; the exports are committed under `supernodes/exports/`).

**Artifact naming.** Constrained patching is the only protocol; the `*_constrained`
suffix is kept for continuity, and each JSON stores the chosen patch end layer ℓ plus
its decision curve per job. Multilingual swap files key each job by
`{chat,raw}_{en,fr,zh}` — the paper's raw open-quote format and our chat adaptation run
side by side in the same cell (the language swap is raw-only: its quote supernodes
exist only on the raw pages). Each notebook's "What actually runs" cell (right under
the title) tables the exact model inputs, formats, supernode files, and file inventory.

Helpers are deliberately kept out of `src/llm_circuits/` — they are task-specific (prompt
formats, operand-grid probes, swap-sweep protocols) and don't generalise.

Execute on a GPU (A100-80GB ≈ 15–30 min per notebook with the eager-decoder load and warm
caches. With the `supernode_inputs` dumps present the notebooks never build graphs, so the
~45 GB feature-label cache is NOT needed; a cold node pays only the model+transcoder
download (~19 GB for the 4b, more for the multilingual §H 8b pair). Install deps with
`uv sync --group notebook`):

```bash
cd notebooks
uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 addition.ipynb        # or multilingual.ipynb
```

or on Slurm: `sbatch hpc/run_addition_notebook.sbatch` / `hpc/run_multilingual_notebook.sbatch`.
