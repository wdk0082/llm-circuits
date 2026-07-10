# Report

LaTeX source for the project report (attribution graphs on Qwen3).

- **Build:** `make` (needs TeX Live; on this node: `texlive-latex-extra`,
  `texlive-fonts-recommended`, `texlive-bibtex-extra`, `latexmk` are installed).
- **Word count:** `make count` (texcount). **Budget: 4,000 words total** — the
  per-section split lives in the header comment of `main.tex`.
- **Status:** Abstract, Introduction (related work folded in), and Method
  (transcoders → attribution-graph generation → pruning → interventions) are
  written. *Experiments and Results* (~1,150 w) and *Discussion* (~350 w) are
  placeholders; their planned narratives are outlined in comments in situ
  (source material: `notebooks/`, `DEVLOG.md`, `notes/biology_digest.md`).
- The abstract's final sentence summarises results and carries a
  `TODO(results)` comment — re-check it once Section 3 is written.
