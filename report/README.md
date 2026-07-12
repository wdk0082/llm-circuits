# Report

LaTeX source for the project report (attribution graphs on Qwen3).

- **Build:** `make` (needs TeX Live; on this node: `texlive-latex-extra`,
  `texlive-fonts-recommended`, `texlive-bibtex-extra`, `latexmk` are installed).
- **Word count:** `make count` (texcount). **Budget: 4,000 words total** — the
  per-section split lives in the header comment of `main.tex`.
- **Status:** Abstract, Introduction, Method, and *Experiments and Results*
  (Section 3 — the multilingual case study only; addition was dropped from the
  report by decision) are written. *Discussion* (~350 w) is a placeholder with
  its planned narrative in a comment in situ.
- **TODO(screenshots):** three framed placeholders await screenshots — Fig.\ 1
  (main text: example reviewed features) and Appendix C (explorer views of the
  chat and raw antonym pages).
- Figures are generated from the committed artifacts (nothing schematic except
  box layouts); regenerate with the session's figure script if artifacts change.
- CJK glyphs (大/小/冷) need `latex-cjk-chinese` + the arphic `gbsn` font
  (Debian: `latex-cjk-chinese-arphic-gbsn00lp`) — installed on this node.
