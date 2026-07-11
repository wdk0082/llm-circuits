# HANDOFF — resume here (written 2026-07-11, old studio decommissioned)

**Read this first.** The previous studio's disk did NOT persist: everything outside
git is gone — `.env`, HF caches (~19 GB), the feature-label cache (~42 GB), the
graph dumps under `artifacts/supernode_inputs/`, review HTMLs, and the agent's
memory/plan files. **This file + the repo are the entire context carrier.**
Delete or rewrite this file once consumed.

## TL;DR

Branch `repro/constrained-supernodes`. The multilingual supernode **selection is
final and committed** (`notebooks/supernodes/multilingual_4b.json`, 16 supernodes,
`approved: true`, selection = explorer-export, user-reviewed 2026-07-11; the raw
export JSONs live in `notebooks/supernodes/exports/` as the review record).
**Next task = phase B**: rewire `notebooks/multilingual.ipynb` to constrained-only
interventions driven by that file, re-execute, clean up, push. Addition is already
v2-done (`addition_4b.json` approved; user wants an export-workflow revisit LATER).

## 0. New-studio bootstrap

```bash
git clone <repo> && cd llm-circuits && git checkout repro/constrained-supernodes
cp .env.example .env        # then fill HF_TOKEN (the old .env died with the studio)
uv sync
set -a; source .env; set +a # required before ANY bash command (CLAUDE.md rule)
```

First model load re-downloads Qwen3-4B + mwhanna transcoders (~19 GB; the
transcoder disk cache uses bf16 via `cache_transcoder(..., dtype=torch.bfloat16)`).
The feature-label cache re-downloads lazily per layer — only needed when BUILDING
graphs with labels; the dumps below already embed labels+examples.

## 1. Restore the input dumps (pick A, else B)

The notebooks and review pages load pruned-graph dumps from
`artifacts/supernode_inputs/4b/` (git-ignored except manifest/polymer JSONs).

**A. Tarball on the orphan branch `repro-dumps-4b`** (pushed 2026-07-11; 121 MB in
two <100 MB parts; delete the branch when the dumps are obsolete):

```bash
git fetch origin repro-dumps-4b
git cat-file blob origin/repro-dumps-4b:si4b.tar.gz.part-aa >  /tmp/si4b.tar.gz
git cat-file blob origin/repro-dumps-4b:si4b.tar.gz.part-ab >> /tmp/si4b.tar.gz
tar -xzf /tmp/si4b.tar.gz -C artifacts/   # -> artifacts/supernode_inputs/4b/
```

bf16 is bit-stable across A100 nodes, so A ≡ B in content.

**B. Rebuild (~35 min on an A100-80GB, after HF caches warm):**

```bash
# addition (0.8) + chat multilingual (0.8):
uv run python notebooks/build_supernode_inputs.py --size 4b \
  --graphs antonym_en antonym_fr antonym_zh synonym_en hot_en
# raw multilingual at 0.95 (they starve at 0.8 — no quote-position nodes):
uv run python notebooks/build_supernode_inputs.py --size 4b --skip-addition \
  --graphs raw_antonym_en raw_antonym_fr raw_antonym_zh raw_synonym_en raw_hot_en \
  --node-threshold 0.95
```

Then `uv run python notebooks/build_supernodes.py --size 4b` re-emits the
`review_chat_*` / `review_raw_*` pages (serves via
`python3 -m http.server 8000 --bind 0.0.0.0 --directory artifacts`), and
`--from-exports notebooks/supernodes/exports/groups_*.json` must reproduce
`multilingual_4b.json` (only `built_from` sha may differ). Run both as a sanity
check after a rebuild.

## 2. What is final (do not redo)

- **Intervention method**: constrained patching bit-equivalent to circuit-tracer,
  3/3 parity incl. `value=` donor injection (`verification/verify_intervention.py`).
  Convention: `m = M_paper − 1` for suppressions (paper −5× ⇒ m=−6); donors use
  `value = mult × stored act`.
- **Multilingual selection** (`multilingual_4b.json`): 16 supernodes — multilingual
  `antonym(7, source) / synonym(8, donor) / small(6, source) / hot(6, donor) /
  say large(7) / say small(7) / say cold(6)` (readouts) + lang-specific
  `say large (en/fr/zh)`, `opposite (en/fr/zh)` (input/evidence), `quote (en/fr/zh)`
  (language-swap source+donor). Unions span chat+raw pages; every member carries
  per-graph `acts` and `positions` dicts. `large/cold (multilingual)`, `say hot`:
  admitted absent. Globally disjoint by (layer, feature) — the loader enforces it.
- **Addition v2**: `addition.ipynb` executed clean; `addition_4b.json` approved
  (grid-evidence selection). Verdicts in its Summary + DEVLOG fifth session.
- **Dumps plan**: chat graphs 0.8, raw graphs 0.95 (recorded per-graph in the
  committed `artifacts/supernode_inputs/4b/manifest.json`).

## 3. Phase B — the task (multilingual notebook rewiring)

Rewire `notebooks/multilingual.ipynb` (currently the fourth-session hybrid:
propagate + constrained cell pairs, in-cell supernode selection) to **load
`multilingual_4b.json` and run constrained patching only**. Use `addition.ipynb`
as the template for structure/prose (that cleanup style is what the user wants).

1. **Loader**: port `load_supernodes` from `addition_helper.py` (or hoist shared);
   it must refuse unapproved/overlapping files. Members give `(layer, feature)`,
   per-graph `acts`/`positions`.
2. **Experiments** (paper multiples, digest §B; per-experiment
   `sweep_patch_end_layer`, state ℓ per result; readouts via
   `supernode_readout_pct` with rows ≤ ℓ pinned/dropped):
   - *Operation swap* (per language, chat + raw variants): suppress
     `antonym (multilingual)` at −5× (m=−6) **at each member's own node position
     in the recipient graph** (fall back to final position when the member has no
     position there) + inject `synonym (multilingual)` donors at
     `value = +6× act[donor graph]` at the recipient's operation-word token span
     (compute via `op_word_positions()` in `build_supernodes.py`; words:
     opposite/contraire/反义词). Readout: `say small` vs `say large`; paper
     endpoint: language-appropriate synonym top-1, crossover ≈4× (optional ladder
     0–6×).
   - *Operand swap*: suppress `small (multilingual)` at −0.5× (m=−1.5) at operand
     positions + inject `hot (multilingual)` at `value = +1.5× act` at the operand
     token. Readout: `say cold` up, `say large` down; paper endpoint cold/froid/冷.
   - *Language swap* (raw pages only — quote nodes exist only there): original
     language `quote (X)` at −5× (m=−6), new language `quote (Y)` injected
     `value = +6× act[its raw graph]` at the final open-quote position. Readout:
     `say large (multilingual)` roughly preserved, lang-specific say-large flips.
3. **Graphs/figures cells** load dumps from `artifacts/supernode_inputs/4b`
   (SN_INPUTS pattern in `addition.ipynb`); annotated explorer HTMLs get groups
   from the supernode file. Non-intervention sections (overlap §H, scale, default
   language, introspection, corpus) stay as-is.
4. Merge each propagate→constrained cell pair into one constrained cell (pairs at
   fourth-session indices 9→10, 12→13, 15→16, 24→25, 26→27, 28→29 — re-locate by
   content). Delete in-cell selection code (`SAY_BIG`, detection scans, etc.).
5. Prose: rewrite headers/at-a-glance tables (supernodes come from the reviewed
   file), Summary as v2 verdicts vs the paper. All continuity numbers reset.
6. Re-execute end-to-end (~15 min GPU with eager decoders), zero cell errors.
7. `git rm` stale propagate-era multilingual artifacts under
   `artifacts/paper_multilingual/4b/` (anything the re-executed notebook no longer
   writes — diff dir listing vs notebook writes).
8. DEVLOG session section; CI trio (`uv run ruff check`, `uv run ruff format
   --check`, `uv run pytest`); push directly (standing instruction).

Afterwards (user-stated intent): revisit ADDITION supernodes via the same
export-review workflow; consider raw-primary vs chat-primary framing (paper is
raw-only; raw synonym caveat: FR echoes pet(it), ZH 小/微 near-tie).

## 4. Gotchas

- Always `set -a; source .env; set +a` first; always `uv run`. Heavy jobs on the
  GPU node, never login nodes (CLAUDE.md).
- Intervention notebooks need **eager decoders** (`lazy_decoder=False`, ~57 GB
  transcoders + 8 GB model on an A100-80GB); encode-only work (overlap §H) keeps
  lazy.
- **Editor-tab autosave clobbers**: files open in the user's IDE can silently
  revert edits — ask the user to reload tabs before editing notebooks/supernode
  JSONs (it has eaten committed work twice).
- gitignore carve-out: `artifacts/**` ignored EXCEPT `*.json/*.png/*.npy`
  (graph_*.json re-excluded; note `.npz` and `.html` are NOT kept — grids.npz and
  review pages are disk-only regenerables).
- Review pages embed groups; "Export groups" downloads `groups_<graph>.json` keyed
  by the unprefixed dump name (`antonym_en` = chat). Ingestion enforces global
  disjointness and unknown group names ingest as role `custom`.
- The A100 node auto-stops after ~1 h idle: commit+push work units without asking.
