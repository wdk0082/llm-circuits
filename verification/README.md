# verification/

Cross-checks of our from-scratch implementations against **circuit-tracer** itself.

These scripts are intentionally kept **outside `src/`**: the package
(`src/llm_circuits/`) uses circuit-tracer *only* to load transcoders and never imports
`ReplacementModel` / `AttributionGraph` / its intervention machinery. The scripts here
*do* import that machinery — purely to confirm our reimplementations match circuit-tracer
numerically. Nothing in `src/` depends on this folder.

## Scripts

- **`verify_intervention.py`** — runs our `run_feature_intervention(mode="steering-base-model")`
  and circuit-tracer's `ReplacementModel.feature_intervention` on the same Qwen3 model,
  the same transcoders, the same prompt/feature/M, and a matching constrained layer range,
  then compares the steering effect on the logits. Because the two use different model
  implementations (HF vs TransformerLens), it compares the **steering delta**
  (steered − clean), which cancels the baseline implementation offset, and PASSes when the
  deltas agree to within a few× the clean-logit noise floor (and cosine > 0.99).

Run on a GPU node (loads two ~4B models):

```bash
set -a; source .env; set +a
uv run python verification/verify_intervention.py
```
