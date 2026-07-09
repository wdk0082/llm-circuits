# Attribution-Graphs Methods Digest — reference notes for faithfulness verification

Sources:
- **Paper**: "Circuit Tracing: Revealing Computational Graphs in Language Models" (Ameisen, Lindsey et al., 2025), https://transformer-circuits.pub/2025/attribution-graphs/methods.html. Quotes below are verbatim (HTML→text; note the stripper ate `<` inside LaTeX, restored here as `ℓ < ℓ_t`).
- **Reference impl**: circuit-tracer pinned **v0.3.1** = `git+https://github.com/safety-research/circuit-tracer.git@v0.3.1` (commit `e09b5f36`, package metadata self-reports "0.1.0"). All `file:line` cites below are relative to
  `/Users/wdk0082/Projects/llm-circuits/.venv/lib/python3.11/site-packages/circuit_tracer/`.
  Two backends exist (`attribution/attribute_transformerlens.py`, `attribution/attribute_nnsight.py`); they implement the same algorithm. TransformerLens backend is cited; nnsight mirrors it.

Notation (paper's): `x^ℓ` residual stream at layer ℓ; `a^ℓ` CLT feature activations at layer ℓ; `y^ℓ` true MLP output at layer ℓ; `ŷ^ℓ` CLT reconstruction; `W_enc^ℓ`, `W_dec^{ℓ'→ℓ}` encoder/decoder matrices; `J^▼_{c_s,ℓ_s→c_t,ℓ_t}` = Jacobian of the underlying model **with stop-gradient on all nonlinearities** (MLP outputs, attention patterns, normalization denominators), from residual at (position c_t, layer ℓ_t) back to residual at (c_s, ℓ_s).

---

## 1. Local replacement model construction

### 1.1 Cross-layer transcoder (CLT) — paper
- Encoder: `a^ℓ = JumpReLU(W_enc^ℓ x^ℓ)` — features at layer ℓ read the residual stream at layer ℓ.
- Decoder: `ŷ^ℓ = Σ_{ℓ'=1}^{ℓ} W_dec^{ℓ'→ℓ} a^{ℓ'}` — a layer-ℓ' feature writes to MLP outputs of layers ℓ', ℓ'+1, …, L with a **separate decoder matrix per output layer**.
- Training losses (context only): `L_MSE = Σ_ℓ ||ŷ^ℓ − y^ℓ||²`; Tanh sparsity `L_sparsity = λ Σ_ℓ Σ_i tanh(c · ||W_dec,i^ℓ|| · a_i^ℓ)`; JumpReLU with STE (bandwidth 1.0, init threshold 0.03); residual-stream inputs and MLP-output targets are **per-layer normalized during training** (Appendix "ML details").

### 1.2 circuit-tracer transcoder modules
- `transcoder/cross_layer_transcoder.py` — `CrossLayerTranscoder`: `W_enc [n_layers, d_tc, d_model]`; `W_dec` = ParameterList, `W_dec[i]` has shape `[d_tc, n_layers − i, d_model]` (feature at layer i writes to layers i…L−1); `b_enc [n_layers, d_tc]`; `b_dec [n_layers, d_model]`; optional `W_skip [n_layers, d_model, d_model]` (:92–125).
  - `encode_layer` (:176–182): `feats = x @ W_enc[ℓ].T + b_enc[ℓ]`, then activation.
  - JumpReLU application (:168–174): `mask = features > threshold[layer]; features * mask` — **strict >**, threshold shape `[n_layers, 1, d_tc]`. Same semantics in `activation_functions.py:11–34` (`x * (x > threshold)`).
  - Reconstruction (:284–301): `recon[ℓ] = Σ_active a_s W_dec + b_dec[ℓ] (+ x^ℓ @ W_skip[ℓ] if skip)` — **b_dec and skip are part of the reconstruction**, therefore NOT part of the error.
- `transcoder/single_layer_transcoder.py` — `SingleLayerTranscoder` (PLT): `encode` (:120–125) `pre = W_enc x + b_enc`; `decode` (:127–135) `acts @ W_dec + b_dec (+ skip)`; `compute_skip` (:137–141) is `x @ W_skip.T` (note the transpose vs CLT's `x @ W_skip[ℓ]`, cross_layer_transcoder.py:307–311). `TranscoderSet` wraps one PLT per layer (:228+); every layer 0…max must have one (:254).
- Hook points where features read/write are configurable per transcoder set (`feature_input_hook`, e.g. `hook_resid_mid` or `ln2.hook_normalized`; `feature_output_hook`, e.g. `hook_mlp_out`); defaults `hook_resid_mid`/`hook_mlp_out` (cross_layer_transcoder.py:59–60); actual values come from the HF repo config (utils/hf_utils.py:151–152).

### 1.3 Replacement model / local replacement model — paper definitions (verbatim)
- Replacement model: "substitutes the cross-layer transcoder features for the model's MLP neurons – that is, where each layer's MLP output is replaced by its reconstruction by all CLTs that write to that layer. … Attention layers are applied as usual, without any freezing or modification."
- Local replacement model (for a fixed prompt p): "Substitutes the CLT for the MLP layers…; Uses the attention patterns and normalization denominators from the underlying model's forward pass on p…; Adds an error adjustment to the CLT output at each (token position, layer) pair equal to the difference between the true MLP output on p and the CLT output on p."
- "After this error adjustment and freezing of attention and normalization nonlinearities … all of the error-corrected replacement model's activations and logit outputs exactly match those of the underlying model."
- "Its weights are the summed interactions over all the linear paths from one feature to another, including via the residual stream and through attention, but not passing through MLP or CLT layers." Error terms are "bias-like nodes … with a connection from each bias to each downstream neuron." "The only nonlinearities in the local replacement model are those applied to feature preactivations."

### 1.4 circuit-tracer realization (key trick: real forward, replaced backward)
`replacement_model/replacement_model_transformerlens.py`:
- Model loaded with `fold_ln=False, center_writing_weights=False, center_unembed=False` (:111–117). MLPs wrapped in `ReplacementMLP` adding `hook_in`/`hook_out` (:29–41); unembed wrapped adding `hook_pre`/`hook_post` (:44–64).
- `_configure_gradient_flow` (:185–209):
  - Permanent `detach()` hooks on `attn.hook_pattern`, `ln1.hook_scale`, `ln2.hook_scale`, `ln1_post/ln2_post.hook_scale` (if present), `ln_final.hook_scale` → **attention patterns and LN/RMSNorm denominators are frozen in the backward pass** (forward unchanged). LN scale detach makes LN a linear map `x ↦ (x−μ)/σ_detached * γ + β` (or RMSNorm `x/σ_detached * γ`).
  - All parameters `requires_grad=False`; `hook_embed` activations get `requires_grad=True` (root of autograd graph).
- `_configure_skip_connection` (:211–246): at each layer's feature-output hook, output is replaced by `grad_hook(skip + (acts − skip).detach())` where `skip = W_skip x` (or 0 if no skip). **Forward value is numerically unchanged (= true MLP output); backward flows only through the transcoder skip connection.** A new `HookPoint hook_out_grad` is appended so backward hooks can read ∂/∂(MLP-out); `model.feature_output_hook = <original> + ".hook_out_grad"` (:172).
- Consequence: the attribution forward pass **is the underlying model** (MLPs run normally). The "local replacement model" exists purely in the gradient structure: gradients flow through residual stream + frozen-pattern attention (OV only) + frozen-denominator norms + W_skip, and are blocked at MLP outputs, QK/pattern computation, and norm denominators. Error nodes need no forward injection because forward is exact.
- **Error nodes** — `setup_attribution` (:405–452): one clean cached forward (`run_with_hooks`, no_grad) caches feature-input and feature-output activations; then
  `error_vectors = mlp_out_cache − reconstruction` (:438), shape `(n_layers, n_pos, d_model)`, where reconstruction = decoder sum + b_dec + skip (§1.2). `error_vectors[:, 0] = 0` (:440). `token_vectors = W_E[tokens].detach()` (:441).
- **BOS / position-0 handling** (implementation choice; paper is silent):
  - `ensure_tokenized` (:348–403) prepends a special token (BOS/PAD/EOS, first non-None) if the prompt doesn't already start with one — rationale in docstring: position 0 has anomalously high norm/feature count; "This prepended token is later ignored during attribution analysis."
  - Feature activations at position 0 are zeroed: `encode_sparse(..., zero_positions=slice(0,1))` (cross_layer_transcoder.py:184–219, single_layer_transcoder.py:151–172) and in activation-caching hooks `transcoder_acts[0] = 0` (replacement_model_transformerlens.py:291). Error vectors at position 0 zeroed (:440). So **position-0 feature and error nodes are excluded (zero)**; the position-0 embedding node still exists as a source column.

---

## 2. Node types

Paper ("Constructing an Attribution Graph"): four types —
1. **Output (logit) nodes**: "candidate output tokens. We only construct output nodes for the tokens required to reach 95% of the probability mass, up to a total of 10." Input vector `v_in = ∇(logit_tok − mean logit)` — i.e. **the unembedding column demeaned across the vocabulary**. No output edges.
2. **Intermediate (feature) nodes**: "active cross-layer transcoder features at each prompt token position." Input vector `v_in^ℓ = W_enc,s^{ℓ_s}` (its encoder row); output vectors `v_out^ℓ = W_dec,s^{ℓ_s→ℓ}` for ℓ_s ≤ ℓ.
3. **Primary input (embedding) nodes**: "the embeddings of the prompt tokens", `v_out = Emb_tok`.
4. **Error nodes**: `v_out = MLP_ℓ(x_{c,ℓ}) − CLT_ℓ(x_c)`; "error nodes have no input edges."

circuit-tracer:
- **Node ordering in the adjacency matrix** (graph.py:37–63): `[active_features(0..n−1), error(layer0,pos0), error(layer0,pos1), …, error(L−1, n_pos−1), tokens(0..n_pos−1), logits(top1..topK)]` — errors are **layer-major then position**; row/col index arithmetic: `error_offset(ℓ) = n_feat + ℓ·n_pos` (context_transformerlens.py:135–136), `tok_start = n_feat + L·n_pos`, `logit_offset = n_feat + (L+1)·n_pos` (attribute_transformerlens.py:163).
- `active_features` tensor `(n_active, 3)` of `(layer, pos, feature_idx)`; `activation_values` aligned.
- **Logit selection** — `utils/salient_logits.py:27–30`: `probs = softmax(logits[last_pos])`; take `topk(max_n_logits=10)`; `cutoff = searchsorted(cumsum(top_p), desired_logit_prob=0.95) + 1`; keep first `cutoff` (i.e. smallest set with cumulative prob ≥ 0.95, capped at 10). Logit nodes exist **only at the final position**. Probabilities are softmax of the real model's logits (from the `setup_attribution` pass — includes logit softcap if the model has one; `ctx.logits[0, −1]`, attribute_transformerlens.py:150–155).
- **Demeaning** (salient_logits.py:35–45): `demeaned_vecs = W_U[:, tok] − mean_over_vocab(W_U)` — matches paper's `∇(logit_tok − mean logit)`. `b_U` and softcap play no role in the injected gradient (`zero_softcap` exists at replacement_model_transformerlens.py:339–346 but is **never called** in v0.3.1; the unembed is never run during attribution).

---

## 3. Edge / adjacency computation

### 3.1 Paper formulas (verbatim, appendix "Attribution Graph Computation")
- Feature s → (feature or logit) t:
  `A_{s→t} = a_s w_{s→t} = a_s Σ_{ℓ_s ≤ ℓ < ℓ_t} (W_dec,s^{ℓ_s→ℓ})ᵀ J^▼_{c_s,ℓ→c_t,ℓ_t} W_enc,t^{ℓ_t}`
  (for logit targets, replace `W_enc,t` by the demeaned unembedding gradient at the final-layer residual).
- Embedding/error s → t: `A_{s→t} = v_out,sᵀ J^▼_{c_s,ℓ_s→c_t,ℓ_t} v_in,t` (e.g. `w_{s→t} = Emb_sᵀ J^▼ W_enc,t^{ℓ_t}`).
- `J^▼` expands as a sum over all paths of residual steps (identity) and attention steps (`a^{h}_{c_i→c_{i+1}} OV_h` with frozen pattern weight): `J^▼ = Σ_{p∈P} Π_i π_i`.
- Edges "originate from feature, embedding, and error nodes, and terminate at feature and output nodes."
- Linearity invariant: "the preactivation h_t of any feature node t is simply the sum of its incoming edges in the graph: `h_t = Σ_{S_t} w_{s→t}`" — with stop-grads, source-activation-scaled. (Caveat for verification: model bias terms — b_dec, attention output biases, LN biases — contribute constant terms that are not graph nodes, so in practice `Σ_s A_{s→t} = h_t − (bias-path contributions) − b_enc,t`. Only error terms are represented as bias-like nodes.)
- QK excluded: "these graphs do not contain information about the influence of nodes on other nodes via their influence on attention patterns, but do contain information about node-to-node influence through the outputs of frozen attention" (OV yes, QK no).
- Practical recipe (paper): iterate over **target** nodes; inject the target's input vector into the residual stream at its (layer, position) (encoder for features; demeaned-logit gradient at final residual for logits); backward pass with stop-grad on MLP outputs, frozen patterns and denominators; for a CLT source, "take the sum of the dot products of its decoder vector in each layer with the gradient in that layer, times the activation of that feature"; for embedding/error sources, dot product of their vector with the gradient. Cost linear in #active features; adaptive queue ordering by influence supported.

### 3.2 circuit-tracer implementation
Orientation: **`adjacency_matrix[target, source] = A_{s→t}`** — rows = targets, columns = sources (graph.py:51–55). Signed values; no normalization/absolute value at this stage.

Pipeline (`attribution/attribute_transformerlens.py`):
- Defaults (attribute.py:22–29): `max_n_logits=10`, `desired_logit_prob=0.95`, `batch_size=512`, `max_feature_nodes=None`, `update_interval=4`.
- Phase 0: `setup_attribution` → `AttributionContext` carrying: sparse `activation_matrix (L, n_pos, d_tc)`; `error_vectors`; `token_vectors`; `encoder_vecs` (`(n_active, d_model)` raw encoder rows); `decoder_vecs` — **pre-scaled by source activation**: `a_s · W_dec^{ℓ_s→ℓ}`, one row per (active feature, output layer) (cross_layer_transcoder.py:235–282, scaling at :260; PLT: single_layer_transcoder.py:174–201 scaling at :186); `encoder_to_decoder_map` maps each decoder row → its source-feature index (so CLT rows sum into one column); `decoder_locations = (layer_ids, pos_ids)` per decoder row.
- Phase 1 (forward): `model.forward(input_ids.expand(batch_size, −1), stop_at_layer=n_layers)`; caching hooks store the residual tensor at each layer's `feature_input_hook`, plus `ctx._resid_activations[−1] = model.ln_final(residual)` applied manually (attribute_transformerlens.py:135–138; context_transformerlens.py:77–91). The batch dimension (all rows = same prompt) lets one backward pass process `batch_size` target nodes.
- Phase 3/4 (backward per batch of targets) — `AttributionContext.compute_batch` (context_transformerlens.py:168–232):
  - Allocate buffer `(row_size = n_feat + (L+1)·n_pos, batch)`.
  - **Gradient injection**: per layer present in the batch, `tensor.register_hook` on the cached residual; the hook `grads.index_put_((batch_rows, positions), inject_values)` — it **replaces** (not adds) the gradient at each target's (row, position) with its input vector (encoder row for feature targets; demeaned unembed column for logit targets, injected at the post-`ln_final` tensor) (:199–219).
  - Seed: `resid_activations[max_layer].backward(gradient=zeros_like(...))` (:222–226) — the zero seed plus per-row replacement makes each batch row compute exactly `∂(v_in,t · x_{c_t,ℓ_t})/∂(...)` through the frozen-linear paths.
  - `retain_graph=True` for every backward except the final feature batch (attribute_transformerlens.py:221).
  - **Backward hooks contract gradients with source output vectors** (context_transformerlens.py:93–157), einsum `"batch position d_model, position d_model -> position batch"`:
    - Feature sources: at every layer ℓ's `feature_output_hook` (`…hook_mlp_out.hook_out_grad`), contract grad at the source position (`read_index = [:, nnz_positions]`) with the pre-scaled `a_s·W_dec^{ℓ_s→ℓ}` rows; `buffer[encoder_to_decoder_map] += …` **accumulates across output layers** → realizes `a_s Σ_ℓ (W_dec^{ℓ_s→ℓ})ᵀ (grad at ℓ, c_s)` (:123–132).
    - Error sources: contract grad at layer ℓ hook with `error_vectors[ℓ]` for all positions → rows `error_offset(ℓ):error_offset(ℓ+1)` (:138–145).
    - Embedding sources: contract grad at `hook_embed` with `token_vectors` → rows `tok_start:tok_start+n_pos` (:147–156).
  - Rows for logit targets computed first (Phase 3, :179–189), then feature targets (Phase 4).
- **Adaptive target ordering** (when `max_feature_nodes < n_active`): every `update_interval` batches, rank unvisited features by current influence estimate `compute_partial_influences(edge_matrix_so_far, logit_p, row_to_node_index)` (attribute_transformerlens.py:201–231) — power iteration `prod = prod[row_to_node_index] @ Â` with `Â = |A| row-normalized (clamp 1e−8)`, seeded with logit probabilities, accumulated up to 128 iters (graph.py:305–331). Unvisited features become columns only (their target rows never computed).
- Assembly (attribute_transformerlens.py:234–259): drop unselected feature columns; sort rows so features come in index order; final square matrix `(N, N)` with feature-target rows on top, logit-target rows at bottom; **error and embedding rows are all zero (no input edges); logit columns are all zero (no output edges)**. Stored on CPU, float32.

---

## 4. Graph pruning

### 4.1 Paper (Appendix "Graph Pruning", verbatim algorithm)
1. "replace all the edge weights with their absolute values … Then we normalize the input edges to each node so that they sum to 1. Let A refer to this normalized, unsigned adjacency matrix" — indexed **(target, source)**; normalization is per **row** = per target's input edges. (Main-text variant allows "or simply clamp negative values to 0"; the reference impl uses abs.)
2. Indirect influence: `B = A + A² + A³ + ⋯ = (I − A)⁻¹ − I` (Neumann series). "Entries of B indicate the sum of the strengths of all paths … strength of a path = product of its constituent edges in A."
3. Logit influence score per node: weighted average of the rows of B corresponding to logit nodes, "weighting according to the model's output probabilities for each token."
4. **Node pruning**: sort non-logit nodes by logit-influence desc; "choose the minimum cutoff index such that the sum of the influence scores prior to the cutoff divided by the total sum exceeds a given threshold. We use a threshold of **0.8**"; prune the rest.
5. **Edge pruning**: recompute Â and influence on the pruned graph; "assign a logit influence score to each edge by multiplying the logit influence score of the edge's output node by the normalized edge weight" (pseudocode: `node_score[logit_weights > 0] = logit_weights[...]`, i.e. logit nodes score = their probability; `edge_score = A * node_score[:, None]`); prune by the same cumulative strategy with cutoff **0.98**.
6. Logit nodes pruned separately: top K with cumprob > 0.95, K ≤ 10 (already done at construction).
7. "**Embedding nodes and error nodes are not pruned.**"
- Thresholds ≈ "the subset of nodes responsible for ~80% of the influence on the logits, and the subset of their input edges responsible for ~98% of the remaining influence." Typical effect: ~10× fewer nodes, ~500× fewer edges; completeness drops ~20% (default 0.8 pruning gave completeness 0.70 vs 0.87 at 0.95 in the acronym example).

### 4.2 circuit-tracer `prune_graph` (graph.py:178–252); defaults `node_threshold=0.8, edge_threshold=0.98`
- `normalize_matrix` (:125–127): `Â = |A| / |A|.sum(dim=1, keepdim=True).clamp(min=1e−10)` — abs + row(=target) normalization.
- `compute_influence` (:130–147): iterative `influence = Σ_k logit_weights @ Â^k`, looped until the running product is all-zero (max 1000 iters; raises on non-convergence) — equivalent to `logit_weights @ ((I−Â)⁻¹ − I)`. `logit_weights` = zeros except logit rows = `logit_probabilities` (**raw top-k probs, not renormalized**) (:205–208).
- Node mask: `node_influence >= find_threshold(node_influence, 0.8)`; `find_threshold` (:162–169) = sort desc, `cumsum/total`, `sorted_scores[searchsorted(cum, threshold)]` (keeps nodes up to and including the one that crosses the threshold). Token-embedding and logit nodes always kept: `node_mask[−n_logits−n_tokens:] = True` (:213–214). **DEVIATION: error nodes are NOT protected — circuit-tracer can prune error nodes, while the paper says error nodes are never pruned.**
- Pruned matrix: zero out masked rows AND columns (:217–219). Edge scores: `compute_edge_influence` (:154–159) = `Â_pruned * (influence + logit_weights)[:, None]` (adding `logit_weights` implements "logit node score = its probability"). Edge mask: `edge_scores >= find_threshold(edge_scores.flatten(), 0.98)` (:223–225).
- Cleanup loop (:227–244): iteratively drop feature/error nodes with no surviving **outgoing** edges and feature nodes with no surviving **incoming** edges (edges to/from dropped nodes removed each round) until fixpoint.
- Returns `PruneResult(node_mask, edge_mask, cumulative_scores)` where `cumulative_scores[i]` = cumulative influence fraction at node i's sorted rank (:246–252). Pruning is non-destructive (masks; the Graph object keeps the dense matrix).

---

## 5. Feature interventions / steering

### 5.1 Paper semantics
- Interventions run **on the underlying model**: "Features can be intervened on by modifying their computed activation and injecting its modified decoding in lieu of the original reconstruction."
- **Constrained patching** (default validation tool): "To perform interventions over layer ranges, we modify the decoding of a feature at each layer in the given range, and run a forward pass starting from the last layer in the range. Since we aren't recomputing a layer's MLP output based on the result of interventions earlier in the range, the only change to the model's MLP outputs will be our intervention. … it doesn't allow an intervention to have second-order effects within its patching range."
- **Attention frozen**: "in our perturbation experiments, we keep attention patterns fixed at the values observed during an unperturbed forward pass" (so QK responses to the perturbation are excluded, matching the graph's assumptions).
- **Steering-factor (M) convention — multiplicative on the original activation**: "a multiplicative version of constrained patching, in which we multiply a target feature's activation by M in the [ℓ−1, ℓ] layer range". Suppression example: "we set every feature in a node's activation to be the opposite of its original value (or equivalently, we steer multiplicatively with a factor of −1)". So paper M: `a_new = M · a_orig` → M=1 no-op, M=0 ablation, M=−1 sign-flip. Negative multiples (not mere ablation) are usually needed to change outputs ("§ Unexplained Variance and Choice of Steering Factors": unexplained variance, inexhaustive feature selection, incomplete cross-layer capture).
- Why not just add decoder vectors during a normal forward: **double counting** — layer-2 activations being reconstructed may be causally upstream of layer-3 reconstructions, so injecting at both layers doubles the effect ("Nuances of Steering with Cross-Layer Features"). Constrained patching clamps MLP outputs to precomputed values instead of adding.
- **Iterative patching** (alternative): "iteratively recompute the values of features which read from layers in the patching range, letting them take on new values due to our intervention … it can often lead to cascading errors"; captures knock-on effects, less precise for edge validation.
- Direct edges are "nearly forced to be confirmed" under frozen attention; the non-trivial test is knock-on (multi-hop) effects.
- Choice of range: typically start at layer 1 (or the feature's layer) and sweep the **end layer** ("Localizing Important Layers": start layer 1, sweep patching end layer); feature-feature validation used constrained range `[i, i+2]`, i = encoder layer (Appendix "Validating Feature-to-feature Influence").

### 5.2 circuit-tracer `feature_intervention` (replacement_model_transformerlens.py:718–771) — **runs on the real model, @torch.no_grad**
- Interventions: sequence of tuples `(layer, position, feature_idx, value)` (:21–26). **`value` is the ABSOLUTE target activation, not a multiplier.** The delta is `value − a_current` (:644–648). To implement "steer at M×", the caller must pass `value = M · a_orig`. (Beware off-by-one between conventions: paper-M multiplies (`M=0` ablates, `M=−1` flips); a delta-style convention `a_new = (1+M)·a` shifts everything by 1 — `M=0` no-op, `M=−1` ablate, `M=−2` flip.)
- Signature knobs: `constrained_layers: range | None = None`, `freeze_attention: bool = True`, `apply_activation_function: bool = True` (False → recorded activations are pre-activations, "as attribution predicts the change in pre-activation feature values").
- **Freezing** — `setup_intervention_with_freeze` (:454–545): a clean pass caches, then freeze-hooks replay cached values:
  - always (`freeze_attention=True` or constrained): every `hook_pattern` (attention patterns frozen to clean values);
  - if `constrained_layers`: also freeze `feature_output_hook` (i.e. **clamp MLP outputs to clean values**) but only for layers inside the range (:486–491);
  - `hook_scale` (LN denominators) frozen **only if the constrained range covers all model layers** (:473–474) — that case = pure direct effects;
  - skip-transcoders: frozen MLP-out would freeze the skip too, so `diff_hook`/`add_diff_hook` (:517–544) add back `(skip(current input) − skip(clean input))` per constrained layer, letting the linear skip path carry direct effects.
- **Delta computation** (`calculate_delta_hook`, :627–668), at each intervened layer's feature-output hook:
  - constrained mode: baseline activations = **original clean activations** (no recomputation → no second-order effects within range);
  - unconstrained (iterative) mode: baseline = **current activations of this perturbed forward pass** (encoded live at `feature_input_hook`), so the feature stays clamped at `value` as its input changes and effects propagate through the real MLPs;
  - `activation_deltas[pos, feat] = value − a_baseline`; decoder delta: PLT `Δ[layer] += δ · W_dec[feat]`; **CLT: for every output layer ℓ ≥ layer, `layer_deltas[ℓ, pos] += δ · W_dec^{layer→ℓ}[feat]`** (:653–668).
- **Application** (`intervention_hook`, :670–696): `acts += layer_deltas[layer]` at each layer's feature-output hook, but **only for layers in `intervention_range = constrained_layers or range(n_layers)`** — a CLT feature's writes beyond the constrained range are dropped; deltas zeroed after use (multi-token generation).
- Hook order at the same hook point: freeze (replace with clean) → delta calc → intervention add ⇒ constrained semantics = "MLP-out clamped to clean + decoder delta". Note: within the range, **attention (frozen patterns, live V) still responds linearly** to accumulated deltas in circuit-tracer, whereas the paper's "run a forward pass starting from the last layer in the range" would exclude within-range attention responses — a subtle semantic difference to keep in mind when validating against paper figures (post-range behavior identical in spirit).
- Logits cached at `unembed.hook_post` with **manual softcap re-application** (`softcap · tanh(logits/softcap)`) since TL applies softcap post-hook (:701–714).
- `feature_intervention_generate` (:791–900): with kv-cache, freeze/constraint applies only to the first generated token; open-ended interventions (`pos = slice(x, None)`) are retargeted to position 0 for incremental steps (:773–789).

---

## 6. Metrics (replacement / completeness scores, faithfulness)

### 6.1 Paper definitions (verbatim, "Evaluating and Comparing Graph Completeness")
- **Graph completeness score**: "measures the fraction of input edges (weighted by the target node's logit influence score) that come from feature or embedding nodes rather than error nodes."
- **Graph replacement score**: "measures the fraction of end-to-end graph paths (weighted by strength) that proceed from embedding nodes to logit nodes via feature nodes (rather than error nodes)."
- "completeness … gives more 'partial credit' …, whereas replacement score rewards complete explanations." Benchmarks: 18L 10M CLT ≈ completeness 0.80, replacement 0.61 (PLT 10M: 0.78 / 0.37). For pruned-graph scoring, "pruned nodes now count towards the error terms."

### 6.2 circuit-tracer `compute_graph_scores` (graph.py:255–302)
With `Â = normalize_matrix(adjacency)`, `logit_weights` = probs on logit rows, `influence = compute_influence(Â, logit_weights)`:
- `replacement_score = Σ influence[token nodes] / (Σ influence[token nodes] + Σ influence[error nodes])` (:293–296) — implemented as relative total logit-influence of embedding vs error nodes (path-mass interpretation of the paper's definition).
- `completeness_score = Σ_t (1 − Σ_{err srcs} Â[t, err]) · w_t / Σ_t w_t` with `w_t = influence_t + logit_weight_t` (:298–300) — non-error input fraction per node, weighted by that node's logit influence (logit nodes weighted by their probability; nodes with no inputs — embeddings, errors — have non_error_fraction 1 due to the normalization clamp).
- Node influence validation (paper appendix): node logit-influence beats direct-edge and activation baselines at predicting ablation-KL; feature→feature influence vs constrained-ablation effect (range `[i, i+2]`): **Spearman ρ = 0.72** over 20 prompts (ablation effect normalized by original activation, absolute value, since influence is unsigned).
- Mechanistic faithfulness of the local replacement model (paper appendix): perturb (encoder direction scaled so the feature's activation rises by 0.1 / random direction of same norm / upstream-feature patch), propagate in both models (attention frozen in both), compare net perturbations per layer by cosine sim and normalized MSE: **~0.8 cos-sim, ~0.4 NMSE one layer after the intervention; discrepancies compound over layers** (directions degrade gradually; magnitudes can degrade catastrophically, worse for larger dictionaries — suspected cause: unfrozen-vs-frozen normalization denominators).

---

## 7. Verification pitfalls checklist (paper ↔ circuit-tracer ↔ re-implementation)

1. Adjacency orientation `[target, source]`; node ordering `[features, errors(layer-major), embeddings, logits]`.
2. `decoder_vecs` pre-scaled by `a_s`; CLT edges **sum** decoder-layer contributions via `encoder_to_decoder_map` (+=).
3. Gradient injection **replaces** (index_put_) the grad at the target's (row, pos); backward seeded with zeros from the max layer; `retain_graph` on all but last batch.
4. Injection points: feature target → residual at its `feature_input_hook` (whatever tensor the encoder reads); logit target → post-`ln_final` tensor (forward ran with `stop_at_layer=n_layers`, `ln_final` applied manually).
5. Logit input vector demeaned over the **whole vocab**; probabilities from real-model logits (softcap included in probs, excluded from gradients); smallest top-k set with cumprob ≥ 0.95, cap 10, final position only.
6. Backward-linear paths: residual, attention OV (patterns detached), norms with detached `hook_scale` (incl. `ln1_post/ln2_post` for Gemma-style blocks), transcoder `W_skip`. Blocked: MLP out (`skip + (acts−skip).detach()`), QK, denominators.
7. Error = true MLP out − (decoder sum + `b_dec` + skip). Errors zeroed at position 0; feature activations zeroed at position 0; special token prepended if absent.
8. JumpReLU uses strict `>` with per-layer threshold; CLT `W_dec[i]` shape `[d_tc, n_layers−i, d_model]`; PLT skip is `x @ W_skip.T` vs CLT `x @ W_skip[ℓ]`.
9. Pruning: abs → row-normalize (clamp 1e−10) → influence by iterated `w @ Â^k` (paper: `B=(I−Â)⁻¹−I`); node threshold 0.8, edge threshold 0.98 on cumulative sorted scores; edge score = `Â_pruned · (influence + logit_prob)[target]`; tokens/logits always kept; **circuit-tracer can prune error nodes (paper says never prune them)**; iterative dangling-node cleanup.
10. Interventions on the **real model**, no_grad, `value` = absolute target activation (delta = value − baseline); paper's M is multiplicative (`M=0` ablate, `M=−1` flip) — beware `(1+M)` off-by-one against delta conventions.
11. Constrained mode: MLP outs clamped to clean within range, deltas from clean activations, CLT writes truncated to range, LN frozen only if range = all layers, attention patterns always frozen (default `freeze_attention=True`); unconstrained mode = iterative re-anchoring of deltas on live activations with real-MLP propagation.
12. `attribute` defaults: `max_n_logits=10, desired_logit_prob=0.95, batch_size=512, update_interval=4`; forward duplicated `batch_size`× for batched backward; adaptive feature selection by partial influences when `max_feature_nodes` set.
