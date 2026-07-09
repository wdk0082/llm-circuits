#!/usr/bin/env python3
"""Cross-verify our attribution graphs against circuit-tracer's ``attribute``.

Four check families, from exact/internal to cross-implementation:

[1] **Forward exactness** (internal, exact): the local replacement model's logits
    must equal the real model's logits (error nodes cancel the reconstruction gap).

[2] **Linearity identity** (internal, exact): in the frozen-linear backward graph,
    every feature pre-activation decomposes EXACTLY as

        h_t = b_enc_t + <dh/d emb, emb> + sum_l <dh/d resid_l, mlp_write_l>

    where ``mlp_write_l = reconstruction + error = captured MLP output`` (the leaf
    constants injected at each layer) and gradients are taken through the frozen
    attention/RMSNorm paths.  This validates the entire linearised backward that
    edge weights are built from, independent of circuit-tracer.  We also verify a
    sample of our emitted edge weights against plain per-target ``autograd.grad``
    (validating the batched/vmapped edge machinery).

[3] **Salient-logit parity** (exact): our ``_select_salient_logits`` must pick the
    same tokens/probabilities as circuit-tracer's ``compute_salient_logits``.

[4] **Graph cross-check** (tolerance-based): build the graph with our
    ``build_attribution_graph`` (HF model) and circuit-tracer's ``attribute``
    (TransformerLens model), same transcoders, same token ids.  Compare feature
    node sets, activations, edge weights on shared pairs, influence rankings and
    pruned node sets.  HF vs TL forward passes differ at the fp32 noise level, so
    activations near the JumpReLU threshold can flip — thresholds below reflect
    that (cosine/correlation close to 1, Jaccard close to but below 1).

Run (fp32; MPS or CPU — loads 0.6B twice plus transcoders):
    set -a; source .env; set +a
    uv run python verification/verify_attribution_graph.py
"""

from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np
import torch
from torch import Tensor

from llm_circuits.circuits.attribution_graph import (
    AttributionGraph,
    _select_salient_logits,
    build_attribution_graph,
)
from llm_circuits.circuits.graph_pruning import (
    _build_adjacency_matrix,
    _compute_influence,
    _compute_logit_weights,
    _normalize_matrix,
    prune_graph,
)
from llm_circuits.circuits.local_replacement_model import LocalReplacementModel, capture_constants
from llm_circuits.models.qwen3 import load_qwen3
from llm_circuits.settings import default_device
from llm_circuits.transcoders.circuit_tracer_loader import load_transcoder

MODEL_SIZE = "0.6b"
TL_NAME = "Qwen/Qwen3-0.6B"
PROMPT = "The capital of France is"
N_LINEARITY_TARGETS = 24  # random feature targets for the exact identity check
N_EDGE_SPOT_CHECKS = 200  # random edges re-derived with plain autograd
SEED = 0

PASSES: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    PASSES.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def node_key(n) -> tuple:
    """Implementation-independent node identity."""
    if n.node_type == "feature":
        return ("feature", n.layer, n.position, n.feature_idx)
    if n.node_type == "error":
        return ("error", n.layer, n.position)
    if n.node_type == "embedding":
        return ("embedding", n.position)
    return ("logit", n.token_id)


def ct_node_keys(g, n_layers: int) -> list[tuple]:
    """Node keys for circuit-tracer's Graph in ITS adjacency order:
    [features, errors (layer-major x pos), embeddings, logits]."""
    keys: list[tuple] = []
    for row in g.active_features[g.selected_features].tolist():
        layer, pos, feat = row
        keys.append(("feature", layer, pos, feat))
    n_pos = g.n_pos
    for layer in range(n_layers):
        for p in range(n_pos):
            keys.append(("error", layer, p))
    for p in range(n_pos):
        keys.append(("embedding", p))
    for tok in g.logit_tokens.tolist():
        keys.append(("logit", tok))
    return keys


def local_forward_with_grads(model, tc, input_ids, n_layers):
    """Replicate the builder's local forward: seeded embedding + in-graph residuals.

    Independent re-plumbing (not calling build_attribution_graph) so check [2]
    does not trust the builder's own hook wiring.
    """
    caps = capture_constants(
        model,
        input_ids,
        n_layers=n_layers,
        mlp_name_template="model.layers.{layer}.mlp",
        attn_name_template="model.layers.{layer}.self_attn",
        layernorm_templates=[
            "model.layers.{layer}.input_layernorm",
            "model.layers.{layer}.post_attention_layernorm",
        ],
        final_norm_name="model.norm",
    )
    embedding_ref: list[Tensor] = []
    residuals: dict[int, Tensor] = {}

    def _embed_hook(_m, _i, output):
        seeded = output.detach().requires_grad_(True)
        embedding_ref.append(seeded)
        return seeded

    def _mk_res_hook(i):
        def hook(_m, _i, output):
            residuals[i] = output[0] if isinstance(output, tuple) else output

        return hook

    handles = [model.get_submodule("model.embed_tokens").register_forward_hook(_embed_hook)]
    for i in range(n_layers):
        handles.append(
            model.get_submodule(f"model.layers.{i}").register_forward_hook(_mk_res_hook(i))
        )
    try:
        lrm = LocalReplacementModel(
            model, tc, caps, include_error=True, n_bos_tokens=1, capture_pre_activations=True
        )
        with lrm as lm:
            ctx = lm.forward(input_ids)
    finally:
        for h in handles:
            h.remove()
    return caps, ctx, embedding_ref[0], residuals


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    # CPU on purpose: fp32 transcoders for two models blow past the MPS allocator cap
    # (~36 GB), and CPU fp32 is the most deterministic backend for tight comparisons.
    device = "cpu"
    dtype = torch.float32  # tight numeric floors; bf16 noise would swamp the comparisons
    print(f"device={device} dtype={dtype} (default_device would be {default_device()})")

    from circuit_tracer.attribution.attribute import attribute as ct_attribute
    from circuit_tracer.graph import prune_graph as ct_prune_graph
    from circuit_tracer.replacement_model.replacement_model_transformerlens import (
        TransformerLensReplacementModel,
    )
    from circuit_tracer.utils.salient_logits import compute_salient_logits

    print(f"Loading our HF Qwen3-{MODEL_SIZE} + transcoders (fp32) ...", flush=True)
    model, _tok = load_qwen3(MODEL_SIZE, dtype_str="fp32", device_map=device)
    model.eval()
    # lazy_decoder: W_dec is re-read from disk per access (transient), keeping the
    # persistent footprint to the encoders — the full fp32 decoder set alone is ~15 GB.
    tc = load_transcoder(
        f"qwen3-{MODEL_SIZE}", device=device, dtype=dtype, lazy_decoder=True
    ).transcoder
    n_layers = len(tc)
    skip = getattr(tc.transcoders[0], "W_skip", None)
    print(f"n_layers={n_layers}  W_skip={'present' if skip is not None else 'absent'}")

    print(f"Loading circuit-tracer ReplacementModel ({TL_NAME}) ...", flush=True)
    rm = TransformerLensReplacementModel.from_pretrained_and_transcoders(
        TL_NAME, tc, device=device, dtype=dtype
    )

    # Same token ids on both sides, with circuit-tracer's special-token prepend.
    input_ids = rm.ensure_tokenized(PROMPT)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(device)
    seq_len = input_ids.shape[1]
    print(f"tokens ({seq_len}): {input_ids[0].tolist()}")

    # ==================================================================
    # [1] + [2] internal exactness on OUR side
    # ==================================================================
    print("\n=== [1] Local replacement forward exactness ===")
    _caps, ctx, embedding, residuals = local_forward_with_grads(model, tc, input_ids, n_layers)
    orig = ctx.original_logits.float()
    local = ctx.logits.float()
    max_logit_diff = (orig - local).abs().max().item()
    scale = orig.abs().max().item()
    check(
        "local logits == original logits",
        max_logit_diff <= 1e-3 * max(scale, 1.0),
        f"max|diff|={max_logit_diff:.3e} (logit scale {scale:.1f})",
    )

    print("\n=== [2] Frozen-linear backward: exact linearity identity ===")
    # mlp_write_l = reconstruction + error = captured MLP output (the injected leaf)
    mlp_write = {
        layer: (ctx.reconstructions[layer] + (e[0] if e.dim() == 3 else e)).float()
        for layer, e in ctx.errors.items()
    }
    emb2d = (embedding[0] if embedding.dim() == 3 else embedding).float()

    # Sample feature targets across layers/positions
    all_targets: list[tuple[int, int, int]] = []  # (layer, pos, feat)
    for layer, f in ctx.features.items():
        f2 = f[0] if f.dim() == 3 else f
        nz = f2[1:].nonzero()  # skip BOS position
        for pos_m1, feat in nz.tolist():
            all_targets.append((layer, pos_m1 + 1, feat))
    sel = rng.choice(
        len(all_targets), size=min(N_LINEARITY_TARGETS, len(all_targets)), replace=False
    )
    worst_rel = 0.0
    for i in sel:
        layer, pos, feat = all_targets[int(i)]
        pre = ctx.pre_activations[layer]
        pre2 = pre[0] if pre.dim() == 3 else pre
        h_t = pre2[pos, feat]
        grads = torch.autograd.grad(
            h_t,
            [embedding] + [residuals[sl] for sl in range(layer)],
            retain_graph=True,
            allow_unused=True,
        )
        b_enc = tc.transcoders[layer].b_enc[feat].float()
        total = b_enc.clone()
        g_emb = grads[0]
        g_emb2 = (g_emb[0] if g_emb.dim() == 3 else g_emb).float()
        total = total + (g_emb2 * emb2d).sum()
        for sl in range(layer):
            g = grads[sl + 1]
            if g is None:
                continue
            g2 = (g[0] if g.dim() == 3 else g).float()
            total = total + (g2 * mlp_write[sl]).sum()
        rel = (total - h_t.float()).abs().item() / max(h_t.float().abs().item(), 1e-3)
        worst_rel = max(worst_rel, rel)
    check(
        f"h_t = b_enc + sum(edges incl. bias paths) on {len(sel)} targets",
        worst_rel <= 5e-3,
        f"worst relative error {worst_rel:.3e}",
    )

    # ==================================================================
    # Build OUR graph (HF side)
    # ==================================================================
    print("\n=== Building OUR attribution graph (HF) ===")
    ours: AttributionGraph = build_attribution_graph(
        model, tc, input_ids, n_bos_tokens=1, feature_selection="all"
    )
    our_keys = [node_key(n) for n in ours.nodes]
    our_index = {k: i for i, k in enumerate(our_keys)}
    n_our_feat = sum(1 for k in our_keys if k[0] == "feature")
    print(f"our graph: {len(ours.nodes)} nodes ({n_our_feat} features), {len(ours.edges)} edges")

    # --- [2b] spot-check emitted edge weights against plain autograd ---
    print("\n=== [2b] Edge spot checks (batched machinery vs plain autograd) ===")
    edge_sel = rng.choice(
        len(ours.edges), size=min(N_EDGE_SPOT_CHECKS, len(ours.edges)), replace=False
    )
    by_target: dict[tuple, list[int]] = defaultdict(list)
    for ei in edge_sel:
        e = ours.edges[int(ei)]
        by_target[our_keys[e.target]].append(int(ei))
    worst_edge_rel = 0.0
    n_checked = 0
    for tkey, eis in by_target.items():
        if tkey[0] == "feature":
            _, layer, pos, feat = tkey
            pre = ctx.pre_activations[layer]
            pre2 = pre[0] if pre.dim() == 3 else pre
            tgt = pre2[pos, feat]
            n_src_layers = layer
        elif tkey[0] == "logit":
            tok = tkey[1]
            tgt = ctx.logits[seq_len - 1, tok] - ctx.logits[seq_len - 1].mean()
            n_src_layers = n_layers
        else:
            continue
        grads = torch.autograd.grad(
            tgt,
            [embedding] + [residuals[sl] for sl in range(n_src_layers)],
            retain_graph=True,
            allow_unused=True,
        )

        def grad_at(layer_idx: int, grads_=grads) -> Tensor | None:
            g = grads_[0] if layer_idx < 0 else grads_[layer_idx + 1]
            if g is None:
                return None
            return (g[0] if g.dim() == 3 else g).float()

        for ei in eis:
            e = ours.edges[ei]
            skey = our_keys[e.source]
            if skey[0] == "embedding":
                g2 = grad_at(-1)
                contrib = emb2d[skey[1]]
                spos = skey[1]
            elif skey[0] == "error":
                g2 = grad_at(skey[1])
                err = ctx.errors[skey[1]]
                contrib = (err[0] if err.dim() == 3 else err)[skey[2]].float()
                spos = skey[2]
            else:  # feature source
                _, slayer, spos, sfeat = skey
                g2 = grad_at(slayer)
                f2 = ctx.features[slayer]
                f2 = f2[0] if f2.dim() == 3 else f2
                act = f2[spos, sfeat].float()
                dec = (
                    tc.transcoders[slayer]
                    ._get_decoder_vectors(torch.tensor([sfeat], device=device))[0]
                    .float()
                )
                contrib = act * dec
            if g2 is None:
                continue
            w_ref = float((g2[spos] * contrib).sum())
            rel = abs(w_ref - e.weight) / max(abs(w_ref), 1e-6)
            worst_edge_rel = max(worst_edge_rel, rel)
            n_checked += 1
    check(
        f"{n_checked} edge weights match plain autograd",
        worst_edge_rel <= 1e-3,
        f"worst relative error {worst_edge_rel:.3e}",
    )

    # ==================================================================
    # [3] salient-logit parity
    # ==================================================================
    print("\n=== [3] Salient logit selection parity ===")
    logit_vec = ctx.logits[seq_len - 1].float()
    W_U = model.get_submodule("lm_head").weight.detach().float()  # (vocab, d_model)
    ct_idx, ct_p, _ = compute_salient_logits(
        logit_vec, W_U, max_n_logits=10, desired_logit_prob=0.95
    )
    our_idx, our_p = _select_salient_logits(logit_vec, 0.95, 10)
    same = ct_idx.tolist() == our_idx.tolist()
    pdiff = (ct_p - our_p).abs().max().item() if same else float("nan")
    check(
        "same salient logit tokens + probs",
        same and pdiff < 1e-6,
        f"tokens ours={our_idx.tolist()} ct={ct_idx.tolist()} max|dp|={pdiff:.2e}",
    )

    # ==================================================================
    # [4] cross-check vs circuit-tracer's attribute()
    # ==================================================================
    print("\n=== [4] circuit-tracer attribute() cross-check ===")
    ct_graph = ct_attribute(input_ids[0], rm, batch_size=128, verbose=True)
    ct_keys = ct_node_keys(ct_graph, n_layers)
    ct_index = {k: i for i, k in enumerate(ct_keys)}
    A_ct = ct_graph.adjacency_matrix.float().cpu().numpy()
    n_ct_feat = sum(1 for k in ct_keys if k[0] == "feature")
    print(f"ct graph: {len(ct_keys)} nodes ({n_ct_feat} features), adjacency {A_ct.shape}")

    # --- node sets ---
    our_feat = {k for k in our_keys if k[0] == "feature"}
    ct_feat = {k for k in ct_keys if k[0] == "feature" and k[2] > 0}  # ct zeroes pos-0 anyway
    inter = our_feat & ct_feat
    jacc = len(inter) / max(len(our_feat | ct_feat), 1)
    check(
        "feature node sets agree (Jaccard)",
        jacc >= 0.98,
        f"ours={len(our_feat)} ct={len(ct_feat)} common={len(inter)} jaccard={jacc:.4f}",
    )

    our_logits_keys = [k for k in our_keys if k[0] == "logit"]
    ct_logits_keys = [k for k in ct_keys if k[0] == "logit"]
    check(
        "same logit nodes",
        set(our_logits_keys) == set(ct_logits_keys),
        f"ours={our_logits_keys} ct={ct_logits_keys}",
    )

    # --- activation agreement on common features ---
    acts_our = np.array([ours.nodes[our_index[k]].activation for k in sorted(inter)])
    ct_act_vals = ct_graph.activation_values[ct_graph.selected_features].float().cpu().numpy()
    ct_act = {k: ct_act_vals[i] for i, k in enumerate(ct_keys[:n_ct_feat])}
    acts_ct = np.array([ct_act[k] for k in sorted(inter)])
    act_rel = np.abs(acts_our - acts_ct) / np.maximum(np.abs(acts_ct), 1e-3)
    check(
        "feature activations match on common nodes",
        float(np.median(act_rel)) < 1e-2 and float(np.mean(act_rel)) < 5e-2,
        f"median rel diff {np.median(act_rel):.2e}, mean {np.mean(act_rel):.2e}, max {act_rel.max():.2e}",
    )

    # --- edge weights on common node pairs ---
    common_nodes = (
        inter
        | {k for k in our_keys if k[0] in ("embedding", "error")} & set(ct_keys)
        | set(our_logits_keys)
    )
    common_nodes &= set(ct_keys)
    our_w: dict[tuple[tuple, tuple], float] = defaultdict(float)
    for e in ours.edges:
        sk, tk = our_keys[e.source], our_keys[e.target]
        if sk in common_nodes and tk in common_nodes:
            our_w[(tk, sk)] += e.weight  # sum any duplicates
    pairs = list(our_w.keys())
    w_ours = np.array([our_w[p] for p in pairs])
    w_ct = np.array([A_ct[ct_index[t], ct_index[s]] for (t, s) in pairs])
    cos = float(np.dot(w_ours, w_ct) / max(np.linalg.norm(w_ours) * np.linalg.norm(w_ct), 1e-12))
    pear = float(np.corrcoef(w_ours, w_ct)[0, 1])
    mad = float(np.abs(w_ours - w_ct).max())
    check(
        f"edge weights agree on {len(pairs)} shared pairs",
        cos > 0.99 and pear > 0.99,
        f"cosine={cos:.5f} pearson={pear:.5f} max|dw|={mad:.3e}",
    )
    # mass we might be missing: ct edges (among common nodes) that we never emitted
    ct_mass = 0.0
    missing_mass = 0.0
    for tk in common_nodes:
        ti = ct_index[tk]
        row = A_ct[ti]
        for sk in common_nodes:
            v = row[ct_index[sk]]
            if v != 0.0:
                ct_mass += abs(v)
                if (tk, sk) not in our_w:
                    missing_mass += abs(v)
    frac_missing = missing_mass / max(ct_mass, 1e-12)
    check(
        "no ct edge mass missing from our graph",
        frac_missing < 1e-3,
        f"missing |w| fraction {frac_missing:.2e}",
    )

    # --- influence + pruning cross-check ---
    print("\n=== [4b] influence + pruning ===")
    A_our = _build_adjacency_matrix(ours.nodes, ours.edges)
    infl_our = _compute_influence(_normalize_matrix(A_our), _compute_logit_weights(ours.nodes))
    logit_w_ct = np.zeros(len(ct_keys))
    for i, k in enumerate(ct_keys):
        if k[0] == "logit":
            logit_w_ct[i] = float(ct_graph.logit_probabilities[ct_logits_keys.index(k)])
    infl_ct = _compute_influence(_normalize_matrix(A_ct.astype(np.float64)), logit_w_ct)
    common_sorted = sorted(common_nodes)
    iv_ours = np.array([infl_our[our_index[k]] for k in common_sorted])
    iv_ct = np.array([infl_ct[ct_index[k]] for k in common_sorted])

    def _spearman(a: np.ndarray, b: np.ndarray) -> float:
        """Spearman rho = Pearson correlation of the ranks (no scipy dependency)."""
        ra = np.argsort(np.argsort(a)).astype(np.float64)
        rb = np.argsort(np.argsort(b)).astype(np.float64)
        return float(np.corrcoef(ra, rb)[0, 1])

    rho = _spearman(iv_ours, iv_ct)
    check("influence rank agreement (Spearman)", rho > 0.98, f"rho={rho:.5f}")

    res_our = prune_graph(ours, node_threshold=0.8, edge_threshold=0.98)
    kept_our = {node_key(n) for n in res_our.graph.nodes}
    ct_res = ct_prune_graph(ct_graph, node_threshold=0.8, edge_threshold=0.98)
    ct_node_mask = (
        ct_res[0].cpu().numpy() if isinstance(ct_res, tuple) else ct_res.node_mask.cpu().numpy()
    )
    kept_ct = {k for i, k in enumerate(ct_keys) if ct_node_mask[i]}
    kept_ct = {
        k
        for k in kept_ct
        if not (k[0] in ("feature", "error") and k[-2 if k[0] == "feature" else -1] == 0)
    }
    ji = len(kept_our & kept_ct) / max(len(kept_our | kept_ct), 1)
    check(
        "pruned node sets agree (Jaccard)",
        ji >= 0.9,
        f"ours={len(kept_our)} ct={len(kept_ct)} jaccard={ji:.4f}",
    )

    # ==================================================================
    # summary
    # ==================================================================
    print("\n" + "=" * 70)
    n_pass = sum(1 for _, ok, _ in PASSES if ok)
    for name, ok, _detail in PASSES:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(
        f"\n{'ALL PASS' if n_pass == len(PASSES) else 'FAILURES PRESENT'} ({n_pass}/{len(PASSES)})"
    )
    sys.exit(0 if n_pass == len(PASSES) else 1)


if __name__ == "__main__":
    main()
