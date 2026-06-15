"""Feature interventions on the local replacement model.

The local replacement model (see :mod:`llm_circuits.circuits.local_replacement_model`)
makes every transcoder feature an explicit, addressable quantity while freezing
attention patterns and LayerNorm denominators and injecting a *constant* error
term.  That makes it the right substrate for **interventions**: zeroing a feature
removes exactly its decoder contribution from the residual stream, downstream
encoders re-read the perturbed residual (so downstream features genuinely
respond), and — because the error nodes are constants captured from the original
run — the effect we measure is purely the feature we ablated, not a
self-cancelling reconstruction.

This is the protocol used to *validate* attribution-graph edges: perturb a source
feature, then check that the predicted downstream features / logits actually move.

The faithful entry point is :func:`run_feature_intervention` (the paper's
constrained patching with negative steering); :func:`run_feature_ablation` is the
simpler zeroing variant, and the ``run_progressive_*`` helpers sweep cumulative
curves.

Scope: per-layer transcoders (Qwen3) and cross-layer transcoders share the same
``ablations`` plumbing; the helpers here are family-agnostic but default to Qwen3
module templates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from llm_circuits.circuits.local_replacement_model import (
    _QWEN3_LAYERNORM_TEMPLATES,
    LocalReplacementModel,
    capture_constants,
)
from llm_circuits.circuits.replacement_model import _is_transcoder_set
from llm_circuits.logging import get_logger

if TYPE_CHECKING:
    from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder
    from circuit_tracer.transcoder.single_layer_transcoder import TranscoderSet

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureAblation:
    """Zero a single transcoder feature.

    Attributes
    ----------
    layer:
        Transformer layer index of the feature.
    feature_idx:
        Transcoder feature index.
    position:
        Sequence position to ablate.  ``None`` (the default) ablates the feature
        at *every* sequence position.
    """

    layer: int
    feature_idx: int
    position: int | None = None


@dataclass(frozen=True)
class FeatureIntervention:
    """Steer/clamp a single transcoder feature to a target value.

    The target is either absolute (``value``) or multiplicative
    (``factor`` * the feature's *clean* activation).  The paper's primary
    protocol is **negative steering** — set the feature to the opposite of its
    original value, i.e. ``factor=-1`` (see :func:`negative_steer`).  ``factor=0``
    or ``value=0`` is plain ablation.  ``position=None`` applies at every
    sequence position.
    """

    layer: int
    feature_idx: int
    position: int | None = None
    value: float | None = None
    factor: float | None = None

    def target(self, clean_activation: float) -> float:
        """Resolve the absolute target value given the feature's clean activation."""
        if self.value is not None:
            return self.value
        if self.factor is not None:
            return self.factor * clean_activation
        return 0.0


def negative_steer(
    layer: int, feature_idx: int, position: int | None = None
) -> FeatureIntervention:
    """The paper's canonical perturbation: steer a feature to ``-1x`` its clean value."""
    return FeatureIntervention(layer, feature_idx, position=position, factor=-1.0)


@dataclass
class AblationResult:
    """Outcome of :func:`run_feature_ablation`."""

    baseline_logits: Tensor
    """Local-model logits with no ablation, shape ``(seq, vocab)``."""

    ablated_logits: Tensor
    """Local-model logits with the ablations applied, shape ``(seq, vocab)``."""

    baseline_features: dict[int, Tensor] = field(default_factory=dict)
    """Per-layer feature activations (detached) before ablation, ``(seq, d_transcoder)``."""

    ablated_features: dict[int, Tensor] = field(default_factory=dict)
    """Per-layer feature activations (detached) after ablation, ``(seq, d_transcoder)``."""

    @property
    def logit_delta(self) -> Tensor:
        """``ablated_logits - baseline_logits``, shape ``(seq, vocab)``."""
        return self.ablated_logits - self.baseline_logits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ablations_to_dict(
    ablations: list[FeatureAblation],
) -> dict[int, list[tuple[int | None, int]]]:
    """Group :class:`FeatureAblation` specs into the ``{layer: [(position, feature_idx)]}``
    mapping consumed by :class:`LocalReplacementModel`.
    """
    grouped: dict[int, list[tuple[int | None, int]]] = {}
    for a in ablations:
        grouped.setdefault(a.layer, []).append((a.position, a.feature_idx))
    return grouped


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_feature_ablation(
    model: nn.Module,
    transcoder: TranscoderSet | CrossLayerTranscoder,
    input_ids: Tensor,
    ablations: list[FeatureAblation],
    *,
    n_bos_tokens: int = 1,
    mlp_name_template: str = "model.layers.{layer}.mlp",
    output_module_template: str | None = None,
    attn_name_template: str = "model.layers.{layer}.self_attn",
    layernorm_templates: list[str] | None = None,
    final_norm_name: str = "model.norm",
) -> AblationResult:
    """Run baseline and ablated local-replacement forward passes and compare them.

    Captures constants once, then runs the local replacement model twice with the
    *same* frozen attention/LayerNorm and constant error nodes: once without
    ablation (baseline) and once with the requested features zeroed.  Both passes
    run under ``no_grad`` (all parameters are frozen, so no graph is needed).

    Args:
        model: The full language model.
        transcoder: A ``TranscoderSet`` or ``CrossLayerTranscoder``.
        input_ids: Token ids, shape ``(batch, seq)`` or ``(seq,)``.
        ablations: Features to zero.  An empty list makes ``ablated_*`` equal the
            baseline (useful as a sanity check).
        n_bos_tokens: Leading positions whose original MLP output is preserved.
        mlp_name_template, output_module_template, attn_name_template,
        layernorm_templates, final_norm_name: Module-name plumbing, forwarded to
        :class:`LocalReplacementModel` (defaults target Qwen3).

    Returns:
        An :class:`AblationResult` with baseline/ablated logits and per-layer
        feature activations.
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if layernorm_templates is None:
        layernorm_templates = list(_QWEN3_LAYERNORM_TEMPLATES)

    is_set = _is_transcoder_set(transcoder)
    n_layers = len(transcoder) if is_set else transcoder.n_layers

    caps = capture_constants(
        model,
        input_ids,
        n_layers=n_layers,
        mlp_name_template=mlp_name_template,
        attn_name_template=attn_name_template,
        layernorm_templates=layernorm_templates,
        final_norm_name=final_norm_name,
    )

    common = dict(
        include_error=True,
        n_bos_tokens=n_bos_tokens,
        mlp_name_template=mlp_name_template,
        output_module_template=output_module_template,
        attn_name_template=attn_name_template,
        layernorm_templates=layernorm_templates,
        final_norm_name=final_norm_name,
    )

    # Baseline (no ablation).  Its per-layer error nodes are the *clean* errors,
    # which we freeze for the ablated pass: otherwise the error term would be
    # re-derived as ``captured_mlp_out - reconstruction`` and exactly cancel the
    # ablation, leaving the output unchanged.
    with torch.no_grad(), LocalReplacementModel(model, transcoder, caps, **common) as lm:
        base = lm.forward(input_ids)

    # Ablated: zero the requested features and reuse the clean error nodes so the
    # ablation actually propagates to the logits.
    abl_dict = ablations_to_dict(ablations)
    with (
        torch.no_grad(),
        LocalReplacementModel(
            model, transcoder, caps, ablations=abl_dict, frozen_errors=base.errors, **common
        ) as lm,
    ):
        abl = lm.forward(input_ids)

    return AblationResult(
        baseline_logits=base.logits.detach(),
        ablated_logits=abl.logits.detach(),
        baseline_features=base.features,
        ablated_features=abl.features,
    )


def run_feature_intervention(
    model: nn.Module,
    transcoder: TranscoderSet | CrossLayerTranscoder,
    input_ids: Tensor,
    interventions: list[FeatureIntervention],
    *,
    mode: str = "constrained",
    n_bos_tokens: int = 1,
    mlp_name_template: str = "model.layers.{layer}.mlp",
    output_module_template: str | None = None,
    attn_name_template: str = "model.layers.{layer}.self_attn",
    layernorm_templates: list[str] | None = None,
    final_norm_name: str = "model.norm",
    decoder_layer_template: str = "model.layers.{layer}",
) -> AblationResult:
    """Steer/clamp features on the local replacement model and compare to baseline.

    ``mode="constrained"`` (the paper's **primary** protocol): freeze attention,
    LayerNorm denominators, and error nodes; steer each feature based on its
    **clean** activation; inject the change as a single residual-stream delta at the
    *last* intervened layer's output, so MLPs *within* the range are not recomputed
    and only layers *after* the range respond.  The change for feature ``f`` is
    ``W_dec[f] * (target_f - clean_act_f)``; with :func:`negative_steer`
    (``factor=-1``) the target is ``-clean_act_f``, so its contribution flips sign.

    ``mode="iterative"`` (the paper's appendix alternative): freeze attention and
    error nodes but let **LayerNorm recompute**, and clamp each feature to its
    target during the live transcoder recompute so effects propagate through every
    layer's recomputed features.

    ``ablated_logits`` in the returned result holds the *intervened* logits.
    Per-layer transcoders only (Qwen3); raises for cross-layer transcoders.
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if layernorm_templates is None:
        layernorm_templates = list(_QWEN3_LAYERNORM_TEMPLATES)

    is_set = _is_transcoder_set(transcoder)
    if not is_set:
        raise NotImplementedError(
            "constrained intervention is implemented for per-layer transcoders only"
        )
    n_layers = len(transcoder)

    caps = capture_constants(
        model,
        input_ids,
        n_layers=n_layers,
        mlp_name_template=mlp_name_template,
        attn_name_template=attn_name_template,
        layernorm_templates=layernorm_templates,
        final_norm_name=final_norm_name,
    )
    common = dict(
        include_error=True,
        n_bos_tokens=n_bos_tokens,
        mlp_name_template=mlp_name_template,
        output_module_template=output_module_template,
        attn_name_template=attn_name_template,
        layernorm_templates=layernorm_templates,
        final_norm_name=final_norm_name,
        freeze_layernorm=mode == "constrained",
    )

    # Clean baseline (gives clean activations for the steering targets + the error
    # nodes we freeze for the intervened pass).
    with torch.no_grad(), LocalReplacementModel(model, transcoder, caps, **common) as lm:
        base = lm.forward(input_ids)

    if not interventions:
        return AblationResult(
            baseline_logits=base.logits.detach(),
            ablated_logits=base.logits.detach(),
            baseline_features=base.features,
            ablated_features=base.features,
        )

    device = input_ids.device
    seq = input_ids.shape[1]

    if mode == "iterative":
        # Attention + errors frozen, LayerNorm recomputes; clamp each feature to its
        # target during the live transcoder recompute (effects propagate everywhere).
        clamps: dict[int, list[tuple]] = {}
        for iv in interventions:
            feats = base.features[iv.layer]
            if feats.dim() == 3:
                feats = feats[0]
            positions = [iv.position] if iv.position is not None else list(range(n_bos_tokens, seq))
            for p in positions:
                clean_act = feats[p, iv.feature_idx].float().item()
                clamps.setdefault(iv.layer, []).append((p, iv.feature_idx, iv.target(clean_act)))
        with (
            torch.no_grad(),
            LocalReplacementModel(
                model, transcoder, caps, ablations=clamps, frozen_errors=base.errors, **common
            ) as lm,
        ):
            intervened = lm.forward(input_ids)
        return AblationResult(
            baseline_logits=base.logits.detach(),
            ablated_logits=intervened.logits.detach(),
            baseline_features=base.features,
            ablated_features=intervened.features,
        )

    if mode != "constrained":
        raise ValueError(f"mode must be 'constrained' or 'iterative', got {mode!r}")

    # Constrained patching: build the residual delta to inject at the last
    # intervened layer's output.
    l_max = max(iv.layer for iv in interventions)
    delta: Tensor | None = None
    for iv in interventions:
        feats = base.features[iv.layer]
        if feats.dim() == 3:
            feats = feats[0]
        dec = (
            transcoder.transcoders[iv.layer]
            ._get_decoder_vectors(torch.tensor([iv.feature_idx], device=device))[0]
            .float()
        )  # (d_model,)
        if delta is None:
            delta = torch.zeros(seq, dec.shape[0], device=device, dtype=torch.float32)
        positions = [iv.position] if iv.position is not None else list(range(n_bos_tokens, seq))
        for p in positions:
            clean_act = feats[p, iv.feature_idx].float().item()
            delta[p] += dec * (iv.target(clean_act) - clean_act)

    dec_layer = model.get_submodule(decoder_layer_template.format(layer=l_max))

    def _inject(_m: nn.Module, _i: Any, output: Any) -> Any:
        res = output[0] if isinstance(output, tuple) else output
        new = res + delta.to(res.dtype)
        return (new, *output[1:]) if isinstance(output, tuple) else new

    with (
        torch.no_grad(),
        LocalReplacementModel(model, transcoder, caps, frozen_errors=base.errors, **common) as lm,
    ):
        handle = dec_layer.register_forward_hook(_inject)
        try:
            intervened = lm.forward(input_ids)
        finally:
            handle.remove()

    return AblationResult(
        baseline_logits=base.logits.detach(),
        ablated_logits=intervened.logits.detach(),
        baseline_features=base.features,
        ablated_features=intervened.features,
    )


def ablation_logit_effect(
    result: AblationResult,
    token_ids: list[int],
    *,
    position: int = -1,
) -> dict[int, float]:
    """Return ``ablated_logit - baseline_logit`` for each token in *token_ids*.

    *position* defaults to the last sequence position (the next-token prediction).
    A large negative value means the ablated feature was *promoting* that token.
    """
    base = result.baseline_logits[position]
    abl = result.ablated_logits[position]
    return {int(t): (abl[t] - base[t]).item() for t in token_ids}


def ablation_prob_effect(
    result: AblationResult,
    token_ids: list[int],
    *,
    position: int = -1,
) -> dict[int, tuple[float, float]]:
    """Return ``(baseline_prob, ablated_prob)`` for each token in *token_ids*.

    Post-softmax probabilities at *position* (default last) — a more intuitive
    companion to :func:`ablation_logit_effect`, e.g. ``0.72 -> 0.08`` makes the
    effect of an ablation clearer than the raw logit delta.
    """
    base = result.baseline_logits[position].softmax(dim=-1)
    abl = result.ablated_logits[position].softmax(dim=-1)
    return {int(t): (base[t].item(), abl[t].item()) for t in token_ids}


# ---------------------------------------------------------------------------
# Progressive (cumulative) ablation
# ---------------------------------------------------------------------------


@dataclass
class ProgressiveAblationResult:
    """Cumulative-ablation curve for one target token.

    ``n_ablated[k]`` features removed → target ``logits[k]`` / ``probs[k]``.
    ``k = 0`` is the clean baseline.
    """

    n_ablated: list[int]
    logits: list[float]
    probs: list[float]
    token_id: int
    position: int


def run_progressive_ablation(
    model: nn.Module,
    transcoder: TranscoderSet | CrossLayerTranscoder,
    input_ids: Tensor,
    ordered_ablations: list[FeatureAblation],
    token_id: int,
    *,
    n_bos_tokens: int = 1,
    position: int = -1,
    mlp_name_template: str = "model.layers.{layer}.mlp",
    output_module_template: str | None = None,
    attn_name_template: str = "model.layers.{layer}.self_attn",
    layernorm_templates: list[str] | None = None,
    final_norm_name: str = "model.norm",
) -> ProgressiveAblationResult:
    """Ablate the first ``k`` of *ordered_ablations* for ``k = 0 … len``, recording
    the *token_id* logit and probability at *position* after each step.

    Constants are captured once and the clean-run errors are frozen, so every step
    is a single forward.  The resulting curve shows how the target collapses as its
    top features are removed in order — the intuitive companion to the single-shot
    :func:`run_feature_ablation`.
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if layernorm_templates is None:
        layernorm_templates = list(_QWEN3_LAYERNORM_TEMPLATES)

    is_set = _is_transcoder_set(transcoder)
    n_layers = len(transcoder) if is_set else transcoder.n_layers

    caps = capture_constants(
        model,
        input_ids,
        n_layers=n_layers,
        mlp_name_template=mlp_name_template,
        attn_name_template=attn_name_template,
        layernorm_templates=layernorm_templates,
        final_norm_name=final_norm_name,
    )
    common = dict(
        include_error=True,
        n_bos_tokens=n_bos_tokens,
        mlp_name_template=mlp_name_template,
        output_module_template=output_module_template,
        attn_name_template=attn_name_template,
        layernorm_templates=layernorm_templates,
        final_norm_name=final_norm_name,
    )

    logits_out: list[float] = []
    probs_out: list[float] = []

    def _record(ctx) -> None:
        row = ctx.logits[position]
        logits_out.append(row[token_id].item())
        probs_out.append(row.softmax(dim=-1)[token_id].item())

    # k = 0: clean baseline (also yields the errors we freeze for later steps).
    with torch.no_grad(), LocalReplacementModel(model, transcoder, caps, **common) as lm:
        base = lm.forward(input_ids)
    frozen = base.errors
    _record(base)

    # k = 1 … K: ablate the first k features, reusing the clean errors.
    for k in range(1, len(ordered_ablations) + 1):
        abl = ablations_to_dict(ordered_ablations[:k])
        with (
            torch.no_grad(),
            LocalReplacementModel(
                model, transcoder, caps, ablations=abl, frozen_errors=frozen, **common
            ) as lm,
        ):
            _record(lm.forward(input_ids))

    return ProgressiveAblationResult(
        n_ablated=list(range(len(ordered_ablations) + 1)),
        logits=logits_out,
        probs=probs_out,
        token_id=token_id,
        position=position,
    )


def run_progressive_intervention(
    model: nn.Module,
    transcoder: TranscoderSet | CrossLayerTranscoder,
    input_ids: Tensor,
    ordered_interventions: list[FeatureIntervention],
    token_id: int,
    *,
    n_bos_tokens: int = 1,
    position: int = -1,
    mlp_name_template: str = "model.layers.{layer}.mlp",
    output_module_template: str | None = None,
    attn_name_template: str = "model.layers.{layer}.self_attn",
    layernorm_templates: list[str] | None = None,
    final_norm_name: str = "model.norm",
    decoder_layer_template: str = "model.layers.{layer}",
) -> ProgressiveAblationResult:
    """Cumulative **constrained-patching** curve (the faithful counterpart to
    :func:`run_progressive_ablation`).

    Applies the first ``k`` of *ordered_interventions* (e.g. :func:`negative_steer`)
    for ``k = 0 … len`` and records the target token's logit and probability at each
    step.  Constants are captured once; each step injects a residual delta at the
    last intervened layer (in-range MLPs are not recomputed).  Per-layer
    transcoders only.
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if layernorm_templates is None:
        layernorm_templates = list(_QWEN3_LAYERNORM_TEMPLATES)
    if not _is_transcoder_set(transcoder):
        raise NotImplementedError(
            "constrained progressive intervention is implemented for per-layer transcoders only"
        )
    n_layers = len(transcoder)

    caps = capture_constants(
        model,
        input_ids,
        n_layers=n_layers,
        mlp_name_template=mlp_name_template,
        attn_name_template=attn_name_template,
        layernorm_templates=layernorm_templates,
        final_norm_name=final_norm_name,
    )
    common = dict(
        include_error=True,
        n_bos_tokens=n_bos_tokens,
        mlp_name_template=mlp_name_template,
        output_module_template=output_module_template,
        attn_name_template=attn_name_template,
        layernorm_templates=layernorm_templates,
        final_norm_name=final_norm_name,
    )

    with torch.no_grad(), LocalReplacementModel(model, transcoder, caps, **common) as lm:
        base = lm.forward(input_ids)
    clean_features = base.features
    frozen = base.errors
    device = input_ids.device
    seq = input_ids.shape[1]

    logits_out: list[float] = []
    probs_out: list[float] = []

    def _record(row: Tensor) -> None:
        logits_out.append(row[token_id].item())
        probs_out.append(row.softmax(dim=-1)[token_id].item())

    _record(base.logits[position])

    for k in range(1, len(ordered_interventions) + 1):
        subset = ordered_interventions[:k]
        l_max = max(iv.layer for iv in subset)
        delta: Tensor | None = None
        for iv in subset:
            feats = clean_features[iv.layer]
            if feats.dim() == 3:
                feats = feats[0]
            dec = (
                transcoder.transcoders[iv.layer]
                ._get_decoder_vectors(torch.tensor([iv.feature_idx], device=device))[0]
                .float()
            )
            if delta is None:
                delta = torch.zeros(seq, dec.shape[0], device=device, dtype=torch.float32)
            positions = [iv.position] if iv.position is not None else list(range(n_bos_tokens, seq))
            for p in positions:
                clean_act = feats[p, iv.feature_idx].float().item()
                delta[p] += dec * (iv.target(clean_act) - clean_act)

        dec_layer = model.get_submodule(decoder_layer_template.format(layer=l_max))

        def _inject(_m: nn.Module, _i: Any, output: Any, _delta: Tensor = delta) -> Any:
            res = output[0] if isinstance(output, tuple) else output
            new = res + _delta.to(res.dtype)
            return (new, *output[1:]) if isinstance(output, tuple) else new

        with (
            torch.no_grad(),
            LocalReplacementModel(model, transcoder, caps, frozen_errors=frozen, **common) as lm,
        ):
            handle = dec_layer.register_forward_hook(_inject)
            try:
                ctx = lm.forward(input_ids)
            finally:
                handle.remove()
        _record(ctx.logits[position])

    return ProgressiveAblationResult(
        n_ablated=list(range(len(ordered_interventions) + 1)),
        logits=logits_out,
        probs=probs_out,
        token_id=token_id,
        position=position,
    )
