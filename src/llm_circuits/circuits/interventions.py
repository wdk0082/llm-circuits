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

This is the protocol used to *validate* attribution-graph edges: ablate a source
feature, then check that the predicted downstream features / logits actually move.

The main entry point is :func:`run_feature_ablation`.

Scope: per-layer transcoders (Qwen3) and cross-layer transcoders share the same
``ablations`` plumbing; the helpers here are family-agnostic but default to Qwen3
module templates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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
