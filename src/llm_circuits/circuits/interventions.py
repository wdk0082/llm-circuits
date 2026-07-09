"""Feature interventions (faithful to circuit-tracer's ``feature_intervention``).

We steer/ablate a transcoder feature by adding its decoder delta to the **real**
model's MLP output: with our **m (additive-delta) convention**, a feature's new
activation is ``(1 + m) * clean``, so the delta is ``m * clean * W_dec`` (``m=0``
no change, ``-1`` ablate, ``-2`` flip).

.. warning::

    The Anthropic papers use a **multiplicative** steering factor M —
    ``a_new = M * a_clean`` (``M=0`` ablate, ``M=-1`` flip, "steer at -2x" means
    ``a_new = -2 * a_clean``).  The two conventions are off by one:
    ``M_paper = 1 + m_ours``.  circuit-tracer's ``feature_intervention`` takes an
    *absolute* target ``value`` instead (``value = M_paper * a_clean``).

This runs the real model (no transcoder reconstruction, no error nodes),
optionally freezing attention patterns and pinning MLP outputs to clean within a
layer range — see :func:`run_feature_intervention`.

This is the protocol used to *validate* attribution-graph edges: perturb a source
feature, then check that the predicted downstream features / logits actually move.
:func:`sweep_patch_end_layer` sweeps the layer-range knob and
:func:`run_progressive_intervention` sweeps a cumulative curve.

Scope: per-layer transcoders (Qwen3) — the decoder writes to its own layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from llm_circuits.circuits.local_replacement_model import (
    _QWEN3_LAYERNORM_TEMPLATES,
    LocalReplacementModel,
    _make_frozen_attn_forward,
    _make_frozen_rmsnorm_hook,
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
class FeatureIntervention:
    """Steer/clamp a single transcoder feature.

    Uses our **m (additive-delta) convention**: the feature's new activation is
    ``(1 + m) * clean``, so the decoder delta added to the MLP output is
    ``m * clean * W_dec``.  Hence:

    * ``m == 0``  → no change,
    * ``m == -1`` → ablation (the default),
    * ``m == -2`` → negative steer (flip the sign: new activation = ``-clean``).

    .. note:: The paper's multiplicative M is ``M_paper = 1 + m_ours`` (paper ``M=-1``
       sign-flip == our ``m=-2``); circuit-tracer's API takes the absolute ``value``.

    ``value`` overrides ``m`` with an absolute target.  ``position=None`` applies at every
    (non-BOS) sequence position.
    """

    layer: int
    feature_idx: int
    position: int | None = None
    value: float | None = None
    m: float = -1.0

    def target(self, clean_activation: float) -> float:
        """Resolve the absolute target activation given the feature's clean activation."""
        if self.value is not None:
            return self.value
        return (1.0 + self.m) * clean_activation


def steer(
    layer: int, feature_idx: int, m: float, position: int | None = None
) -> FeatureIntervention:
    """Steer a feature by additive multiple ``m`` (new activation = ``(1 + m) * clean``).

    This is the single entry point for the M convention: ``m=-1`` ablates, ``m=-2`` negative
    -steers (the paper's sign-flip), ``m>0`` amplifies.
    """
    return FeatureIntervention(layer, feature_idx, position=position, m=m)


@dataclass
class AblationResult:
    """Outcome of an intervention: baseline vs intervened logits (+ feature acts)."""

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
# Feature intervention (steer/ablate on the real model) — main entry point
# ---------------------------------------------------------------------------


def _make_base_mlp_hook(pin: bool, clean_out: Tensor | None, delta: Tensor | None) -> Any:
    """Forward hook on a *real* MLP module (steering-base-model).

    Optionally pins the MLP output to its clean recorded value (``pin``), then adds the
    decoder ``delta``.  Shapes broadcast: ``(1, seq, d_model)`` output + ``(seq, d_model)``
    delta.
    """

    def hook(_m: nn.Module, _inp: Any, output: Any) -> Any:
        res = output[0] if isinstance(output, tuple) else output
        if pin and clean_out is not None:
            res = clean_out.to(res.dtype)
        if delta is not None:
            res = res + delta.to(res.dtype)
        return (res, *output[1:]) if isinstance(output, tuple) else res

    return hook


def _steer_base_model(
    model: nn.Module,
    transcoder: TranscoderSet | CrossLayerTranscoder,
    input_ids: Tensor,
    interventions: list[FeatureIntervention],
    base: Any,
    caps: Any,
    *,
    l_max: int,
    patch_end_layer: int | None,
    freeze_attention: bool,
    n_bos_tokens: int,
    mlp_name_template: str,
    attn_name_template: str,
    layernorm_templates: list[str],
    final_norm_name: str,
    readout_layers: list[int] | None = None,
) -> AblationResult:
    """circuit-tracer's ``feature_intervention`` on the REAL model (per-layer transcoders).

    Adds ``m * clean * W_dec`` to each steered feature's own-layer MLP output; freezes
    attention patterns (all layers) when ``freeze_attention`` OR a constrained range is set
    (circuit-tracer couples these — a constrained range always freezes attention); pins MLP
    outputs to clean within ``[0, patch_end_layer]`` (and freezes LayerNorm if the range is
    the whole model) so within-range MLPs don't recompute; runs the real model after
    ``patch_end_layer``.
    """
    n_layers = len(transcoder)
    seq = input_ids.shape[1]
    device = input_ids.device

    if patch_end_layer is not None and not (l_max <= patch_end_layer < n_layers):
        raise ValueError(
            f"patch_end_layer must be in [{l_max}, {n_layers - 1}] "
            f"(>= last steered layer, < n_layers), got {patch_end_layer}"
        )
    ell = patch_end_layer  # None => propagate (pin nothing); else pin [0, ell]
    freeze_ln = ell is not None and ell >= n_layers - 1  # pinning all layers => direct effect
    # circuit-tracer couples these: a constrained range ALWAYS freezes the attention pattern
    # (its setup_intervention_with_freeze freezes hook_pattern whenever it runs, and it runs
    # for freeze_attention OR constrained_layers). So freeze_attention=False only takes effect
    # in the unconstrained (propagate) case.
    freeze_attn = freeze_attention or ell is not None

    # Per-layer decoder delta from the CLEAN feature activations (M convention).
    deltas: dict[int, Tensor] = {}
    for iv in interventions:
        feats = base.features[iv.layer]
        if feats.dim() == 3:
            feats = feats[0]
        dec = (
            transcoder.transcoders[iv.layer]
            ._get_decoder_vectors(torch.tensor([iv.feature_idx], device=device))[0]
            .float()
        )  # (d_model,)
        d = deltas.setdefault(
            iv.layer, torch.zeros(seq, dec.shape[0], device=device, dtype=torch.float32)
        )
        positions = [iv.position] if iv.position is not None else list(range(n_bos_tokens, seq))
        for p in positions:
            clean_act = feats[p, iv.feature_idx].float().item()
            d[p] += dec * (iv.target(clean_act) - clean_act)  # = m * clean * W_dec

    # Optional feature readout: capture the intervened forward's MLP inputs at the
    # requested layers so downstream feature activations (the paper's "% of baseline"
    # node annotations) can be measured on the SAME perturbed pass.
    readout_caps: dict[int, Tensor] = {}

    def _make_readout_hook(layer_idx: int) -> Any:
        def hook(_m: nn.Module, inp: Any, _out: Any) -> None:
            readout_caps[layer_idx] = inp[0].detach()

        return hook

    handles: list[Any] = []
    saved_attn: dict[int, Any] = {}
    try:
        if readout_layers:
            for i in sorted(set(readout_layers)):
                handles.append(
                    model.get_submodule(mlp_name_template.format(layer=i)).register_forward_hook(
                        _make_readout_hook(i)
                    )
                )
        if freeze_attn:
            for i in range(n_layers):
                attn_mod = model.get_submodule(attn_name_template.format(layer=i))
                saved_attn[i] = attn_mod.forward
                attn_mod.forward = _make_frozen_attn_forward(attn_mod, caps.attn_weights[i])
        if freeze_ln:
            for tmpl in layernorm_templates:
                for i in range(n_layers):
                    name = tmpl.format(layer=i)
                    handles.append(
                        model.get_submodule(name).register_forward_hook(
                            _make_frozen_rmsnorm_hook(name, caps.rmsnorm_scales)
                        )
                    )
            handles.append(
                model.get_submodule(final_norm_name).register_forward_hook(
                    _make_frozen_rmsnorm_hook(final_norm_name, caps.rmsnorm_scales)
                )
            )
        for i in range(n_layers):
            pin = ell is not None and i <= ell
            delta = deltas.get(i)
            if not pin and delta is None:
                continue
            handles.append(
                model.get_submodule(mlp_name_template.format(layer=i)).register_forward_hook(
                    _make_base_mlp_hook(pin, caps.mlp_outputs.get(i) if pin else None, delta)
                )
            )
        with torch.no_grad():
            out = model(input_ids)
        logits = out.logits
        intervened_logits = (logits[0] if logits.dim() == 3 else logits).detach()
    finally:
        for i, fwd in saved_attn.items():
            model.get_submodule(attn_name_template.format(layer=i)).forward = fwd
        for h in handles:
            h.remove()

    # Encode the captured perturbed MLP inputs -> intervened feature activations.
    ablated_features: dict[int, Tensor] = {}
    with torch.no_grad():
        for i, x in readout_caps.items():
            feats = transcoder.transcoders[i].encode(x)
            ablated_features[i] = (feats[0] if feats.dim() == 3 else feats).detach()

    base_logits = caps.original_logits
    base_logits = (base_logits[0] if base_logits.dim() == 3 else base_logits).detach()
    return AblationResult(
        baseline_logits=base_logits,
        ablated_logits=intervened_logits,
        baseline_features=base.features,
        ablated_features=ablated_features,
    )


def run_feature_intervention(
    model: nn.Module,
    transcoder: TranscoderSet | CrossLayerTranscoder,
    input_ids: Tensor,
    interventions: list[FeatureIntervention],
    *,
    patch_end_layer: int | None = None,
    freeze_attention: bool = True,
    n_bos_tokens: int = 1,
    readout_layers: list[int] | None = None,
    mlp_name_template: str = "model.layers.{layer}.mlp",
    output_module_template: str | None = None,
    attn_name_template: str = "model.layers.{layer}.self_attn",
    layernorm_templates: list[str] | None = None,
    final_norm_name: str = "model.norm",
) -> AblationResult:
    """Steer/clamp features on the REAL model and compare logits to the clean baseline.

    Faithful to circuit-tracer's ``feature_intervention`` (per-layer transcoders): adds the
    decoder delta ``m * clean * W_dec`` to each feature's *real* MLP output — no transcoder
    reconstruction, no error nodes.  **M convention**: new activation = ``(1 + m) * clean``
    (``m=0`` no change, ``-1`` ablate, ``-2`` flip).

    ``freeze_attention=True`` freezes attention **patterns** on every layer (V/O still
    respond); a constrained range forces this too.  ``patch_end_layer=L`` pins MLP outputs
    to their clean recorded values for layers ``[0, L]`` (within-range MLPs don't recompute)
    and runs the real model after L; if L is the last layer, LayerNorm is frozen too (pure
    direct/linear effect).  ``patch_end_layer=None`` pins nothing — the delta propagates
    through the real downstream MLPs + LayerNorm.  Defaults to the last steered layer (max
    downstream recompute); must be in ``[max steered layer, n_layers-1]``.

    ``readout_layers`` captures the intervened forward's MLP inputs at those layers and
    fills :attr:`AblationResult.ablated_features` with the perturbed feature activations —
    the paper's "% activation relative to baseline" node measurements.
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
        freeze_layernorm=True,
    )

    # Clean baseline. The local-replacement clean run reproduces the real residual stream
    # exactly (error nodes), so base.features are the real clean feature activations — used
    # as the steering targets' clean values.
    with torch.no_grad(), LocalReplacementModel(model, transcoder, caps, **common) as lm:
        base = lm.forward(input_ids)

    if not interventions:
        # No intervention → the baseline is just the real clean logits (consistent with
        # steering-base-model, which compares against the real model).
        clean_logits = caps.original_logits
        clean_logits = (clean_logits[0] if clean_logits.dim() == 3 else clean_logits).detach()
        return AblationResult(
            baseline_logits=clean_logits,
            ablated_logits=clean_logits,
            baseline_features=base.features,
            ablated_features=base.features,
        )

    l_max = max(iv.layer for iv in interventions)
    return _steer_base_model(
        model,
        transcoder,
        input_ids,
        interventions,
        base,
        caps,
        l_max=l_max,
        patch_end_layer=patch_end_layer,
        freeze_attention=freeze_attention,
        n_bos_tokens=n_bos_tokens,
        mlp_name_template=mlp_name_template,
        attn_name_template=attn_name_template,
        layernorm_templates=layernorm_templates,
        final_norm_name=final_norm_name,
        readout_layers=readout_layers,
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


def run_progressive_intervention(
    model: nn.Module,
    transcoder: TranscoderSet | CrossLayerTranscoder,
    input_ids: Tensor,
    ordered_interventions: list[FeatureIntervention],
    token_id: int,
    *,
    n_bos_tokens: int = 1,
    position: int = -1,
    patch_end_layer: int | None = None,
    freeze_attention: bool = True,
    mlp_name_template: str = "model.layers.{layer}.mlp",
    output_module_template: str | None = None,
    attn_name_template: str = "model.layers.{layer}.self_attn",
    layernorm_templates: list[str] | None = None,
    final_norm_name: str = "model.norm",
) -> ProgressiveAblationResult:
    """Cumulative steering curve: apply the first ``k`` of *ordered_interventions* for
    ``k = 0 … len`` and record the target token's logit + probability at each step.

    Delegates to :func:`run_feature_intervention` (the faithful real-model intervention).
    Per-layer transcoders only.
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    kw = dict(
        patch_end_layer=patch_end_layer,
        freeze_attention=freeze_attention,
        n_bos_tokens=n_bos_tokens,
        mlp_name_template=mlp_name_template,
        output_module_template=output_module_template,
        attn_name_template=attn_name_template,
        layernorm_templates=layernorm_templates,
        final_norm_name=final_norm_name,
    )

    logits_out: list[float] = []
    probs_out: list[float] = []

    def _record(row: Tensor) -> None:
        logits_out.append(row[token_id].item())
        probs_out.append(row.softmax(dim=-1)[token_id].item())

    for k in range(len(ordered_interventions) + 1):
        # k=0 → no interventions → run_feature_intervention returns the clean baseline.
        res = run_feature_intervention(
            model, transcoder, input_ids, ordered_interventions[:k], **kw
        )
        _record(res.ablated_logits[position])

    return ProgressiveAblationResult(
        n_ablated=list(range(len(ordered_interventions) + 1)),
        logits=logits_out,
        probs=probs_out,
        token_id=token_id,
        position=position,
    )


# ---------------------------------------------------------------------------
# Layer-range (patch end-layer) sweep
# ---------------------------------------------------------------------------


@dataclass
class LayerSweepResult:
    """Constrained-patching target logit/prob as a function of the patch END layer.

    ``end_layers[i]`` -> ``logits[i]`` / ``probs[i]`` for ``token_id`` at ``position``.
    ``end_layers[0]`` is the last steered layer (maximum downstream recompute); higher
    end layers freeze more of the model, so fewer layers recompute.  This is the curve
    the paper sweeps to choose the most-suppressive patching range.
    """

    end_layers: list[int]
    logits: list[float]
    probs: list[float]
    baseline_logit: float
    baseline_prob: float
    token_id: int
    position: int

    @property
    def delta_logits(self) -> list[float]:
        """``logit(end) - baseline_logit`` per end layer (negative = suppression)."""
        return [lg - self.baseline_logit for lg in self.logits]

    @property
    def delta_probs(self) -> list[float]:
        """``prob(end) - baseline_prob`` per end layer."""
        return [p - self.baseline_prob for p in self.probs]

    @property
    def best_end_layer(self) -> int:
        """The end layer with the largest logit *suppression* (most negative delta)."""
        i = min(range(len(self.logits)), key=lambda j: self.logits[j])
        return self.end_layers[i]


def sweep_patch_end_layer(
    model: nn.Module,
    transcoder: TranscoderSet | CrossLayerTranscoder,
    input_ids: Tensor,
    interventions: list[FeatureIntervention],
    token_id: int,
    *,
    n_bos_tokens: int = 1,
    position: int = -1,
    freeze_attention: bool = True,
    mlp_name_template: str = "model.layers.{layer}.mlp",
    output_module_template: str | None = None,
    attn_name_template: str = "model.layers.{layer}.self_attn",
    layernorm_templates: list[str] | None = None,
    final_norm_name: str = "model.norm",
) -> LayerSweepResult:
    """Sweep the **patch end layer** (the paper's layer-range knob).

    For each end layer from the last steered layer up to the final layer, runs
    :func:`run_feature_intervention` with that ``patch_end_layer`` and records the
    *token_id* logit + probability at *position*.  A higher end layer pins more of the
    model (fewer layers recompute); the paper picks the end layer that suppresses the
    target logit the most (:attr:`LayerSweepResult.best_end_layer`).  Per-layer
    transcoders only (Qwen3).
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if not _is_transcoder_set(transcoder):
        raise NotImplementedError(
            "patch-end-layer sweep is implemented for per-layer transcoders only"
        )
    if not interventions:
        raise ValueError("sweep_patch_end_layer requires at least one intervention")
    n_layers = len(transcoder)
    l_max = max(iv.layer for iv in interventions)

    kw = dict(
        freeze_attention=freeze_attention,
        n_bos_tokens=n_bos_tokens,
        mlp_name_template=mlp_name_template,
        output_module_template=output_module_template,
        attn_name_template=attn_name_template,
        layernorm_templates=layernorm_templates,
        final_norm_name=final_norm_name,
    )

    # Clean baseline (real clean logits) + per-end-layer intervened forwards.
    base_row = run_feature_intervention(model, transcoder, input_ids, [], **kw).baseline_logits[
        position
    ]
    end_layers = list(range(l_max, n_layers))
    logits_out: list[float] = []
    probs_out: list[float] = []
    for end in end_layers:
        row = run_feature_intervention(
            model, transcoder, input_ids, interventions, patch_end_layer=end, **kw
        ).ablated_logits[position]
        logits_out.append(row[token_id].item())
        probs_out.append(row.softmax(dim=-1)[token_id].item())

    return LayerSweepResult(
        end_layers=end_layers,
        logits=logits_out,
        probs=probs_out,
        baseline_logit=base_row[token_id].item(),
        baseline_prob=base_row.softmax(dim=-1)[token_id].item(),
        token_id=token_id,
        position=position,
    )
