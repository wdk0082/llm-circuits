"""Engines behind the interactive server.

``RealEngine`` holds a Qwen3 model + its transcoders resident on the GPU and wraps
the already-validated build/steer/sweep functions.  ``MockEngine`` mirrors the same
interface with canned data so the UI and API can be developed/tested on CPU.
Use :func:`get_engine` to pick one (env ``LLM_CIRCUITS_SERVE_MOCK`` / ``_REAL``).
"""

from __future__ import annotations

import gc
import json
import os
import threading

# Reduce CUDA fragmentation across load/free cycles (set before torch initialises CUDA).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from llm_circuits.circuits.graph_explorer import render_graph_explorer_html_str
from llm_circuits.logging import get_logger
from llm_circuits.settings import artifacts_dir
from llm_circuits.transcoders.registry import get_spec, list_registry

log = get_logger(__name__)

# Qwen3 layer counts (mock + display); RealEngine reports the true value from the model.
_QWEN3_LAYERS = {"0.6b": 28, "1.7b": 28, "4b": 36, "8b": 36, "14b": 40}


def qwen3_sizes() -> list[str]:
    """Registered Qwen3 sizes (e.g. ``["0.6b", "1.7b", "4b", "8b", "14b"]``)."""
    return [k.split("-", 1)[1] for k, _ in list_registry() if k.startswith("qwen3-")]


def _known_size(size: str) -> bool:
    return f"qwen3-{size}" in {k for k, _ in list_registry()}


class BaseEngine:
    mock = False

    def __init__(self) -> None:
        self.loaded_size: str | None = None
        self.n_layers: int | None = None

    def available_sizes(self) -> list[str]:
        return qwen3_sizes()

    # -- subclasses implement these -------------------------------------------
    def load(self, size: str) -> None:
        raise NotImplementedError

    def build(self, req) -> dict:
        raise NotImplementedError

    def reprune(self, req) -> dict:
        raise NotImplementedError

    def steer(self, req) -> dict:
        raise NotImplementedError

    def sweep(self, req) -> dict:
        raise NotImplementedError


class MockEngine(BaseEngine):
    """Canned responses (no model) so the frontend + API work on CPU."""

    mock = True

    def load(self, size: str) -> None:
        if not _known_size(size):
            raise ValueError(f"unknown size {size!r}")
        self.loaded_size = size
        self.n_layers = _QWEN3_LAYERS.get(size, 36)

    def _saved_graph(self) -> dict:
        out_dir = artifacts_dir() / "addition_circuit"
        cands = sorted(out_dir.glob("addition_graph_qwen3-*plus*.json"))
        if cands:
            return json.loads(cands[0].read_text())
        # minimal synthetic graph if nothing is built yet
        return {
            "prompt": "mock",
            "answer_token": "5",
            "tokens": ["the", "sum"],
            "logit_token_strs": {"5": "5"},
            "nodes": [
                {"node_type": "embedding", "layer": -1, "position": 0, "activation": 1.0},
                {
                    "node_type": "feature",
                    "layer": 10,
                    "position": 1,
                    "feature_idx": 7,
                    "activation": 2.0,
                    "label": {"top_logits": ["five"], "bottom_logits": []},
                },
                {
                    "node_type": "logit",
                    "layer": 28,
                    "position": 1,
                    "token_id": 5,
                    "activation": 3.0,
                },
            ],
            "edges": [
                {"source": 0, "target": 1, "weight": 1.0},
                {"source": 1, "target": 2, "weight": 2.0},
            ],
        }

    def build(self, req) -> dict:
        if self.loaded_size is None:
            raise RuntimeError("load a model first")
        self._built = True
        return self._render(req.text[:60] or "graph")

    def reprune(self, req) -> dict:
        if not getattr(self, "_built", False):
            raise RuntimeError("build a graph first")
        return self._render("reprune")

    def _render(self, title: str) -> dict:
        d = self._saved_graph()
        html = render_graph_explorer_html_str(d, title=title)
        nfeat = sum(1 for n in d["nodes"] if n["node_type"] == "feature")
        return {"html": html, "n_nodes": len(d["nodes"]), "n_feature_nodes": nfeat}

    def steer(self, req) -> dict:
        # Deterministic, M-dependent fake distribution so the slider visibly moves.
        base = [("5", 5, 0.80), ("6", 6, 0.10), ("4", 4, 0.05), ("3", 3, 0.03)]
        shift = max(0.0, 1.0 - abs(req.m) / 2.0)  # M=0 -> unchanged, far -> flattened
        steered = []
        for tok, tid, p in base:
            np = p * shift if tok == "5" else p + (0.8 * p) * (1 - shift)
            steered.append({"token": tok, "token_id": tid, "prob": round(np, 4)})
        return {
            "baseline": [{"token": t, "token_id": i, "prob": p} for t, i, p in base],
            "steered": steered,
            "target_token": "5",
            "target_delta_logit": round(-8.0 * (1 - shift), 3),
        }

    def sweep(self, req) -> dict:
        l_max = max((n.layer for n in req.nodes), default=10)
        n = self.n_layers or 36
        ends = list(range(l_max, n))
        # synthetic dip then recovery
        dl = [round(-20.0 + 3.0 * abs(i - 2), 2) for i in range(len(ends))]
        probs = [round(max(0.02, 0.9 + d / 40.0), 3) for d in dl]
        best = ends[dl.index(min(dl))] if ends else l_max
        return {
            "end_layers": ends,
            "delta_logits": dl,
            "probs": probs,
            "best_end_layer": best,
            "target_token": "5",
        }


class RealEngine(BaseEngine):
    """Model-backed engine (GPU). Validated on an A100 node."""

    # Refuse builds whose full edge matrix would exceed this many feature targets: the
    # dense N*N prune/influence matrix grows as N^2, so ~40k is the practical ceiling on
    # an 80GB GPU + ~1TB host (243k wedged the GPU). no-cap/influence work below this.
    MAX_TARGETS_GUARD = 40_000

    def __init__(self) -> None:
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.tc = None
        self._ctx: dict | None = None  # cached build context (input_ids, n_bos, tokens)
        self._load_lock = threading.Lock()  # serialize loads (no concurrent stacking)
        self._build_lock = threading.Lock()  # serialize builds (one model, no concurrency)

    def _free(self) -> None:
        """Release the resident model/transcoder so a reload doesn't stack GPU memory."""
        import torch

        self.model = self.tokenizer = self.tc = self._ctx = None
        self.loaded_size = self.n_layers = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load(self, size: str) -> None:
        import torch

        from llm_circuits.models.qwen3 import load_qwen3
        from llm_circuits.settings import default_device
        from llm_circuits.transcoders.circuit_tracer_loader import load_transcoder

        if not _known_size(size):
            raise ValueError(f"unknown size {size!r}")
        if not self._load_lock.acquire(blocking=False):
            raise RuntimeError("a model load is already in progress")
        key = f"qwen3-{size}"
        try:
            self._free()  # drop the previous model first (switching sizes must not stack)
            device = default_device()
            dtype = torch.bfloat16
            log.info("Loading Qwen3-%s ...", size)
            self.model, self.tokenizer = load_qwen3(size, dtype_str="bf16", device_map=device)
            self.model.eval()
            try:
                self.tc = load_transcoder(
                    key, device=device, dtype=dtype, lazy_decoder=False
                ).transcoder
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.tc = load_transcoder(
                    key, device=device, dtype=dtype, lazy_decoder=True
                ).transcoder
            self.loaded_size = size
            self.n_layers = len(self.tc)
            self._ctx = None
        except Exception:
            self._free()  # release partial allocations so the server stays usable
            raise
        finally:
            self._load_lock.release()

    def _tokenize(self, text: str, use_chat: bool):
        if use_chat:
            from llm_circuits.instrumentation.chat import prepare_messages

            messages, _, tkw = prepare_messages(text, "qwen3", enable_thinking=False)
            ids = self.tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True, **tkw
            )
            n_bos = 1
        else:
            ids = self.tokenizer(text, return_tensors="pt").input_ids
            n_bos = 0
        return ids.to(self.model.device), n_bos

    def build(self, req) -> dict:
        """Expensive step: model forward + autograd edges. Caches the *raw* (unpruned)
        graph so threshold changes can re-prune via :meth:`reprune` without the model."""
        if self.model is None:
            raise RuntimeError("load a model first")
        import torch

        from llm_circuits.circuits.attribution_graph import build_attribution_graph

        # One model, no concurrency: reject overlapping builds cleanly instead of running
        # two autograd passes on the same graph (which corrupts state / strands GPU memory).
        if not self._build_lock.acquire(blocking=False):
            raise RuntimeError("a build is already in progress; wait for it to finish")
        try:
            input_ids, n_bos = self._tokenize(req.text, req.use_chat)
            tokens = [self.tokenizer.decode(t) for t in input_ids[0]]
            with torch.no_grad():
                answer_id = int(self.model(input_ids).logits[0, -1].argmax().item())

            # build_attribution_graph uses torch.autograd.grad internally -> NOT no_grad.
            # "influence" mode = circuit-tracer's dynamic selection: attribute features in
            # influence order (never materialising the full edge matrix), capped to
            # max_feature_nodes by influence. "activation" mode caps cheaply by |activation|
            # before edges are computed. max_feature_nodes is the cap in BOTH modes.
            influence_cap = getattr(req, "node_selection", "activation") == "influence"
            graph = build_attribution_graph(
                self.model,
                self.tc,
                input_ids,
                n_bos_tokens=n_bos,
                max_feature_targets=req.max_feature_targets,
                max_feature_nodes=req.max_feature_nodes,
                feature_selection="influence_ranked" if influence_cap else "all",
                # Safety backstop: a no-cap build (or influence with no cap) on a very dense
                # prompt produces 100k+ targets -> the dense N*N prune matrix would OOM.
                # Fail fast (-> 400 with a helpful message) instead of wedging the device.
                max_targets_guard=self.MAX_TARGETS_GUARD,
            )
            logit_token_strs = {
                str(n.token_id): self.tokenizer.decode(n.token_id)
                for n in graph.nodes
                if n.node_type == "logit"
            }
            self._ctx = {
                "input_ids": input_ids,
                "n_bos": n_bos,
                "tokens": tokens,
                "raw_graph": graph,  # cached for fast re-pruning
                "answer_str": self.tokenizer.decode(answer_id),
                "logit_token_strs": logit_token_strs,
                "prompt": req.text,
                "repo": get_spec(f"qwen3-{self.loaded_size}").transcoder_repo,
                "label_cache": {},  # (layer, feature_idx) -> label, reused across reprunes
                "ex_cache": {},
            }
            return self._prune_and_render(req.node_threshold, req.edge_threshold)
        except Exception:
            # Free GPU memory stranded by an aborted/failed build (guard, OOM, ...) so the
            # next build/load starts clean rather than on a near-full device.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        finally:
            self._build_lock.release()

    def reprune(self, req) -> dict:
        """Cheap step: re-prune the cached raw graph at new thresholds (no model)."""
        if not self._ctx or "raw_graph" not in self._ctx:
            raise RuntimeError("build a graph first")
        return self._prune_and_render(req.node_threshold, req.edge_threshold)

    def _prune_and_render(self, node_threshold: float, edge_threshold: float) -> dict:
        from llm_circuits.circuits.graph_pruning import graph_to_dict, prune_graph
        from llm_circuits.transcoders.feature_labels import (
            load_feature_examples,
            load_feature_labels,
        )

        c = self._ctx
        pruned = prune_graph(
            c["raw_graph"], node_threshold=node_threshold, edge_threshold=edge_threshold
        )
        pg = pruned.graph

        # Fetch labels/examples only for surviving features not already cached this build.
        lc, ec = c["label_cache"], c["ex_cache"]
        need: dict[int, list[int]] = {}
        for nd in pg.nodes:
            if nd.node_type == "feature" and (nd.layer, nd.feature_idx) not in lc:
                need.setdefault(nd.layer, []).append(nd.feature_idx)
        for lyr, idxs in need.items():
            labs = load_feature_labels(c["repo"], lyr, idxs)
            exs = load_feature_examples(c["repo"], lyr, idxs, n_per_quantile=5)
            for fid in idxs:  # cache misses too, so we don't re-request them
                lc[(lyr, fid)] = labs[fid].to_dict() if fid in labs else {}
                ec[(lyr, fid)] = exs.get(fid, [])
        for nd in pg.nodes:
            if nd.node_type == "feature":
                merged = dict(lc.get((nd.layer, nd.feature_idx)) or {})
                merged["examples"] = ec.get((nd.layer, nd.feature_idx), [])
                nd.label = merged

        d = graph_to_dict(
            pg,
            influence_scores=pruned.influence_scores,
            prompt=c["prompt"],
            model=f"qwen3-{self.loaded_size}",
            tokens=c["tokens"],
            n_bos_tokens=c["n_bos"],
            logit_token_strs=c["logit_token_strs"],
            answer_token=c["answer_str"],
        )
        html = render_graph_explorer_html_str(d, title=(c["prompt"][:60] or "graph"))
        nfeat = sum(1 for n in d["nodes"] if n["node_type"] == "feature")
        return {"html": html, "n_nodes": len(d["nodes"]), "n_feature_nodes": nfeat}

    def _interventions(self, nodes, m):
        from llm_circuits.circuits.interventions import FeatureIntervention

        return [
            FeatureIntervention(n.layer, n.feature_idx, position=n.position, m=m) for n in nodes
        ]

    def _top(self, logits_row, k: int) -> list[dict]:
        import torch

        probs = torch.softmax(logits_row.float(), dim=-1)
        vals, idxs = probs.topk(k)
        return [
            {
                "token": self.tokenizer.decode([int(i)]),
                "token_id": int(i),
                "prob": round(float(v), 5),
            }
            for v, i in zip(vals.tolist(), idxs.tolist(), strict=False)
        ]

    def steer(self, req) -> dict:
        if self._ctx is None:
            raise RuntimeError("build a graph first")
        from llm_circuits.circuits.interventions import run_feature_intervention

        ivs = self._interventions(req.nodes, req.m)
        res = run_feature_intervention(
            self.model,
            self.tc,
            self._ctx["input_ids"],
            ivs,
            freeze_attention=req.freeze_attention,
            patch_end_layer=req.patch_end_layer,
            n_bos_tokens=self._ctx["n_bos"],
        )
        base_row, steer_row = res.baseline_logits[-1], res.ablated_logits[-1]
        tgt_id = int(base_row.argmax())
        return {
            "baseline": self._top(base_row, req.top_k),
            "steered": self._top(steer_row, req.top_k),
            "target_token": self.tokenizer.decode([tgt_id]),
            "target_delta_logit": round(float(steer_row[tgt_id] - base_row[tgt_id]), 4),
        }

    def sweep(self, req) -> dict:
        if self._ctx is None:
            raise RuntimeError("build a graph first")
        from llm_circuits.circuits.interventions import (
            run_feature_intervention,
            sweep_patch_end_layer,
        )

        ivs = self._interventions(req.nodes, req.m)
        clean = run_feature_intervention(
            self.model, self.tc, self._ctx["input_ids"], [], n_bos_tokens=self._ctx["n_bos"]
        )
        tgt_id = int(clean.baseline_logits[-1].argmax())
        res = sweep_patch_end_layer(
            self.model,
            self.tc,
            self._ctx["input_ids"],
            ivs,
            tgt_id,
            freeze_attention=req.freeze_attention,
            n_bos_tokens=self._ctx["n_bos"],
        )
        return {
            "end_layers": res.end_layers,
            "delta_logits": [round(x, 4) for x in res.delta_logits],
            "probs": [round(x, 5) for x in res.probs],
            "best_end_layer": res.best_end_layer,
            "target_token": self.tokenizer.decode([tgt_id]),
        }


def get_engine() -> BaseEngine:
    """Pick the engine: mock on CPU / when forced, real on GPU."""
    if os.environ.get("LLM_CIRCUITS_SERVE_MOCK"):
        log.info("Serve: MockEngine (forced by env)")
        return MockEngine()
    if os.environ.get("LLM_CIRCUITS_SERVE_REAL"):
        return RealEngine()
    try:
        import torch

        if torch.cuda.is_available():
            return RealEngine()
    except Exception:
        pass
    log.info("Serve: MockEngine (no CUDA)")
    return MockEngine()
