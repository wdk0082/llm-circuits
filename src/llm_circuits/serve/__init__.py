"""Interactive attribution-graph + steering server (FastAPI).

A thin live wrapper around the existing circuit toolkit: it holds a Qwen3 model +
its transcoders resident on the GPU and exposes a small JSON API so the browser can
(i) pick a model size, (ii) type a sentence, (iii) build its attribution graph,
(iv) group nodes into supernodes (client-side), and (v) steer those supernodes.

The model-backed :class:`~llm_circuits.serve.engine.RealEngine` runs on a GPU node;
a :class:`~llm_circuits.serve.engine.MockEngine` (selected on CPU or via
``LLM_CIRCUITS_SERVE_MOCK=1``) serves canned graphs/steers so the whole UI and API
can be developed and tested without a model.
"""

from __future__ import annotations
