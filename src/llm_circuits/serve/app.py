"""FastAPI app for the interactive attribution-graph + steering tool.

Run (GPU node):   uv run --group serve uvicorn llm_circuits.serve.app:app --port 8000
Dev (CPU, mock):  LLM_CIRCUITS_SERVE_MOCK=1 uv run --group serve uvicorn llm_circuits.serve.app:app --port 8000

The browser reaches it via VS Code's automatic port-forwarding (bind 127.0.0.1).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from llm_circuits.serve.engine import BaseEngine, get_engine
from llm_circuits.serve.schemas import (
    BuildRequest,
    BuildResponse,
    LoadRequest,
    ModelsResponse,
    RepruneRequest,
    SteerRequest,
    SteerResponse,
    SweepRequest,
    SweepResponse,
)

_STATIC = Path(__file__).parent / "static"

app = FastAPI(title="llm-circuits interactive")
_engine: BaseEngine = get_engine()


def set_engine(engine: BaseEngine) -> None:
    """Swap the engine (used by tests to force the mock)."""
    global _engine
    _engine = engine


@app.get("/api/models", response_model=ModelsResponse)
def api_models() -> ModelsResponse:
    return ModelsResponse(
        sizes=_engine.available_sizes(),
        loaded=_engine.loaded_size,
        n_layers=_engine.n_layers,
        mock=_engine.mock,
    )


@app.post("/api/load", response_model=ModelsResponse)
def api_load(req: LoadRequest) -> ModelsResponse:
    try:
        _engine.load(req.size)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:  # OOM, CUDA, IO -> readable message instead of bare 500
        raise HTTPException(status_code=500, detail=f"load failed: {exc}") from exc
    return ModelsResponse(
        sizes=_engine.available_sizes(),
        loaded=_engine.loaded_size,
        n_layers=_engine.n_layers,
        mock=_engine.mock,
    )


@app.post("/api/build", response_model=BuildResponse)
def api_build(req: BuildRequest) -> BuildResponse:
    try:
        return BuildResponse(**_engine.build(req))
    except ValueError as exc:  # too-dense prompt guard -> actionable message
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:  # not loaded / build in progress / CUDA OOM
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:  # readable message instead of a bare 500
        raise HTTPException(status_code=500, detail=f"build failed: {exc}") from exc


@app.post("/api/reprune", response_model=BuildResponse)
def api_reprune(req: RepruneRequest) -> BuildResponse:
    try:
        return BuildResponse(**_engine.reprune(req))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/steer", response_model=SteerResponse)
def api_steer(req: SteerRequest) -> SteerResponse:
    try:
        return SteerResponse(**_engine.steer(req))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/sweep", response_model=SweepResponse)
def api_sweep(req: SweepRequest) -> SweepResponse:
    try:
        return SweepResponse(**_engine.sweep(req))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


if _STATIC.exists():
    app.mount("/static", StaticFiles(directory=_STATIC), name="static")
