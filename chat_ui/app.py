"""
HAVOC chat web UI — FastAPI + SSE.

This file no longer imports `transformers` or `bitsandbytes`. It points
directly at the HAVOC inference engine + refinement + orchestrator.

Endpoints:
    GET  /                       static HTML
    GET  /status                 model load state, VRAM, dimensions
    GET  /vram                   live VRAM usage
    POST /chat/stream            single-pass streaming (SSE)
    POST /chat/refine/stream     iterative refinement (SSE)
    POST /chat/orchestrate/stream  retrieval + tools + refinement (SSE)

To run:
    set HAVOC_CKPT=models\\checkpoints\\best.pt   (optional - default best.pt)
    python chat_ui/app.py

If no checkpoint is found, the app still serves the UI and reports
"no model loaded" so you can develop the front-end before HAVOC
finishes training.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import AsyncGenerator, List, Dict, Optional

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

# Add scripts/ to path for HAVOC modules
_HERE     = os.path.dirname(os.path.abspath(__file__))
_PROJ     = os.path.dirname(_HERE)
_SCRIPTS  = os.path.join(_PROJ, "scripts")
sys.path.insert(0, _SCRIPTS)

from inference   import InferenceEngine               # noqa: E402
from refinement  import RefinementEngine              # noqa: E402
from orchestrator import Orchestrator                  # noqa: E402

app = FastAPI(title="HAVOC Chat")


# ── Engine bootstrapping ──────────────────────────────────────────────────


CKPT_PATH = os.environ.get("HAVOC_CKPT") or os.path.join(_PROJ, "models", "checkpoints", "best.pt")
TOK_PATH  = os.environ.get("HAVOC_TOK")  or os.path.join(_PROJ, "models", "tokenizers", "havoc_bpe")

engine: InferenceEngine = InferenceEngine()
load_error: Optional[str] = None

if os.path.isfile(CKPT_PATH):
    try:
        meta = engine.load_model(CKPT_PATH, tokenizer_dir=TOK_PATH)
        print(f"Loaded HAVOC: {meta['n_params']:,} params from {CKPT_PATH}")
    except Exception as exc:
        load_error = f"{type(exc).__name__}: {exc}"
        print(f"  WARN: failed to load {CKPT_PATH}: {load_error}")
else:
    load_error = f"checkpoint not found at {CKPT_PATH} - train HAVOC first"
    print(f"  WARN: {load_error}")


# ── Request schemas ───────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    system_prompt:  str   = ""
    temperature:    float = 0.7
    top_p:          float = 0.9
    top_k:          int   = 40
    max_new_tokens: int   = 512
    cot:            bool  = False


class RefineRequest(BaseModel):
    user_message:        str
    system_prompt:       str   = ""
    max_passes:          int   = 10
    confidence_threshold: float = 0.85
    similarity_threshold: float = 0.9
    temperature:         float = 0.6
    top_k:               int   = 40
    top_p:               float = 0.9
    max_pass_tokens:     int   = 256
    max_final_tokens:    int   = 256


class OrchestrateRequest(BaseModel):
    user_message:       str
    system_prompt:      str   = ""
    enable_retrieval:   bool  = True
    enable_tools:       bool  = True
    enable_refinement:  bool  = True
    retrieval_top_k:    int   = 4
    max_new_tokens:     int   = 512
    temperature:        float = 0.7
    top_p:              float = 0.9
    top_k:              int   = 40


# ── Static / metadata ─────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def root():
    p = os.path.join(_HERE, "templates", "index.html")
    with open(p, encoding="utf-8") as f:
        return f.read()


@app.get("/status")
async def status():
    return {
        "loaded":     engine.loaded,
        "load_error": load_error,
        "ckpt_path":  CKPT_PATH if os.path.isfile(CKPT_PATH) else None,
        "meta":       engine.ckpt_meta,
    }


@app.get("/vram")
async def vram():
    if torch.cuda.is_available():
        used  = torch.cuda.memory_allocated(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return {"used_gb": used, "total_gb": total}
    return {"used_gb": 0.0, "total_gb": 0.0}


# ── Helpers ──────────────────────────────────────────────────────────────


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, default=str)}\n\n"


def _ensure_loaded():
    if not engine.loaded:
        return _sse({"type": "error", "content": load_error or "model not loaded"})
    return None


# ── Endpoints ────────────────────────────────────────────────────────────


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Single-pass streaming."""
    async def gen() -> AsyncGenerator[str, None]:
        err = _ensure_loaded()
        if err:
            yield err
            yield _sse({"type": "done"})
            return

        # Build prompt from messages: take the last user message, set system
        # prompt from req.system_prompt, ignore prior assistant turns for now
        # (chat_ui maintains its own history). The engine wraps in chat template.
        system_prompt = req.system_prompt or ""
        user_message  = ""
        for m in reversed(req.messages):
            if m.get("role") == "user":
                user_message = m.get("content", "")
                break
        if not user_message:
            yield _sse({"type": "error", "content": "no user message"})
            yield _sse({"type": "done"})
            return

        engine.set_system_prompt(system_prompt)
        loop = asyncio.get_running_loop()

        def gen_iter():
            return engine.generate_stream(
                prompt          = user_message,
                max_new_tokens  = req.max_new_tokens,
                temperature     = req.temperature,
                top_k           = req.top_k,
                top_p           = req.top_p,
                cot             = req.cot,
            )

        it = await loop.run_in_executor(None, gen_iter)

        def next_event():
            try:
                return next(it)
            except StopIteration:
                return None

        # CoT mode: split the stream on the literal "<|/think|>" decoded text.
        # Tokens before that boundary are reasoning; after, response.
        END_THINK = "<|/think|>"
        in_think = bool(req.cot)
        buf = ""
        if in_think:
            yield _sse({"type": "thinking_start"})

        while True:
            ev = await loop.run_in_executor(None, next_event)
            if ev is None:
                break
            tok, done, _stats = ev
            if tok:
                buf += tok
                while buf:
                    if in_think:
                        idx = buf.find(END_THINK)
                        if idx != -1:
                            pre = buf[:idx]
                            if pre:
                                yield _sse({"type": "thinking", "content": pre})
                            buf = buf[idx + len(END_THINK):]
                            in_think = False
                            yield _sse({"type": "thinking_end"})
                        else:
                            # Hold a partial-tag tail in case "<|/think|>" is split across chunks
                            partial = 0
                            for i in range(1, min(len(END_THINK), len(buf) + 1)):
                                if buf.endswith(END_THINK[:i]):
                                    partial = i
                            safe_len = len(buf) - partial
                            if safe_len > 0:
                                yield _sse({"type": "thinking", "content": buf[:safe_len]})
                                buf = buf[safe_len:]
                            break
                    else:
                        if buf:
                            yield _sse({"type": "response", "content": buf})
                        buf = ""
                        break
            if done:
                # Flush any held buffer
                if buf:
                    yield _sse({"type": "thinking" if in_think else "response", "content": buf})
                if in_think:
                    yield _sse({"type": "thinking_end"})
                break
        yield _sse({"type": "done"})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no",
                                      "Connection": "keep-alive"})


@app.post("/chat/refine/stream")
async def chat_refine(req: RefineRequest):
    """Iterative refinement streaming."""
    async def gen() -> AsyncGenerator[str, None]:
        err = _ensure_loaded()
        if err:
            yield err
            yield _sse({"type": "done"})
            return

        # Build a per-request RefinementEngine bound to our InferenceEngine
        from inference import _BareStreamingAdapter
        refiner = RefinementEngine(
            engine                = _BareStreamingAdapter(engine),
            max_passes            = req.max_passes,
            confidence_threshold  = req.confidence_threshold,
            similarity_threshold  = req.similarity_threshold,
            max_pass_tokens       = req.max_pass_tokens,
            max_final_tokens      = req.max_final_tokens,
            temperature           = req.temperature,
            top_k                 = req.top_k,
            top_p                 = req.top_p,
        )

        loop = asyncio.get_running_loop()
        stream = refiner.stream(req.user_message, system_prompt=req.system_prompt)

        def next_event():
            try:
                return next(stream)
            except StopIteration:
                return None

        while True:
            ev = await loop.run_in_executor(None, next_event)
            if ev is None:
                break
            yield _sse(ev)
            if ev.get("type") == "done":
                break

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no",
                                      "Connection": "keep-alive"})


@app.post("/chat/orchestrate/stream")
async def chat_orchestrate(req: OrchestrateRequest):
    """Full orchestration pipeline."""
    async def gen() -> AsyncGenerator[str, None]:
        err = _ensure_loaded()
        if err:
            yield err
            yield _sse({"type": "done"})
            return

        orch = Orchestrator.from_engine(engine)
        orch.enable_retrieval  = req.enable_retrieval
        orch.enable_tools      = req.enable_tools
        orch.enable_refinement = req.enable_refinement
        orch.retrieval_top_k   = req.retrieval_top_k

        loop = asyncio.get_running_loop()
        stream = orch.stream(
            req.user_message,
            system_prompt   = req.system_prompt,
            max_new_tokens  = req.max_new_tokens,
            temperature     = req.temperature,
            top_p           = req.top_p,
            top_k           = req.top_k,
        )

        def next_event():
            try:
                return next(stream)
            except StopIteration:
                return None

        while True:
            ev = await loop.run_in_executor(None, next_event)
            if ev is None:
                break
            yield _sse(ev)
            if ev.get("type") == "done":
                break

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no",
                                      "Connection": "keep-alive"})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
