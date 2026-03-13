"""
main.py — transCallMapping Refiner API  v3.0

Your role in the pipeline:
  RECEIVE  POST /refine  ← upstream STT API sends raw transcript
  PROCESS  3 LLM calls   (arabic refine / english refine / full analysis)
  RETURN   enriched JSON  → downstream post-call details API

Endpoints
─────────────────────────────────────────────────────────────────────────────
POST /refine              Blocking  — waits, returns one JSON response
POST /refine/stream       SSE       — streams tokens as text/event-stream
WS   /ws/refine           WebSocket — streams tokens as JSON messages
GET  /health

All three return the same output contract (see refiner.py docstring).
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator
from pathlib import Path
import sys
import os

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from dspy_refiner import run_pipeline, stream_pipeline

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "transCallMapping — Refiner API",
    description = (
        "Receives a raw call transcript, runs three LLM steps, "
        "and returns the fully enriched call data matching the post-call details schema.\n\n"
        "| Endpoint | Streaming | When to use |\n"
        "|---|---|---|\n"
        "| `POST /refine` | ❌ | Simple integrations |\n"
        "| `POST /refine/stream` | ✅ SSE | Browser, curl, Postman |\n"
        "| `WS /ws/refine` | ✅ WS | Real-time UI |\n"
    ),
    version = "3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── Request model ─────────────────────────────────────────────────────────────

class RefineRequest(BaseModel):
    """
    Sent by the upstream STT API after transcribing the call audio.
    """
    transcript:   str = Field(
        ...,
        description = "Raw transcript text from the STT service.",
        example     = "[00:00:01] ألو معك خدمه عملاء العربى",
    )
    context_info: str = Field(
        "",
        description = (
            "Optional call metadata to improve analysis accuracy.\n"
            "Recommended format:\n"
            "  Call ID: C-20240315-001\n"
            "  Agent Name: Ahmed Samir\n"
            "  Call Date: 2024-03-15 14:30\n"
            "  Call Duration: 4m 22s\n"
            "  Call Type: Inbound"
        ),
        example = "Call ID: C-001\nAgent Name: Ahmed Samir\nCall Date: 2024-03-15",
    )


# ── Response schema (documented, not enforced — summary only) ─────────────────

RESPONSE_DESCRIPTION = """
```json
{
  "original_transcription": "raw input",
  "context_info":           "...",
  "transcript_arabic":      "[00:00:01] Agent: أهلاً وسهلاً...",
  "transcript_english":     "[00:00:01] Agent: Hello, ELAraby customer service...",
  "main_subject":           "Customer inquiring about AC unit price",
  "call_outcome":           "Resolved | Unresolved | Escalated | Follow-up Needed",
  "issue_resolution":       "Agent provided pricing for Carrier and Midea units...",
  "call_summary":           "Customer called to ask about 2.25-ton inverter AC...",
  "keywords":               ["AC unit", "Carrier", "price inquiry", "inverter", "warranty"],
  "call_category":          "Inquiry | Complaint | Technical Support | Billing | Sales | Feedback",
  "service":                "AC Unit Sales",
  "agent_attitude":         "Friendly | Neutral | Rude",
  "customer_satisfaction":  "Satisfied | Neutral | Dissatisfied",
  "language":               "Arabic | English | Mixed",
  "call_score":             86,
  "score_breakdown": {
    "greeted_professionally":      {"result": "Pass", "note": "..."},
    "identified_customer_need":    {"result": "Pass", "note": "..."},
    "provided_accurate_info":      {"result": "Pass", "note": "..."},
    "maintained_professional_tone":{"result": "Pass", "note": "..."},
    "offered_complete_solution":   {"result": "Pass", "note": "..."},
    "confirmed_resolution":        {"result": "Fail", "note": "..."},
    "proper_closing":              {"result": "Pass", "note": "..."}
  }
}
```
"""


# ── Async bridge: blocking generator → async generator ───────────────────────

async def _iter_pipeline(
    transcript:   str,
    context_info: str,
) -> AsyncGenerator[dict, None]:
    """
    Runs stream_pipeline() (blocking, in a thread) and forwards each
    event dict into async context via an asyncio.Queue.
    """
    loop:  asyncio.AbstractEventLoop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict | None] = asyncio.Queue()

    def _producer() -> None:
        try:
            for event in stream_pipeline(transcript, context_info):
                loop.call_soon_threadsafe(queue.put_nowait, event)
        except Exception as exc:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"event": "error", "message": str(exc)},
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    future = loop.run_in_executor(None, _producer)

    try:
        while True:
            event = await queue.get()
            if event is None:
                break
            yield event
    finally:
        await asyncio.shield(future)


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 0 — Health
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["System"])
async def health():
    return {
        "status":  "ok",
        "service": "transCallMapping-refiner",
        "version": "3.0.0",
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 1 — Blocking HTTP POST /refine
# ══════════════════════════════════════════════════════════════════════════════

@app.post(
    "/refine",
    tags        = ["1 · HTTP Blocking"],
    summary     = "Refine transcript — blocking",
    description = (
        "Runs the full pipeline and waits until all three LLM steps finish, "
        "then returns one JSON response.\n\n"
        "**Typical wait time:** 30–90 seconds depending on transcript length.\n\n"
        "Use `/refine/stream` or `/ws/refine` for real-time token streaming.\n\n"
        + RESPONSE_DESCRIPTION
    ),
)
async def refine_blocking(req: RefineRequest):
    if not req.transcript.strip():
        raise HTTPException(422, "transcript must not be empty.")
    try:
        result = await asyncio.to_thread(
            run_pipeline, req.transcript, req.context_info
        )
    except Exception as exc:
        raise HTTPException(500, str(exc))
    return JSONResponse(result)


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 2 — SSE  POST /refine/stream
# ══════════════════════════════════════════════════════════════════════════════

@app.post(
    "/refine/stream",
    tags        = ["2 · SSE Streaming"],
    summary     = "Refine transcript — SSE token stream",
    description = (
        "Streams tokens as `text/event-stream` the moment Azure OpenAI generates them.\n\n"
        "### SSE message format\n"
        "Each line: `data: <json>\\n\\n`\n\n"
        "### Event types\n"
        "```\n"
        'data: {"event":"start",        "section":"arabic"}\n'
        'data: {"event":"token",        "section":"arabic",   "token":"أهلاً"}\n'
        'data: {"event":"section_done", "section":"arabic",   "text":"<full arabic>"}\n'
        'data: {"event":"start",        "section":"english"}\n'
        "... english tokens ...\n"
        'data: {"event":"start",        "section":"analysis"}\n'
        "... analysis JSON tokens ...\n"
        'data: {"event":"done", "transcript_arabic":"...", "call_score":86, ...}\n'
        "data: [DONE]\n"
        "```\n\n"
        "### Test with curl\n"
        "```bash\n"
        "curl -N -X POST http://localhost:8000/refine/stream \\\n"
        '  -H "Content-Type: application/json" \\\n'
        "  -d '{\"transcript\":\"...\",\"context_info\":\"\"}'\n"
        "```\n"
        "(`-N` disables curl buffering so tokens appear as they arrive)\n\n"
        "### Test with Postman\n"
        "POST → `http://localhost:8000/refine/stream` → Body → raw JSON → Send\n"
        "Watch the Response panel fill up token by token."
    ),
)
async def refine_sse(req: RefineRequest):
    if not req.transcript.strip():
        raise HTTPException(422, "transcript must not be empty.")

    async def _generate():
        async for event in _iter_pipeline(req.transcript, req.context_info):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _generate(),
        media_type = "text/event-stream",
        headers    = {
            "Cache-Control":     "no-cache",
            "Connection":        "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 3 — WebSocket  /ws/refine
# ══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/refine")
async def ws_refine(websocket: WebSocket):
    """
    WebSocket streaming — identical events to SSE but without the `data:` prefix.

    Client sends ONE JSON message to start:
        {"transcript": "...", "context_info": "..."}

    Server streams events until:
        {"event": "done", "transcript_arabic": "...", "call_score": 86, ...}
    or:
        {"event": "error", "message": "..."}

    ### Test with Postman
    1. New → WebSocket → `ws://localhost:8000/ws/refine` → Connect
    2. Paste JSON in message box → Send
    3. Watch messages arrive token by token in the Messages panel
    """
    await websocket.accept()

    try:
        raw = await websocket.receive_text()
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await websocket.send_text(json.dumps({
                "event":   "error",
                "message": 'Invalid JSON. Send: {"transcript":"...","context_info":"..."}',
            }))
            return

        transcript   = msg.get("transcript",   "").strip()
        context_info = msg.get("context_info", "")

        if not transcript:
            await websocket.send_text(json.dumps({
                "event": "error", "message": "transcript must not be empty.",
            }))
            return

        async for event in _iter_pipeline(transcript, context_info):
            await websocket.send_text(json.dumps(event, ensure_ascii=False))

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_text(
                json.dumps({"event": "error", "message": str(exc)})
            )
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# DSPy ENDPOINTS  —  /dspy/*
# ══════════════════════════════════════════════════════════════════════════════

from dspy_module.optimizer import (
    optimize_program  as _dspy_optimize,
    load_results      as _dspy_load_results,
    get_best_compiled as _dspy_best_compiled,
)
from dspy_module.programs  import ArabicRefiner, EnglishRefiner, CallAnalyser
from dspy_module.metrics   import arabic_metric, english_metric, analysis_metric
from dspy_module.trainset  import get_arabic_trainset, get_english_trainset, get_analysis_trainset
from dspy_module.optimizer import evaluate_program as _dspy_evaluate


class DSPyOptimizeRequest(BaseModel):
    program:    str = Field("arabic",    description="arabic | english | analysis | all")
    optimizer:  str = Field("bootstrap", description="bootstrap | random_search | mipro")
    demos:      int = Field(3,           description="Max bootstrapped demos (1-5)")
    candidates: int = Field(8,           description="Candidate programs for random_search/mipro")


# ── GET /dspy/status ──────────────────────────────────────────────────────────

@app.get("/dspy/status", tags=["DSPy"])
async def dspy_status():
    """
    Show which DSPy programs are active and whether compiled versions exist.
    """
    from dspy_module.lm_setup import configure_dspy
    import dspy
    configure_dspy()

    programs = {}
    for name in ["arabic", "english", "analysis"]:
        best = _dspy_best_compiled(name)
        from pathlib import Path
        programs[name] = {
            "compiled_exists": best is not None and Path(best).exists() if best else False,
            "compiled_path":   best,
        }

    results = _dspy_load_results()
    latest  = {}
    for r in results:
        prog = r.get("program")
        if prog not in latest or r["timestamp"] > latest[prog]["timestamp"]:
            latest[prog] = r

    return {
        "dspy_version": dspy.__version__,
        "programs":     programs,
        "latest_scores": {
            prog: {
                "before":      r.get("before",{}).get("avg_score"),
                "after":       r.get("after", {}).get("avg_score"),
                "improvement": r.get("improvement"),
                "optimizer":   r.get("optimizer"),
                "timestamp":   r.get("timestamp"),
            }
            for prog, r in latest.items()
        },
    }


# ── POST /dspy/optimize ───────────────────────────────────────────────────────

@app.post("/dspy/optimize", tags=["DSPy"])
async def dspy_optimize(req: DSPyOptimizeRequest):
    """
    Run DSPy prompt optimization.

    **What it does:**
    1. Takes your training examples
    2. Runs the selected optimizer (BootstrapFewShot / MIPROv2)
    3. DSPy finds the best few-shot demos and/or instruction prefix
    4. Saves the compiled program to `dspy_module/compiled/`
    5. Returns before/after scores

    **Optimizers:**
    - `bootstrap` — Fast, no extra LLM calls for optimization itself
    - `random_search` — Tries multiple random seeds, picks best (slower)
    - `mipro` — Most powerful, generates candidate instructions (costs more)

    ⚠️ Makes real Azure OpenAI API calls during optimization.
    """
    programs = ["arabic","english","analysis"] if req.program == "all" else [req.program]

    results = {}
    for prog in programs:
        try:
            result = await asyncio.to_thread(
                _dspy_optimize,
                program_name           = prog,
                optimizer              = req.optimizer,
                max_bootstrapped_demos = req.demos,
                num_candidates         = req.candidates,
                verbose                = False,
            )
            results[prog] = result
        except Exception as exc:
            results[prog] = {"error": str(exc)}

    return {
        "optimizer": req.optimizer,
        "results":   results,
        "summary": {
            prog: {
                "before":      r.get("before",{}).get("avg_score"),
                "after":       r.get("after", {}).get("avg_score"),
                "improvement": r.get("improvement"),
            }
            for prog, r in results.items() if "error" not in r
        },
    }


# ── GET /dspy/results ─────────────────────────────────────────────────────────

@app.get("/dspy/results", tags=["DSPy"])
async def dspy_results():
    """Return all DSPy optimization run history."""
    results = _dspy_load_results()
    if not results:
        return {"message": "No optimization runs yet. POST /dspy/optimize to start.", "runs": []}
    return {
        "total_runs": len(results),
        "runs": [
            {
                "program":     r.get("program"),
                "optimizer":   r.get("optimizer"),
                "timestamp":   r.get("timestamp"),
                "before":      r.get("before",{}).get("avg_score"),
                "after":       r.get("after", {}).get("avg_score"),
                "improvement": r.get("improvement"),
                "elapsed_s":   r.get("elapsed_s"),
            }
            for r in results
        ],
    }


# ── POST /dspy/score ──────────────────────────────────────────────────────────

class DSPyScoreRequest(BaseModel):
    program:        str = Field(..., description="arabic | english | analysis")
    raw_transcript: str = Field(..., description="Original input transcript")
    output:         dict = Field(..., description="DSPy prediction output fields")


@app.post("/dspy/score", tags=["DSPy"])
async def dspy_score(req: DSPyScoreRequest):
    """
    Score a DSPy prediction using the built-in metrics.
    No API calls — runs in milliseconds.
    """
    METRICS = {
        "arabic":   arabic_metric,
        "english":  english_metric,
        "analysis": analysis_metric,
    }
    if req.program not in METRICS:
        raise HTTPException(400, f"Unknown program. Choose: {list(METRICS)}")

    metric_fn = METRICS[req.program]

    # Build fake example + pred for the metric
    class FakeEx:
        def __init__(self, d): self.__dict__.update(d)
    class FakePred:
        def __init__(self, d): self.__dict__.update(d)

    example = FakeEx({"raw_transcript": req.raw_transcript})
    pred    = FakePred(req.output)
    score   = metric_fn(example, pred)

    return {
        "program": req.program,
        "score":   score,
        "label":   "Good" if score >= 0.75 else "Needs improvement" if score >= 0.5 else "Poor",
    }


# ── POST /refine/dspy — use DSPy pipeline instead of raw LLM ─────────────────

@app.post("/refine/dspy", tags=["DSPy"])
async def refine_with_dspy(req: RefineRequest):
    """
    Run the pipeline using DSPy programs (compiled if available).

    Same output contract as POST /refine but uses DSPy's
    optimized prompts under the hood.

    Check GET /dspy/status to see if compiled programs are active.
    """
    if not req.transcript.strip():
        raise HTTPException(422, "transcript must not be empty.")

    from app.dspy_refiner import run_pipeline as dspy_run
    try:
        result = await asyncio.to_thread(dspy_run, req.transcript, req.context_info)
    except Exception as exc:
        raise HTTPException(500, str(exc))
    return JSONResponse(result)


if __name__ == "__main__":
    # Quick test
    import sys
    import argparse
    import uvicorn
    from pathlib import Path

    uvicorn.run(app, host="0.0.0.0", port=8000)
    if len(sys.argv) != 2:
        print("Usage: python main.py <transcript.txt>")
        sys.exit(1)
    transcript = Path(sys.argv[1]).read_text(encoding="utf-8")
    context = "Test call between customer and agent at Miraco Company."
    for event in stream_pipeline(transcript, context):
        print(event)


    # uvicorn main:app --reload
    # uvicorn main:app --reload --port 8000

    # run dspy_main.py transcript through the DSPy pipeline and print the result:
    # from app.dspy_refiner import run_pipeline
    # result = run_pipeline("ألو معك خدمه عملاء العربى", "Call ID: C-001\nAgent Name: Ahmed Samir\nCall Date: 2024-03-15")
    # print(result)