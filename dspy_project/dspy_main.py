"""
dspy_main.py — FastAPI server for transCallMapping refiner.

Run:
  cd dspy_project
  uvicorn dspy_main:app --reload --port 8000

Endpoints:
  GET  /health
  POST /refine          blocking
  POST /refine/stream   SSE token stream
  WS   /ws/refine       WebSocket token stream
  POST /refine/dspy     DSPy compiled pipeline
  GET  /dspy/status
  POST /dspy/optimize
  GET  /dspy/results
  POST /dspy/score
"""
from __future__ import annotations
import asyncio, json
from typing import AsyncGenerator
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

import lm_setup          # triggers dspy.configure() once at module load (main thread)
from dspy_refiner import run_pipeline, stream_pipeline

app = FastAPI(
    title="transCallMapping — Refiner API",
    version="3.0.0",
    description=(
        "Receives a raw call transcript, runs three LLM steps, "
        "returns fully enriched call data.\n\n"
        "| Endpoint | Streaming |\n|---|---|\n"
        "| `POST /refine` | ❌ |\n"
        "| `POST /refine/stream` | ✅ SSE |\n"
        "| `WS /ws/refine` | ✅ WS |\n"
        "| `POST /refine/dspy` | ❌ (DSPy compiled) |\n"
    ),
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class RefineRequest(BaseModel):
    transcript:   str = Field(..., description="Raw transcript text from STT service")
    context_info: str = Field("",  description="Optional metadata: Call ID, Agent Name, Date…")


# ── Async bridge: blocking generator → async generator ────────────────────────
async def _iter_pipeline(transcript: str, context_info: str) -> AsyncGenerator[dict, None]:
    loop  = asyncio.get_running_loop()
    queue: asyncio.Queue[dict | None] = asyncio.Queue()

    def _producer():
# /*************  ✨ Windsurf Command ⭐  *************/
#     """
#     Runs stream_pipeline in a separate thread and puts each event into the queue.
#     If an exception occurs, it is put into the queue as a dict with event='error' and message=str(exc).
#     Finally, None is put into the queue to signal that the producer is done.
#     """
# /*******  b72a790b-0c5e-4ae9-9ae9-9c40887bfc94  *******/    
        try:
            for event in stream_pipeline(transcript, context_info):
                loop.call_soon_threadsafe(queue.put_nowait, event)
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, {"event":"error","message":str(exc)})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    future = loop.run_in_executor(None, _producer)
    try:
        while True:
            event = await queue.get()
            if event is None: break
            yield event
    finally:
        await asyncio.shield(future)


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "service": "transCallMapping-refiner", "version": "3.0.0"}


# ── Blocking HTTP ─────────────────────────────────────────────────────────────
@app.post("/refine", tags=["Refiner"])
async def refine_blocking(req: RefineRequest):
    if not req.transcript.strip():
        raise HTTPException(422, "transcript must not be empty.")
    try:
        result = await asyncio.to_thread(run_pipeline, req.transcript, req.context_info)
    except Exception as exc:
        raise HTTPException(500, str(exc))
    return JSONResponse(result)


# ── SSE streaming ─────────────────────────────────────────────────────────────
@app.post("/refine/stream", tags=["Refiner"])
async def refine_sse(req: RefineRequest):
    if not req.transcript.strip():
        raise HTTPException(422, "transcript must not be empty.")

    async def _generate():
        async for event in _iter_pipeline(req.transcript, req.context_info):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


# ── WebSocket streaming ───────────────────────────────────────────────────────
@app.websocket("/ws/refine")
async def ws_refine(websocket: WebSocket):
    await websocket.accept()
    try:
        raw = await websocket.receive_text()
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await websocket.send_text(json.dumps({"event":"error","message":'Invalid JSON. Send: {"transcript":"...","context_info":"..."}'}))
            return

        transcript   = msg.get("transcript",   "").strip()
        context_info = msg.get("context_info", "")

        if not transcript:
            await websocket.send_text(json.dumps({"event":"error","message":"transcript must not be empty."}))
            return

        async for event in _iter_pipeline(transcript, context_info):
            await websocket.send_text(json.dumps(event, ensure_ascii=False))

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_text(json.dumps({"event":"error","message":str(exc)}))
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ── DSPy endpoints ────────────────────────────────────────────────────────────
from optimizer import (
    optimize_program  as _dspy_optimize,
    load_results      as _dspy_load_results,
    get_best_compiled as _dspy_best_compiled,
)
from metrics import arabic_metric, english_metric, analysis_metric


class DSPyOptimizeRequest(BaseModel):
    program:    str = Field("arabic",    description="arabic | english | analysis | all")
    optimizer:  str = Field("bootstrap", description="bootstrap | random_search | mipro")
    demos:      int = Field(3)
    candidates: int = Field(8)


@app.get("/dspy/status", tags=["DSPy"])
async def dspy_status():
    import dspy as _dspy  # DSPy already configured at startup via lm_setup import

    programs = {}
    for name in ["arabic","english","analysis"]:
        best = _dspy_best_compiled(name)
        programs[name] = {
            "compiled_exists": bool(best and Path(best).exists()),
            "compiled_path":   best,
        }

    results = _dspy_load_results()
    latest  = {}
    for r in results:
        prog = r.get("program")
        if prog not in latest or r["timestamp"] > latest[prog]["timestamp"]:
            latest[prog] = r

    return {
        "dspy_version":  _dspy.__version__,
        "programs":      programs,
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


@app.post("/dspy/optimize", tags=["DSPy"])
async def dspy_optimize(req: DSPyOptimizeRequest):
    programs = ["arabic","english","analysis"] if req.program == "all" else [req.program]
    results  = {}
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


@app.get("/dspy/results", tags=["DSPy"])
async def dspy_results():
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


class DSPyScoreRequest(BaseModel):
    program:        str  = Field(..., description="arabic | english | analysis")
    raw_transcript: str  = Field(..., description="Original input transcript")
    output:         dict = Field(..., description="DSPy prediction output fields")


@app.post("/dspy/score", tags=["DSPy"])
async def dspy_score(req: DSPyScoreRequest):
    METRICS = {"arabic": arabic_metric, "english": english_metric, "analysis": analysis_metric}
    if req.program not in METRICS:
        raise HTTPException(400, f"Unknown program. Choose: {list(METRICS)}")

    class _Ex:
        def __init__(self, d): self.__dict__.update(d)
    class _Pred:
        def __init__(self, d): self.__dict__.update(d)

    score = METRICS[req.program](_Ex({"raw_transcript": req.raw_transcript}), _Pred(req.output))
    return {
        "program": req.program,
        "score":   score,
        "label":   "Good" if score >= 0.75 else "Needs improvement" if score >= 0.5 else "Poor",
    }


@app.post("/refine/dspy", tags=["DSPy"])
async def refine_with_dspy(req: RefineRequest):
    if not req.transcript.strip():
        raise HTTPException(422, "transcript must not be empty.")
    try:
        result = await asyncio.to_thread(run_pipeline, req.transcript, req.context_info)
    except Exception as exc:
        raise HTTPException(500, str(exc))
    return JSONResponse(result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # import argparse, sys
    # parser = argparse.ArgumentParser(description="Run transCallMapping Refiner API server")
    # parser.add_argument("--host", default="0.0.0.0")
    # parser.add_argument("--port", default=8000)
    # args = parser.parse_args()
    # uvicorn.run(app, host=args.host, port=args.port)
    # sys.exit(0)

# EOF
