"""
main.py – FastAPI — three ways to call the refiner pipeline.

┌─────────────────────┬────────────────┬───────────────┬──────────────────┐
│ Endpoint            │ Method         │ Response      │ Streaming        │
├─────────────────────┼────────────────┼───────────────┼──────────────────┤
│ POST /refine        │ HTTP           │ JSON          │ ❌ blocking      │
│ POST /refine/stream │ HTTP SSE       │ text/event-   │ ✅ token-by-tok  │
│                     │                │ stream        │                  │
│ WS   /ws/refine     │ WebSocket      │ JSON messages │ ✅ token-by-tok  │
└─────────────────────┴────────────────┴───────────────┴──────────────────┘

All three share the same event schema (SSE wraps it in  data: ...\n\n):

  {"event":"start",        "section":"arabic"|"english"|"summary"}
  {"event":"token",        "section":"...", "token":"<text>"}
  {"event":"section_done", "section":"...", "text":"<full section text>"}
  {"event":"done",         "arabic_refined":"...",
                           "english_refined":"...", "summary":{...}}
  {"event":"error",        "message":"..."}
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from refiner import run_pipeline, stream_pipeline

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="transCallMapping — Refiner API",
    description=(
        "**Refine a raw call transcript three ways:**\n\n"
        "| Endpoint | Streaming | Use when |\n"
        "|---|---|---|\n"
        "| `POST /refine` | ❌ | Simple integrations, Postman quick test |\n"
        "| `POST /refine/stream` | ✅ SSE | Browser `EventSource`, curl, Postman |\n"
        "| `WS /ws/refine` | ✅ WS | Real-time UI, persistent connections |\n"
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request model (shared by all three endpoints) ─────────────────────────────

class RefineRequest(BaseModel):
    transcript:   str = Field(..., description="Raw transcript from the STT API")
    context_info: str = Field("",  description="Optional: call date, agent ID, topic…")


# ── Core async bridge ─────────────────────────────────────────────────────────
# stream_pipeline() is a *synchronous* blocking generator (it calls Azure OpenAI
# with stream=True in a regular for-loop).  We run it in a thread-pool executor
# and forward each yielded event through an asyncio.Queue so FastAPI's async
# event loop can consume it without blocking.

async def _iter_pipeline(transcript: str, context_info: str) -> AsyncGenerator[dict, None]:
    """
    Bridges the blocking stream_pipeline() generator into async-land.

    Yields dicts:
        {"event":"start"|"token"|"section_done"|"done"|"error", ...}
    """
    loop  = asyncio.get_running_loop()          # ← correct API (3.7+)
    queue: asyncio.Queue[dict | None] = asyncio.Queue()

    def _producer() -> None:
        """Runs in a thread — calls Azure, pushes events onto the queue."""
        try:
            for event in stream_pipeline(transcript, context_info):
                loop.call_soon_threadsafe(queue.put_nowait, event)
        except Exception as exc:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"event": "error", "message": str(exc)},
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)   # sentinel

    # Schedule producer in thread-pool and immediately start consuming the queue
    future = loop.run_in_executor(None, _producer)   # non-blocking

    try:
        while True:
            event = await queue.get()
            if event is None:       # sentinel → producer finished
                break
            yield event
    finally:
        await asyncio.shield(future)   # make sure the thread finishes cleanly


# ── 1. Blocking HTTP  POST /refine ────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "service": "transCallMapping-refiner", "version": "2.0.0"}


@app.post("/refine", tags=["1 · HTTP Blocking"])
async def refine_blocking(req: RefineRequest):
    """
    Runs the full pipeline and returns **one JSON response** when everything
    is done.  Simple to call — but the client waits silently for 30–90 s.

    ```json
    {
      "original_transcription": "...",
      "context_info": "...",
      "arabic_refined": "...",
      "english_refined": "...",
      "summary": { ... }
    }
    ```
    """
    if not req.transcript.strip():
        raise HTTPException(status_code=422, detail="transcript must not be empty.")
    try:
        result = await asyncio.to_thread(run_pipeline, req.transcript, req.context_info)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return JSONResponse(result)


# ── 2. SSE streaming  POST /refine/stream ────────────────────────────────────

@app.post("/refine/stream", tags=["2 · SSE Streaming"])
async def refine_sse(req: RefineRequest):
    """
    **Server-Sent Events** — tokens stream as `text/event-stream` the moment
    Azure OpenAI generates them.

    ### SSE message format
    Each message is one line: `data: <json>\\n\\n`

    ```
    data: {"event":"start","section":"arabic"}

    data: {"event":"token","section":"arabic","token":"أهلاً"}

    data: {"event":"token","section":"arabic","token":" وسهلاً"}

    data: {"event":"section_done","section":"arabic","text":"أهلاً وسهلاً..."}

    data: {"event":"start","section":"english"}
    ...
    data: {"event":"done","arabic_refined":"...","english_refined":"...","summary":{}}

    data: [DONE]
    ```

    ### How to test in Postman
    1. New request → **POST** → `http://localhost:8000/refine/stream`
    2. Body → raw → JSON → paste request body
    3. Hit **Send**
    4. Watch the **Response** panel — lines appear token-by-token in real time

    ### How to test with curl
    ```bash
    curl -N -X POST http://localhost:8000/refine/stream \\
      -H "Content-Type: application/json" \\
      -d '{"transcript":"مساء الخدمة...","context_info":""}'
    ```
    (`-N` disables curl buffering so you see tokens as they arrive)
    """
    if not req.transcript.strip():
        raise HTTPException(status_code=422, detail="transcript must not be empty.")

    async def _sse_generator():
        async for event in _iter_pipeline(req.transcript, req.context_info):
            line = json.dumps(event, ensure_ascii=False)
            yield f"data: {line}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "Connection":        "keep-alive",
            "X-Accel-Buffering": "no",    # disable nginx / proxy buffering
        },
    )


# ── 3. WebSocket streaming  WS /ws/refine ────────────────────────────────────

@app.websocket("/ws/refine")
async def ws_refine(websocket: WebSocket):
    """
    **WebSocket** — identical event stream to SSE, but over a persistent
    bi-directional connection (no `data:` prefix, raw JSON per message).

    ### Protocol
    **Client → Server** (one message to start the pipeline):
    ```json
    {"transcript": "مساء الخدمة...", "context_info": "Agent: AGT001"}
    ```

    **Server → Client** (many messages, token-by-token):
    ```json
    {"event":"start","section":"arabic"}
    {"event":"token","section":"arabic","token":"أهلاً"}
    {"event":"token","section":"arabic","token":" وسهلاً"}
    ...
    {"event":"section_done","section":"arabic","text":"أهلاً وسهلاً..."}
    {"event":"start","section":"english"}
    ...
    {"event":"done","arabic_refined":"...","english_refined":"...","summary":{}}
    ```

    ### How to test in Postman
    1. New → **WebSocket**
    2. URL: `ws://localhost:8000/ws/refine`
    3. Click **Connect**
    4. In the message box paste the JSON above → **Send**
    5. Watch messages arrive token-by-token in the **Messages** panel
    """
    await websocket.accept()

    try:
        # ── Receive the single request message ────────────────────────────────
        raw = await websocket.receive_text()
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await websocket.send_text(json.dumps({
                "event":   "error",
                "message": 'Invalid JSON. Expected: {"transcript":"...","context_info":"..."}',
            }))
            return

        transcript   = msg.get("transcript",   "").strip()
        context_info = msg.get("context_info", "")

        if not transcript:
            await websocket.send_text(json.dumps({
                "event": "error", "message": "transcript must not be empty.",
            }))
            return

        # ── Stream every event to the client ──────────────────────────────────
        async for event in _iter_pipeline(transcript, context_info):
            await websocket.send_text(json.dumps(event, ensure_ascii=False))

    except WebSocketDisconnect:
        pass    # client closed the tab / connection — nothing to do

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




if __name__ == "__main__":
    # Quick test
    import sys
    import argparse
    from pathlib import Path
    import unicorn

    
    if len(sys.argv) != 2:
        print("Usage: python main.py <transcript.txt>")
        sys.exit(1)
    transcript = Path(sys.argv[1]).read_text(encoding="utf-8")
    context = "Test call between customer and agent at Miraco Company."
    for event in stream_pipeline(transcript, context):
        print(event)
