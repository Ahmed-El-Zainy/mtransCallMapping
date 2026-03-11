"""
main.py – FastAPI application for transCallMapping.

Endpoints
---------
POST /transcribe              Upload audio → transcription only (HTTP)
POST /process                 Upload audio → full pipeline (HTTP)
WS   /ws/process              Full pipeline with real-time progress (WebSocket)
GET  /health                  Health check
GET  /outputs                 List saved output files
GET  /outputs/{filename}      Download a saved output file
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from config import OUTPUT_DIR, SUPPORTED_AUDIO_EXTENSIONS, extract_context_from_filename
from app.transcriber import transcribe_audio
from app.refiner import refine_arabic, refine_english
from app.writer import save_outputs
from app.ws_process import run_pipeline

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="transCallMapping API",
    description=(
        "Audio call transcription + Arabic & English refinement "
        "for ELAraby Group customer-service calls.\n\n"
        "**WebSocket** `/ws/process` — real-time progress streaming.\n"
        "**HTTP** `/process` — classic single-response endpoint."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _validate_audio(file: UploadFile) -> None:
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Accepted formats: {', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))}"
            ),
        )


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws/process")
async def ws_process(websocket: WebSocket):
    """
    WebSocket endpoint — full pipeline with real-time progress.

    ## Protocol

    ### 1. Connect
    ```
    ws://localhost:8000/ws/process
    ```

    ### 2. Client sends ONE message (JSON):
    ```json
    {
      "filename": "20240315_AGT001_CUST4892_complaint.mp3",
      "audio_b64": "<base64-encoded audio bytes>"
    }
    ```

    ### 3. Server streams progress events:
    ```json
    { "event": "progress", "step": 1, "total": 6, "message": "Validating audio file…" }
    { "event": "progress", "step": 2, "total": 6, "message": "Context detected:\\nCall Date: 2024-03-15\\n…" }
    { "event": "progress", "step": 3, "total": 6, "message": "Transcribing audio with Whisper…" }
    { "event": "progress", "step": 3, "total": 6, "message": "Transcription complete — 42 segments" }
    { "event": "progress", "step": 4, "total": 6, "message": "Refining Arabic version…" }
    { "event": "progress", "step": 4, "total": 6, "message": "Arabic refinement complete ✓" }
    { "event": "progress", "step": 5, "total": 6, "message": "Refining English version…" }
    { "event": "progress", "step": 5, "total": 6, "message": "English refinement complete ✓" }
    { "event": "progress", "step": 6, "total": 6, "message": "Saving output files…" }
    { "event": "progress", "step": 6, "total": 6, "message": "All files saved ✓" }
    ```

    ### 4. Server sends final result:
    ```json
    {
      "event": "result",
      "filename": "…",
      "context_info": "…",
      "processed_at": "…",
      "original_transcription": "…",
      "arabic_refined": "…",
      "english_refined": "…",
      "output_files": { "original": "…", "arabic": "…", "english": "…", "json": "…" }
    }
    ```

    ### On error:
    ```json
    { "event": "error", "step": 3, "total": 6, "message": "Transcription failed: …" }
    ```
    """
    await websocket.accept()

    try:
        # Receive the initial message from client
        raw = await websocket.receive_text()
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await websocket.send_text(json.dumps({
                "event":   "error",
                "step":    0,
                "total":   6,
                "message": "Invalid JSON. Send: {\"filename\": \"…\", \"audio_b64\": \"…\"}",
            }))
            await websocket.close()
            return

        filename  = msg.get("filename", "upload.mp3")
        audio_b64 = msg.get("audio_b64", "")

        if not audio_b64:
            await websocket.send_text(json.dumps({
                "event":   "error",
                "step":    0,
                "total":   6,
                "message": "Missing 'audio_b64' field. Send base64-encoded audio bytes.",
            }))
            await websocket.close()
            return

        # Decode audio
        try:
            audio_bytes = base64.b64decode(audio_b64)
        except Exception:
            await websocket.send_text(json.dumps({
                "event":   "error",
                "step":    0,
                "total":   6,
                "message": "Failed to decode 'audio_b64'. Must be valid base64.",
            }))
            await websocket.close()
            return

        # Run the pipeline — streams progress back to client
        await run_pipeline(websocket, audio_bytes, filename)

    except WebSocketDisconnect:
        pass  # Client disconnected mid-stream — nothing to do

    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ── HTTP endpoints (kept for backwards compatibility) ─────────────────────────

@app.get("/health", tags=["System"])
async def health():
    """Simple liveness check."""
    return {"status": "ok", "service": "transCallMapping", "version": "2.0.0"}


@app.post("/transcribe", tags=["HTTP"])
async def transcribe_only(audio: UploadFile = File(...)):
    """
    Transcribe an audio file using Whisper (HTTP, no streaming).

    - **audio**: Audio file (mp3, wav, m4a, ogg, flac, webm, mp4)
    """
    _validate_audio(audio)
    audio_bytes  = await audio.read()
    context_info = extract_context_from_filename(audio.filename)

    try:
        transcript = transcribe_audio(audio_bytes, audio.filename)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")

    return JSONResponse({
        "filename":      audio.filename,
        "context_info":  context_info,
        "transcription": transcript,
    })


@app.post("/process", tags=["HTTP"])
async def process_call(audio: UploadFile = File(...)):
    """
    Full pipeline — HTTP version (no streaming, waits for complete result).

    Prefer `/ws/process` for real-time progress updates.
    """
    _validate_audio(audio)

    audio_bytes  = await audio.read()
    filename     = audio.filename
    context_info = extract_context_from_filename(filename)

    try:
        original_transcription = transcribe_audio(audio_bytes, filename)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")

    try:
        arabic_refined = refine_arabic(original_transcription, context_info)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Arabic refinement failed: {exc}")

    try:
        english_refined = refine_english(original_transcription, context_info)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"English refinement failed: {exc}")

    result = save_outputs(
        filename=filename,
        original_transcription=original_transcription,
        arabic_refined=arabic_refined,
        english_refined=english_refined,
        context_info=context_info,
    )
    return JSONResponse(result)


@app.get("/outputs/{output_filename}", tags=["Files"])
async def download_output(output_filename: str):
    """Download a previously generated output file by name."""
    file_path = OUTPUT_DIR / output_filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    try:
        file_path.resolve().relative_to(OUTPUT_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied.")
    media_type = "application/json" if output_filename.endswith(".json") else "text/plain"
    return FileResponse(file_path, media_type=media_type, filename=output_filename)


@app.get("/outputs", tags=["Files"])
async def list_outputs():
    """List all files currently in the output directory."""
    files = [
        {"name": f.name, "size_bytes": f.stat().st_size}
        for f in sorted(OUTPUT_DIR.iterdir())
        if f.is_file()
    ]
    return {"output_dir": str(OUTPUT_DIR), "files": files}