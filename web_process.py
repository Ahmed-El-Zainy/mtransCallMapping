
"""
ws_process.py – WebSocket pipeline runner.

Sends real-time progress events to the connected client as it works through:
  1. Validating the audio
  2. Parsing filename → context_info
  3. Transcribing via Whisper
  4. Refining in Arabic via Azure OpenAI
  5. Refining in English via Azure OpenAI
  6. Saving output files

Message schema (all JSON):
  { "event": "progress",  "step": int, "total": 6, "message": str }
  { "event": "result",    ...full result payload... }
  { "event": "error",     "step": int, "message": str }
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import WebSocket

from config import SUPPORTED_AUDIO_EXTENSIONS, extract_context_from_filename
from transcriber import transcribe_audio
from refiner import refine_arabic, refine_english
from writer import save_outputs

TOTAL_STEPS = 6


async def _send(ws: WebSocket, payload: dict[str, Any]) -> None:
    """Send a JSON message over the WebSocket."""
    await ws.send_text(json.dumps(payload, ensure_ascii=False))


async def _progress(ws: WebSocket, step: int, message: str) -> None:
    await _send(ws, {
        "event":   "progress",
        "step":    step,
        "total":   TOTAL_STEPS,
        "message": message,
    })


async def _error(ws: WebSocket, step: int, message: str) -> None:
    await _send(ws, {
        "event":   "error",
        "step":    step,
        "total":   TOTAL_STEPS,
        "message": message,
    })


async def run_pipeline(ws: WebSocket, audio_bytes: bytes, filename: str) -> None:
    """
    Execute the full processing pipeline, streaming progress events
    back to the WebSocket client as each step completes.
    """

    # ── Step 1 · Validate ────────────────────────────────────────────────────
    await _progress(ws, 1, f"Validating audio file: {filename}")
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_AUDIO_EXTENSIONS:
        await _error(ws, 1,
            f"Unsupported file type '{ext}'. "
            f"Accepted: {', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))}"
        )
        return

    # ── Step 2 · Parse filename ──────────────────────────────────────────────
    await _progress(ws, 2, "Extracting context from filename…")
    context_info = extract_context_from_filename(filename)
    await _progress(ws, 2, f"Context detected:\n{context_info}")

    # ── Step 3 · Transcribe ──────────────────────────────────────────────────
    await _progress(ws, 3, "Transcribing audio with Whisper… (this may take a moment)")
    try:
        # Run blocking Whisper call in a thread so we don't block the event loop
        original_transcription = await asyncio.to_thread(
            transcribe_audio, audio_bytes, filename
        )
    except Exception as exc:
        await _error(ws, 3, f"Transcription failed: {exc}")
        return

    line_count = len([l for l in original_transcription.splitlines() if l.strip()])
    await _progress(ws, 3, f"Transcription complete — {line_count} segments")

    # ── Step 4 · Arabic refinement ───────────────────────────────────────────
    await _progress(ws, 4, "Refining Arabic (Egyptian dialect) version via Azure OpenAI…")
    try:
        arabic_refined = await asyncio.to_thread(
            refine_arabic, original_transcription, context_info
        )
    except Exception as exc:
        await _error(ws, 4, f"Arabic refinement failed: {exc}")
        return

    await _progress(ws, 4, "Arabic refinement complete ✓")

    # ── Step 5 · English refinement ──────────────────────────────────────────
    await _progress(ws, 5, "Refining English version via Azure OpenAI…")
    try:
        english_refined = await asyncio.to_thread(
            refine_english, original_transcription, context_info
        )
    except Exception as exc:
        await _error(ws, 5, f"English refinement failed: {exc}")
        return

    await _progress(ws, 5, "English refinement complete ✓")

    # ── Step 6 · Save files ──────────────────────────────────────────────────
    await _progress(ws, 6, "Saving output files…")
    try:
        result = await asyncio.to_thread(
            save_outputs,
            filename,
            original_transcription,
            arabic_refined,
            english_refined,
            context_info,
        )
    except Exception as exc:
        await _error(ws, 6, f"Failed to save output files: {exc}")
        return

    await _progress(ws, 6, "All files saved ✓")

    # ── Final result ─────────────────────────────────────────────────────────
    await _send(ws, {"event": "result", **result})