"""
dspy_refiner.py — Drop-in replacement for refiner.py that uses DSPy.

Uses compiled (optimized) DSPy programs if they exist,
falls back to uncompiled programs otherwise.

Exposes the same public API as refiner.py:
  run_pipeline(transcript, context_info) → dict
  stream_pipeline(transcript, context_info) → Iterator[dict]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterator

# ROOT = Path(__file__).parent.parent
# sys.path.insert(0, str(ROOT))

from lm_setup  import configure_dspy
from programs  import ArabicRefiner, EnglishRefiner, CallAnalyser
from optimizer import get_best_compiled

# Configure DSPy once on import
configure_dspy()


# ── Load programs (compiled if available, else uncompiled) ─────────────────────

def _load_program(ProgramClass, program_name: str):
    """Load the best compiled version or return a fresh uncompiled program."""
    best_path = get_best_compiled(program_name)
    if best_path and Path(best_path).exists():
        prog = ProgramClass()
        prog.load(best_path)
        return prog, True   # (program, is_compiled)
    return ProgramClass(), False


_arabic_program,  _arabic_compiled  = _load_program(ArabicRefiner,  "arabic")
_english_program, _english_compiled = _load_program(EnglishRefiner, "english")
_analysis_program,_analysis_compiled= _load_program(CallAnalyser,   "analysis")


def get_program_info() -> dict:
    """Return which program versions are active."""
    return {
        "arabic":   {"compiled": _arabic_compiled},
        "english":  {"compiled": _english_compiled},
        "analysis": {"compiled": _analysis_compiled},
    }


# ── Blocking pipeline ─────────────────────────────────────────────────────────

def run_pipeline(original_transcription: str, context_info: str = "") -> dict:
    """
    Run the full DSPy pipeline and return the enriched result dict.
    Same output contract as refiner.run_pipeline().
    """
    # Step 1: Arabic
    arabic_pred = _arabic_program(
        raw_transcript = original_transcription,
        context_info   = context_info,
    )
    arabic_text = getattr(arabic_pred, 'refined_transcript', '') or ''

    # Step 2: English
    english_pred = _english_program(
        raw_transcript = original_transcription,
        context_info   = context_info,
    )
    english_text = getattr(english_pred, 'refined_transcript', '') or ''

    # Step 3: Analysis
    analysis_pred = _analysis_program(
        raw_transcript = original_transcription,
        context_info   = context_info,
    )
    analysis = _analysis_program.to_analysis_dict(analysis_pred)

    return {
        "original_transcription": original_transcription,
        "context_info":           context_info,
        "transcript_arabic":      arabic_text,
        "transcript_english":     english_text,
        **analysis,
        "_dspy": {
            "arabic_compiled":   _arabic_compiled,
            "english_compiled":  _english_compiled,
            "analysis_compiled": _analysis_compiled,
        },
    }


# ── Streaming pipeline ─────────────────────────────────────────────────────────

def stream_pipeline(
    original_transcription: str,
    context_info: str = "",
) -> Iterator[dict]:
    """
    Streaming version — yields the same event dicts as refiner.stream_pipeline().
    DSPy calls are blocking, so we yield start/done events around each call.
    """
    try:
        # ── Arabic ────────────────────────────────────────────────────────────
        yield {"event": "start", "section": "arabic"}
        arabic_pred = _arabic_program(
            raw_transcript = original_transcription,
            context_info   = context_info,
        )
        arabic_text = getattr(arabic_pred, 'refined_transcript', '') or ''
        yield {"event": "section_done", "section": "arabic", "text": arabic_text}

        # ── English ───────────────────────────────────────────────────────────
        yield {"event": "start", "section": "english"}
        english_pred = _english_program(
            raw_transcript = original_transcription,
            context_info   = context_info,
        )
        english_text = getattr(english_pred, 'refined_transcript', '') or ''
        yield {"event": "section_done", "section": "english", "text": english_text}

        # ── Analysis ──────────────────────────────────────────────────────────
        yield {"event": "start", "section": "analysis"}
        analysis_pred = _analysis_program(
            raw_transcript = original_transcription,
            context_info   = context_info,
        )
        analysis = _analysis_program.to_analysis_dict(analysis_pred)
        yield {"event": "section_done", "section": "analysis",
               "text": json.dumps(analysis, ensure_ascii=False)}

        # ── Done ──────────────────────────────────────────────────────────────
        yield {
            "event":                  "done",
            "original_transcription": original_transcription,
            "context_info":           context_info,
            "transcript_arabic":      arabic_text,
            "transcript_english":     english_text,
            **analysis,
        }

    except Exception as exc:
        yield {"event": "error", "message": str(exc)}
