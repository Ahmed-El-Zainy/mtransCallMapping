"""
dspy_refiner.py — Pipeline using DSPy programs.
Loads compiled (optimized) programs if they exist, falls back to uncompiled.

Public API (same contract as any refiner):
  run_pipeline(transcript, context_info)    → dict
  stream_pipeline(transcript, context_info) → Iterator[dict]
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterator

import lm_setup  # importing this module triggers configure_dspy() at module level
from programs  import ArabicRefiner, EnglishRefiner, CallAnalyser
from optimizer import get_best_compiled


def _load_program(ProgramClass, program_name: str):
    """Return (program, is_compiled). Loads best compiled version if available."""
    best_path = get_best_compiled(program_name)
    if best_path and Path(best_path).exists():
        prog = ProgramClass()
        prog.load(best_path)
        return prog, True
    return ProgramClass(), False


_arabic_program,   _arabic_compiled   = _load_program(ArabicRefiner,  "arabic")
_english_program,  _english_compiled  = _load_program(EnglishRefiner, "english")
_analysis_program, _analysis_compiled = _load_program(CallAnalyser,   "analysis")


def get_program_info() -> dict:
    return {
        "arabic":   {"compiled": _arabic_compiled},
        "english":  {"compiled": _english_compiled},
        "analysis": {"compiled": _analysis_compiled},
    }


def run_pipeline(original_transcription: str, context_info: str = "") -> dict:
    arabic_pred   = _arabic_program(raw_transcript=original_transcription, context_info=context_info)
    english_pred  = _english_program(raw_transcript=original_transcription, context_info=context_info)
    analysis_pred = _analysis_program(raw_transcript=original_transcription, context_info=context_info)
    analysis      = _analysis_program.to_analysis_dict(analysis_pred)

    return {
        "original_transcription": original_transcription,
        "context_info":           context_info,
        "transcript_arabic":      getattr(arabic_pred,  'refined_transcript', '') or '',
        "transcript_english":     getattr(english_pred, 'refined_transcript', '') or '',
        **analysis,
        "_dspy": {
            "arabic_compiled":   _arabic_compiled,
            "english_compiled":  _english_compiled,
            "analysis_compiled": _analysis_compiled,
        },
    }


def stream_pipeline(original_transcription: str, context_info: str = "") -> Iterator[dict]:
    try:
        yield {"event": "start", "section": "arabic"}
        arabic_pred  = _arabic_program(raw_transcript=original_transcription, context_info=context_info)
        arabic_text  = getattr(arabic_pred, 'refined_transcript', '') or ''
        yield {"event": "section_done", "section": "arabic", "text": arabic_text}

        yield {"event": "start", "section": "english"}
        english_pred = _english_program(raw_transcript=original_transcription, context_info=context_info)
        english_text = getattr(english_pred, 'refined_transcript', '') or ''
        yield {"event": "section_done", "section": "english", "text": english_text}

        yield {"event": "start", "section": "analysis"}
        analysis_pred = _analysis_program(raw_transcript=original_transcription, context_info=context_info)
        analysis      = _analysis_program.to_analysis_dict(analysis_pred)
        yield {"event": "section_done", "section": "analysis", "text": json.dumps(analysis, ensure_ascii=False)}

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
