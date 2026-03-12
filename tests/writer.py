"""
writer.py – Saves transcript results to .txt files and returns structured JSON.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from config import OUTPUT_DIR


def _safe_stem(filename: str) -> str:
    """Return a filesystem-safe stem from an audio filename."""
    stem = Path(filename).stem
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    return safe[:80]  # cap length


def save_outputs(
    filename: str,
    original_transcription: str,
    arabic_refined: str,
    english_refined: str,
    context_info: str,
) -> dict:
    """
    Write three .txt files (original / arabic / english) under OUTPUT_DIR
    and return a JSON-serialisable result dict.

    Returns:
        {
            "filename":              str,
            "context_info":          str,
            "processed_at":          str (ISO-8601),
            "original_transcription": str,
            "arabic_refined":        str,
            "english_refined":       str,
            "output_files": {
                "original": str,
                "arabic":   str,
                "english":  str,
            }
        }
    """
    stem      = _safe_stem(filename)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    prefix    = f"{stem}_{timestamp}"

    paths = {
        "original": OUTPUT_DIR / f"{prefix}_original.txt",
        "arabic":   OUTPUT_DIR / f"{prefix}_arabic.txt",
        "english":  OUTPUT_DIR / f"{prefix}_english.txt",
    }

    paths["original"].write_text(original_transcription, encoding="utf-8")
    paths["arabic"].write_text(arabic_refined,           encoding="utf-8")
    paths["english"].write_text(english_refined,         encoding="utf-8")

    # Also write a combined JSON summary
    result = {
        "filename":               filename,
        "context_info":           context_info,
        "processed_at":           datetime.utcnow().isoformat() + "Z",
        "original_transcription": original_transcription,
        "arabic_refined":         arabic_refined,
        "english_refined":        english_refined,
        "output_files": {k: str(v) for k, v in paths.items()},
    }

    json_path = OUTPUT_DIR / f"{prefix}_result.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["output_files"]["json"] = str(json_path)

    return result