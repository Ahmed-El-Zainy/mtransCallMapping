"""
transcriber.py – Azure Speech-to-Text with continuous recognition.

Features:
  - Continuous recognition (handles long calls, not just 60-second clips)
  - PhraseList grammar for domain-specific terms (Inverter, Eco master, etc.)
  - Timestamps per segment in [HH:MM:SS.mmm] format (offset from call start)
  - Auto language detection: Arabic (Egypt) + English
  - Converts mp3/m4a/ogg/webm → wav in-memory before sending to Azure
  - Speaker labels where available (Guest-1 / Guest-2 → Agent / Customer)
"""

from __future__ import annotations

import io
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import NamedTuple

import azure.cognitiveservices.speech as speechsdk

from config import (
    AZURE_SPEECH_ENDPOINT,
    AZURE_SPEECH_KEY,
    AZURE_SPEECH_PHRASE_LIST,
    AZURE_SPEECH_REGION,
    SUPPORTED_AUDIO_EXTENSIONS,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

class Segment(NamedTuple):
    offset_ms: int   # milliseconds from start of audio
    text: str
    speaker: str     # "Agent" | "Customer" | ""


def _ticks_to_ms(ticks: int) -> int:
    """Azure Speech offsets are in 100-nanosecond ticks → convert to ms."""
    return ticks // 10_000


def _ms_to_ts(ms: int) -> str:
    """Milliseconds → HH:MM:SS.mmm"""
    h   = ms // 3_600_000
    ms -= h * 3_600_000
    m   = ms // 60_000
    ms -= m * 60_000
    s   = ms // 1_000
    ms -= s * 1_000
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _to_wav(audio_bytes: bytes, filename: str) -> bytes:
    """
    Convert any supported audio format to 16kHz mono WAV using pydub/ffmpeg.
    Falls back to returning the original bytes if pydub is not installed
    (Azure SDK can handle WAV natively; for other formats ffmpeg is needed).
    """
    suffix = Path(filename).suffix.lower()
    if suffix == ".wav":
        return audio_bytes  # already wav

    try:
        from pydub import AudioSegment  # type: ignore
    except ImportError:
        # pydub not installed — pass bytes through and let Azure handle it
        return audio_bytes

    fmt_map = {
        ".mp3": "mp3", ".m4a": "mp4", ".ogg": "ogg",
        ".flac": "flac", ".webm": "webm", ".mp4": "mp4",
    }
    fmt = fmt_map.get(suffix, suffix.lstrip("."))

    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
    audio = audio.set_frame_rate(16_000).set_channels(1)

    buf = io.BytesIO()
    audio.export(buf, format="wav")
    return buf.getvalue()


def _speaker_label(raw: str) -> str:
    """
    Map Azure speaker IDs (Guest-1, Guest-2, AGENT, CUSTOMER …) to
    friendly labels. Without diarisation the field is empty.
    """
    if not raw:
        return ""
    r = raw.upper()
    if any(x in r for x in ("AGENT", "STAFF", "EMP", "1")):
        return "Agent"
    if any(x in r for x in ("CUSTOMER", "CLIENT", "CUST", "2")):
        return "Customer"
    return raw  # keep original if unrecognised


# ── Main transcription function ───────────────────────────────────────────────

def transcribe_audio(audio_bytes: bytes, filename: str) -> str:
    """
    Transcribe audio using Azure Cognitive Services Speech SDK.

    Returns a formatted transcript:
        [HH:MM:SS.mmm] Agent: مرحبا، معك خدمة عملاء شركة ميراكوا
        [HH:MM:SS.mmm] Customer: عايز أعرف إيه اللي بيحصل مع التلاجة

    Args:
        audio_bytes : Raw bytes of the audio file.
        filename    : Original filename (used for format detection).

    Returns:
        Multi-line transcript string.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValueError(
            f"Unsupported format '{suffix}'. "
            f"Supported: {', '.join(SUPPORTED_AUDIO_EXTENSIONS)}"
        )

    # ── Convert to WAV if needed ──────────────────────────────────────────────
    wav_bytes = _to_wav(audio_bytes, filename)

    # ── Write to a temp file (SDK needs a file path or stream) ───────────────
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes)
        tmp_path = tmp.name

    try:
        segments = _run_continuous_recognition(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return _format_segments(segments)


def _build_speech_config() -> speechsdk.SpeechConfig:
    """Build SpeechConfig with auto language detection (AR-EG + EN-US)."""
    endpoint_url = AZURE_SPEECH_ENDPOINT
    if not endpoint_url.startswith("https://"):
        endpoint_url = f"https://{endpoint_url}"
    # Ensure no double slashes
    endpoint_url = endpoint_url.rstrip("/")

    speech_config = speechsdk.SpeechConfig(
        subscription=AZURE_SPEECH_KEY,
        endpoint=f"{endpoint_url}",
    )

    # Output detailed results including timing
    speech_config.output_format = speechsdk.OutputFormat.Detailed

    # Enable diarization (speaker separation) — works for 2-speaker calls
    speech_config.set_property(
        speechsdk.PropertyId.SpeechServiceConnection_SingleLanguageIdPriority,
        "Latency",
    )
    speech_config.request_word_level_timestamps()

    # Enable profanity — keep original words for call-centre review
    speech_config.set_profanity(speechsdk.ProfanityOption.Raw)

    return speech_config


def _run_continuous_recognition(wav_path: str) -> list[Segment]:
    """
    Run continuous recognition on a WAV file.
    Blocks until the file is fully recognised and returns all segments.
    """
    speech_config = _build_speech_config()

    # ── Auto language detection: Arabic (Egypt) + English (US) ───────────────
    auto_detect_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
        languages=["ar-EG", "en-US"]
    )

    audio_config = speechsdk.AudioConfig(filename=wav_path)

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config,
        auto_detect_source_language_config=auto_detect_config,
    )

    # ── Inject phrase hints ───────────────────────────────────────────────────
    if AZURE_SPEECH_PHRASE_LIST:
        phrase_list = speechsdk.PhraseListGrammar.from_recognizer(recognizer)
        for phrase in AZURE_SPEECH_PHRASE_LIST:
            phrase_list.addPhrase(phrase)

    # ── Collect results via callbacks ─────────────────────────────────────────
    segments: list[Segment] = []
    done      = threading.Event()
    errors: list[str] = []

    def on_recognized(evt: speechsdk.SpeechRecognitionEventArgs):
        result = evt.result
        if result.reason == speechsdk.ResultReason.RecognizedSpeech and result.text.strip():
            offset_ms = _ticks_to_ms(result.offset)
            # Speaker ID is buried in the JSON result for diarised sessions
            speaker = _extract_speaker(result)
            segments.append(Segment(offset_ms=offset_ms, text=result.text.strip(), speaker=speaker))

    def on_canceled(evt: speechsdk.SpeechRecognitionCanceledEventArgs):
        if evt.result.reason == speechsdk.ResultReason.Canceled:
            details = evt.result.cancellation_details
            if details.reason == speechsdk.CancellationReason.Error:
                errors.append(details.error_details)
        done.set()

    def on_session_stopped(_evt):
        done.set()

    recognizer.recognized.connect(on_recognized)
    recognizer.canceled.connect(on_canceled)
    recognizer.session_stopped.connect(on_session_stopped)

    recognizer.start_continuous_recognition()
    done.wait(timeout=600)   # 10-minute max for very long calls
    recognizer.stop_continuous_recognition()

    if errors:
        raise RuntimeError(f"Azure Speech error: {errors[0]}")

    return sorted(segments, key=lambda s: s.offset_ms)


def _extract_speaker(result: speechsdk.SpeechRecognitionResult) -> str:
    """
    Try to pull the speaker ID from the detailed JSON result.
    Falls back to empty string if diarization is not enabled / available.
    """
    try:
        import json as _json
        detail = _json.loads(result.json)
        # NBest[0].Speaker or SpeakerId field (varies by API version)
        nbest = detail.get("NBest", [])
        if nbest:
            raw = nbest[0].get("Speaker", "") or nbest[0].get("SpeakerId", "")
            return _speaker_label(raw)
        raw = detail.get("SpeakerId", "")
        return _speaker_label(raw)
    except Exception:
        return ""


# ── Formatting ────────────────────────────────────────────────────────────────

def _format_segments(segments: list[Segment]) -> str:
    """
    Format segments into the timestamped transcript string.

    With speaker:    [00:00:03.120] Agent: أهلاً وسهلاً...
    Without speaker: [00:00:03.120] أهلاً وسهلاً...
    """
    lines: list[str] = []
    for seg in segments:
        ts     = _ms_to_ts(seg.offset_ms)
        prefix = f"{seg.speaker}: " if seg.speaker else ""
        lines.append(f"[{ts}] {prefix}{seg.text}")

    if not lines:
        lines = ["[00:00:00.000] (no speech detected)"]

    return "\n".join(lines)





if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe an audio file using Azure Speech-to-Text.")
    parser.add_argument("input", help="Path to the input audio file (mp3/m4a/ogg/webm/wav).")
    args = parser.parse_args()

    input_path = args.input
    if not os.path.isfile(input_path):
        print(f"Error: File '{input_path}' does not exist.")
        exit(1)

    with open(input_path, "rb") as f:
        audio_bytes = f.read()

    try:
        transcript = transcribe_audio(audio_bytes, filename=input_path)
        print(transcript)
    except Exception as e:
        print(f"Error during transcription: {e}")