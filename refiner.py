"""
refiner.py – LLM calls with true token-by-token streaming.

Non-streaming (original):
    refine_arabic()   → str
    refine_english()  → str
    summarise()       → dict
    run_pipeline()    → dict

Streaming generators (new):
    stream_arabic()   → Iterator[str]   — yields tokens as they arrive
    stream_english()  → Iterator[str]   — yields tokens as they arrive
    stream_summarise()→ Iterator[str]   — yields tokens, caller collects + parses
    stream_pipeline() → Iterator[dict]  — yields structured event dicts:
        {"event": "token",    "section": "arabic"|"english"|"summary", "token": str}
        {"event": "section_done", "section": "...", "text": str}
        {"event": "done",     "arabic_refined": str, "english_refined": str, "summary": dict}
        {"event": "error",    "message": str}
"""

from __future__ import annotations

import json
from typing import Iterator

import openai

from config import (
    ARABIC_SYSTEM_PROMPT,
    ARABIC_USER_PROMPT_TEMPLATE,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
    ENGLISH_SYSTEM_PROMPT,
    ENGLISH_USER_PROMPT_TEMPLATE,
    SUMMARY_SYSTEM_PROMPT,
    SUMMARY_USER_PROMPT_TEMPLATE,
)


# ── Client ────────────────────────────────────────────────────────────────────

def _get_client() -> openai.AzureOpenAI:
    return openai.AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )


# ── Non-streaming (blocking) ──────────────────────────────────────────────────

def _call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
    """Blocking single call — returns full content string."""
    client   = _get_client()
    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=4096,
    )
    return response.choices[0].message.content.strip()


def refine_arabic(original_transcription: str, context_info: str = "") -> str:
    user_prompt = ARABIC_USER_PROMPT_TEMPLATE.format(
        context_info=context_info,
        original_transcription=original_transcription,
    )
    return _call_llm(ARABIC_SYSTEM_PROMPT, user_prompt)


def refine_english(original_transcription: str, context_info: str = "") -> str:
    user_prompt = ENGLISH_USER_PROMPT_TEMPLATE.format(
        context_info=context_info,
        original_transcription=original_transcription,
    )
    return _call_llm(ENGLISH_SYSTEM_PROMPT, user_prompt)


def summarise(original_transcription: str, context_info: str = "") -> dict:
    user_prompt = SUMMARY_USER_PROMPT_TEMPLATE.format(
        context_info=context_info,
        original_transcription=original_transcription,
    )
    raw   = _call_llm(SUMMARY_SYSTEM_PROMPT, user_prompt, temperature=0.1)
    clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {"raw_summary": clean, "parse_error": "LLM did not return valid JSON"}


def run_pipeline(original_transcription: str, context_info: str = "") -> dict:
    """Blocking pipeline — returns complete result dict."""
    return {
        "original_transcription": original_transcription,
        "context_info":           context_info,
        "arabic_refined":         refine_arabic(original_transcription,  context_info),
        "english_refined":        refine_english(original_transcription, context_info),
        "summary":                summarise(original_transcription,      context_info),
    }


# ── Streaming generators ──────────────────────────────────────────────────────

def _stream_llm(
    system_prompt: str,
    user_prompt:   str,
    temperature:   float = 0.3,
) -> Iterator[str]:
    """
    Core streaming call.
    Yields each text token as it arrives from Azure OpenAI.
    """
    client = _get_client()
    stream = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=4096,
        stream=True,                   # ← the key difference
    )
    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            yield delta.content


def stream_arabic(original_transcription: str, context_info: str = "") -> Iterator[str]:
    """Yields Arabic tokens one by one as the model generates them."""
    user_prompt = ARABIC_USER_PROMPT_TEMPLATE.format(
        context_info=context_info,
        original_transcription=original_transcription,
    )
    yield from _stream_llm(ARABIC_SYSTEM_PROMPT, user_prompt)


def stream_english(original_transcription: str, context_info: str = "") -> Iterator[str]:
    """Yields English tokens one by one as the model generates them."""
    user_prompt = ENGLISH_USER_PROMPT_TEMPLATE.format(
        context_info=context_info,
        original_transcription=original_transcription,
    )
    yield from _stream_llm(ENGLISH_SYSTEM_PROMPT, user_prompt)


def stream_summarise(original_transcription: str, context_info: str = "") -> Iterator[str]:
    """Yields summary JSON tokens one by one. Caller must collect + parse."""
    user_prompt = SUMMARY_USER_PROMPT_TEMPLATE.format(
        context_info=context_info,
        original_transcription=original_transcription,
    )
    yield from _stream_llm(SUMMARY_SYSTEM_PROMPT, user_prompt, temperature=0.1)


def _parse_summary(raw: str) -> dict:
    """Parse collected summary tokens into a dict."""
    clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {"raw_summary": clean, "parse_error": "LLM did not return valid JSON"}


def stream_pipeline(
    original_transcription: str,
    context_info: str = "",
) -> Iterator[dict]:
    """
    Full streaming pipeline — yields structured event dicts.

    Event types:
      {"event": "start",        "section": "arabic"|"english"|"summary"}
      {"event": "token",        "section": "...", "token": "<text>"}
      {"event": "section_done", "section": "...", "text": "<full text>"}
      {"event": "done",         "arabic_refined": str,
                                "english_refined": str, "summary": dict}
      {"event": "error",        "message": str}

    Usage:
        for event in stream_pipeline(transcript, context):
            # send event to client
    """
    try:
        # ── Section 1: Arabic ─────────────────────────────────────────────────
        yield {"event": "start", "section": "arabic"}
        arabic_tokens: list[str] = []

        for token in stream_arabic(original_transcription, context_info):
            arabic_tokens.append(token)
            yield {"event": "token", "section": "arabic", "token": token}

        arabic_text = "".join(arabic_tokens).strip()
        yield {"event": "section_done", "section": "arabic", "text": arabic_text}

        # ── Section 2: English ────────────────────────────────────────────────
        yield {"event": "start", "section": "english"}
        english_tokens: list[str] = []

        for token in stream_english(original_transcription, context_info):
            english_tokens.append(token)
            yield {"event": "token", "section": "english", "token": token}

        english_text = "".join(english_tokens).strip()
        yield {"event": "section_done", "section": "english", "text": english_text}

        # ── Section 3: Summary ────────────────────────────────────────────────
        yield {"event": "start", "section": "summary"}
        summary_tokens: list[str] = []

        for token in stream_summarise(original_transcription, context_info):
            summary_tokens.append(token)
            yield {"event": "token", "section": "summary", "token": token}

        summary = _parse_summary("".join(summary_tokens))
        yield {"event": "section_done", "section": "summary",
               "text": json.dumps(summary, ensure_ascii=False)}

        # ── Final done event ──────────────────────────────────────────────────
        yield {
            "event":                  "done",
            "original_transcription": original_transcription,
            "context_info":           context_info,
            "arabic_refined":         arabic_text,
            "english_refined":        english_text,
            "summary":                summary,
        }

    except Exception as exc:
        yield {"event": "error", "message": str(exc)}
