"""
refiner.py — Three LLM calls, one pipeline.

Flow:
  raw transcript
       │
       ├─► refine_arabic()   →  cleaned Egyptian-Arabic transcript
       ├─► refine_english()  →  professional English transcript
       └─► analyse()         →  all analysis fields as one dict

Public API:
  run_pipeline(transcript, context_info)  →  full result dict
  stream_pipeline(transcript, context_info)  →  generator of event dicts

Output dict keys (matches Scenario 1 user story):
  transcript_arabic       str
  transcript_english      str
  main_subject            str
  call_outcome            str   Resolved | Unresolved | Escalated | Follow-up Needed
  issue_resolution        str
  call_summary            str
  keywords                list[str]   max 5, English only
  call_category           str   Inquiry | Complaint | Technical Support | Billing | Sales | Feedback
  service                 str
  agent_attitude          str   Friendly | Neutral | Rude
  customer_satisfaction   str   Satisfied | Neutral | Dissatisfied
  language                str   Arabic | English | Mixed
  call_score              int   0-100
  score_breakdown         dict  7 questions with Pass/Fail + note
"""

from __future__ import annotations

import json
from typing import Iterator

import openai

from config import (
    ANALYSIS_SYSTEM_PROMPT,
    ANALYSIS_USER_PROMPT_TEMPLATE,
    ARABIC_SYSTEM_PROMPT,
    ARABIC_USER_PROMPT_TEMPLATE,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
    ENGLISH_SYSTEM_PROMPT,
    ENGLISH_USER_PROMPT_TEMPLATE,
)


# ── Azure client ──────────────────────────────────────────────────────────────

def _get_client() -> openai.AzureOpenAI:
    return openai.AzureOpenAI(
        api_key        = AZURE_OPENAI_API_KEY,
        api_version    = AZURE_OPENAI_API_VERSION,
        azure_endpoint = AZURE_OPENAI_ENDPOINT,
    )


# ── Shared blocking call ──────────────────────────────────────────────────────

def _call_llm(
    system_prompt: str,
    user_prompt:   str,
    temperature:   float = 0.3,
) -> str:
    client   = _get_client()
    response = client.chat.completions.create(
        model       = AZURE_OPENAI_DEPLOYMENT,
        messages    = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature = temperature,
        max_tokens  = 4096,
    )
    return response.choices[0].message.content.strip()


# ── Shared streaming call ─────────────────────────────────────────────────────

def _stream_llm(
    system_prompt: str,
    user_prompt:   str,
    temperature:   float = 0.3,
) -> Iterator[str]:
    """Yields one text token at a time from Azure OpenAI."""
    client = _get_client()
    stream = client.chat.completions.create(
        model       = AZURE_OPENAI_DEPLOYMENT,
        messages    = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature = temperature,
        max_tokens  = 4096,
        stream      = True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            yield delta.content


# ── JSON parser helper ────────────────────────────────────────────────────────

def _parse_json(raw: str) -> dict:
    """Strip markdown fences and parse JSON safely."""
    clean = (
        raw.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    try:
        return json.loads(clean)
    except json.JSONDecodeError as exc:
        return {
            "parse_error": str(exc),
            "raw_output":  raw,
        }


# ══════════════════════════════════════════════════════════════════════════════
# BLOCKING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def refine_arabic(original_transcription: str, context_info: str = "") -> str:
    """
    Return a cleaned Egyptian-Arabic transcript.
    Every line:  [timestamp?]  Agent: text  or  Customer: text
    """
    user_prompt = ARABIC_USER_PROMPT_TEMPLATE.format(
        context_info           = context_info,
        original_transcription = original_transcription,
    )
    return _call_llm(ARABIC_SYSTEM_PROMPT, user_prompt)


def refine_english(original_transcription: str, context_info: str = "") -> str:
    """
    Return a professional English translation/refinement of the transcript.
    Every line:  [timestamp?]  Agent: text  or  Customer: text
    """
    user_prompt = ENGLISH_USER_PROMPT_TEMPLATE.format(
        context_info           = context_info,
        original_transcription = original_transcription,
    )
    return _call_llm(ENGLISH_SYSTEM_PROMPT, user_prompt)


def analyse(original_transcription: str, context_info: str = "") -> dict:
    """
    Return all analysis fields as a dict.

    Keys returned (all matching Scenario 1 user story):
        main_subject, call_outcome, issue_resolution, call_summary,
        keywords, call_category, service, agent_attitude,
        customer_satisfaction, language, call_score, score_breakdown
    """
    user_prompt = ANALYSIS_USER_PROMPT_TEMPLATE.format(
        context_info           = context_info,
        original_transcription = original_transcription,
    )
    raw    = _call_llm(ANALYSIS_SYSTEM_PROMPT, user_prompt, temperature=0.1)
    result = _parse_json(raw)

    # Guarantee call_score is always an int even if LLM returns a string
    if "call_score" in result:
        try:
            result["call_score"] = int(result["call_score"])
        except (ValueError, TypeError):
            result["call_score"] = 0

    # Guarantee keywords is always a list of max 5
    if "keywords" in result and isinstance(result["keywords"], list):
        result["keywords"] = result["keywords"][:5]

    return result


def run_pipeline(
    original_transcription: str,
    context_info: str = "",
) -> dict:
    """
    Run all three steps and return the complete enriched result.

    Input  (from upstream STT API):
        original_transcription : raw transcript text
        context_info           : optional call metadata string

    Output (to downstream post-call details API):
        {
            "original_transcription": str,
            "context_info":           str,
            "transcript_arabic":      str,   ← Scenario 1: Call Transcript (Arabic)
            "transcript_english":     str,   ← Scenario 1: Call Transcript (English)
            "main_subject":           str,   ← Scenario 1: Main Subject
            "call_outcome":           str,   ← Scenario 1: Call Outcome
            "issue_resolution":       str,   ← Scenario 1: Issue Resolution
            "call_summary":           str,   ← Scenario 1: Call Summary
            "keywords":               list,  ← Scenario 1: Keywords (≤5, English)
            "call_category":          str,   ← Scenario 1: Call Category (English)
            "service":                str,   ← Scenario 1: Services (English)
            "agent_attitude":         str,   ← Scenario 1: Agent Attitude (English)
            "customer_satisfaction":  str,   ← Scenario 1: Customer Satisfaction
            "language":               str,
            "call_score":             int,   ← Scenario 1: Call Score (0-100)
            "score_breakdown":        dict,  ← Scenario 1: Question Evaluation
        }
    """
    arabic_transcript  = refine_arabic(original_transcription,  context_info)
    english_transcript = refine_english(original_transcription, context_info)
    analysis           = analyse(original_transcription,        context_info)

    return {
        "original_transcription": original_transcription,
        "context_info":           context_info,
        "transcript_arabic":      arabic_transcript,
        "transcript_english":     english_transcript,
        # ── All analysis fields from the single analyse() call ────────────────
        "main_subject":           analysis.get("main_subject",          ""),
        "call_outcome":           analysis.get("call_outcome",          ""),
        "issue_resolution":       analysis.get("issue_resolution",      ""),
        "call_summary":           analysis.get("call_summary",          ""),
        "keywords":               analysis.get("keywords",              []),
        "call_category":          analysis.get("call_category",         ""),
        "service":                analysis.get("service",               ""),
        "agent_attitude":         analysis.get("agent_attitude",        ""),
        "customer_satisfaction":  analysis.get("customer_satisfaction", ""),
        "language":               analysis.get("language",              ""),
        "call_score":             analysis.get("call_score",            0),
        "score_breakdown":        analysis.get("score_breakdown",       {}),
        # ── Pass through any extra fields the LLM returns ─────────────────────
        **{k: v for k, v in analysis.items()
           if k not in {
               "main_subject", "call_outcome", "issue_resolution", "call_summary",
               "keywords", "call_category", "service", "agent_attitude",
               "customer_satisfaction", "language", "call_score", "score_breakdown",
               "parse_error", "raw_output",
           }
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# STREAMING PIPELINE
# Yields structured event dicts consumed by SSE and WebSocket endpoints.
# ══════════════════════════════════════════════════════════════════════════════

def stream_pipeline(
    original_transcription: str,
    context_info: str = "",
) -> Iterator[dict]:
    """
    Streaming version of run_pipeline.

    Yields event dicts:
      {"event": "start",        "section": "arabic"|"english"|"analysis"}
      {"event": "token",        "section": "...",   "token": "<text>"}
      {"event": "section_done", "section": "...",   "text":  "<full text>"}
      {"event": "done",         ...all pipeline output fields...}
      {"event": "error",        "message": "..."}

    The "done" event contains the identical keys as run_pipeline().
    """
    try:
        collected: dict[str, str] = {}

        # ── Section helper ────────────────────────────────────────────────────
        def stream_section(section: str, sys_p: str, usr_p: str, temp: float = 0.3) -> str:
            yield {"event": "start", "section": section}
            tokens: list[str] = []
            for token in _stream_llm(sys_p, usr_p, temp):
                tokens.append(token)
                yield {"event": "token", "section": section, "token": token}
            full = "".join(tokens).strip()
            yield {"event": "section_done", "section": section, "text": full}
            return full

        # ── 1. Arabic ─────────────────────────────────────────────────────────
        arabic_prompt = ARABIC_USER_PROMPT_TEMPLATE.format(
            context_info           = context_info,
            original_transcription = original_transcription,
        )
        gen = stream_section("arabic", ARABIC_SYSTEM_PROMPT, arabic_prompt)
        arabic_text = ""
        for event in gen:
            if isinstance(event, dict):
                yield event
                if event["event"] == "section_done":
                    arabic_text = event["text"]

        # ── 2. English ────────────────────────────────────────────────────────
        english_prompt = ENGLISH_USER_PROMPT_TEMPLATE.format(
            context_info           = context_info,
            original_transcription = original_transcription,
        )
        gen = stream_section("english", ENGLISH_SYSTEM_PROMPT, english_prompt)
        english_text = ""
        for event in gen:
            if isinstance(event, dict):
                yield event
                if event["event"] == "section_done":
                    english_text = event["text"]

        # ── 3. Analysis ───────────────────────────────────────────────────────
        analysis_prompt = ANALYSIS_USER_PROMPT_TEMPLATE.format(
            context_info           = context_info,
            original_transcription = original_transcription,
        )
        gen = stream_section("analysis", ANALYSIS_SYSTEM_PROMPT, analysis_prompt, temp=0.1)
        analysis_raw = ""
        for event in gen:
            if isinstance(event, dict):
                yield event
                if event["event"] == "section_done":
                    analysis_raw = event["text"]

        # ── Parse analysis and build final result ─────────────────────────────
        analysis = _parse_json(analysis_raw)
        if "call_score" in analysis:
            try:
                analysis["call_score"] = int(analysis["call_score"])
            except (ValueError, TypeError):
                analysis["call_score"] = 0
        if "keywords" in analysis and isinstance(analysis["keywords"], list):
            analysis["keywords"] = analysis["keywords"][:5]

        yield {
            "event":                  "done",
            "original_transcription": original_transcription,
            "context_info":           context_info,
            "transcript_arabic":      arabic_text,
            "transcript_english":     english_text,
            "main_subject":           analysis.get("main_subject",          ""),
            "call_outcome":           analysis.get("call_outcome",          ""),
            "issue_resolution":       analysis.get("issue_resolution",      ""),
            "call_summary":           analysis.get("call_summary",          ""),
            "keywords":               analysis.get("keywords",              []),
            "call_category":          analysis.get("call_category",         ""),
            "service":                analysis.get("service",               ""),
            "agent_attitude":         analysis.get("agent_attitude",        ""),
            "customer_satisfaction":  analysis.get("customer_satisfaction", ""),
            "language":               analysis.get("language",              ""),
            "call_score":             analysis.get("call_score",            0),
            "score_breakdown":        analysis.get("score_breakdown",       {}),
        }

    except Exception as exc:
        yield {"event": "error", "message": str(exc)}
