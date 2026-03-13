"""
signatures.py — DSPy Signatures for each pipeline step.

A Signature is DSPy's way of declaring:
  - What goes IN  (InputField)
  - What comes OUT (OutputField)
  - The task description (docstring)

DSPy uses the signature to auto-generate, evaluate, and OPTIMISE
the prompt. You never write prompt text here — DSPy does it.

Three signatures matching our pipeline:
  1. ArabicRefinement   — clean Egyptian-Arabic transcript
  2. EnglishRefinement  — professional English transcript
  3. CallAnalysis       — all Scenario 1 analysis fields
"""

from __future__ import annotations
import dspy

from pathlib import Path
import sys
import os

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
# ══════════════════════════════════════════════════════════════════════════════
# SIGNATURE 1 — Arabic transcript refinement
# ══════════════════════════════════════════════════════════════════════════════

class ArabicRefinement(dspy.Signature):
    """
    You are an expert Egyptian Arabic call center transcript editor for Miraco Company.
    Clean and refine the raw transcript into professional Egyptian dialect Arabic.
    Every output line must start with Agent: or Customer: followed by the text.
    Preserve all timestamps exactly. Never add new information.
    Use natural Egyptian colloquial Arabic — never Modern Standard Arabic (فصحى).
    """

    # ── Inputs ────────────────────────────────────────────────────────────────
    raw_transcript: str = dspy.InputField(
        desc="Raw call transcript from STT — may contain errors, repetition, missing speaker labels"
    )
    context_info: str = dspy.InputField(
        desc="Optional call metadata: Call ID, Agent Name, Date, Duration, Call Type",
        default="",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    refined_transcript: str = dspy.OutputField(
        desc=(
            "Cleaned Egyptian-Arabic transcript. "
            "Each line: [HH:MM:SS.mmm] Agent: text  OR  [HH:MM:SS.mmm] Customer: text. "
            "Timestamps preserved exactly. Egyptian dialect. No preamble."
        )
    )


# ══════════════════════════════════════════════════════════════════════════════
# SIGNATURE 2 — English transcript refinement
# ══════════════════════════════════════════════════════════════════════════════

class EnglishRefinement(dspy.Signature):
    """
    You are a professional call center transcript editor for Miraco Company.
    Translate and refine the Arabic/mixed call transcript into professional English.
    Every output line must start with Agent: or Customer: followed by the text.
    Translate Arabic naturally — never word-for-word. Preserve product names exactly.
    Preserve all timestamps exactly. Never add new information.
    """

    raw_transcript: str = dspy.InputField(
        desc="Raw call transcript from STT — may be Arabic, English, or mixed"
    )
    context_info: str = dspy.InputField(
        desc="Optional call metadata: Call ID, Agent Name, Date, Duration, Call Type",
        default="",
    )

    refined_transcript: str = dspy.OutputField(
        desc=(
            "Professional English transcript. "
            "Each line: [HH:MM:SS.mmm] Agent: text  OR  [HH:MM:SS.mmm] Customer: text. "
            "Natural translation, not literal. Timestamps preserved. No preamble."
        )
    )


# ══════════════════════════════════════════════════════════════════════════════
# SIGNATURE 3 — Full call analysis (all Scenario 1 fields)
# ══════════════════════════════════════════════════════════════════════════════

class CallAnalysis(dspy.Signature):
    """
    You are an expert call center quality analyst for Miraco Company.
    Analyse the call transcript and extract all required fields accurately.
    Score the call using the 7-question evaluation framework.
    All text classification fields must use only the specified allowed values.
    """

    raw_transcript: str = dspy.InputField(
        desc="Raw or refined call transcript to analyse"
    )
    context_info: str = dspy.InputField(
        desc="Optional call metadata to improve analysis accuracy",
        default="",
    )

    # ── Classification outputs ─────────────────────────────────────────────
    main_subject: str = dspy.OutputField(
        desc="One sentence describing the main reason for this call"
    )
    call_outcome: str = dspy.OutputField(
        desc="Exactly one of: Resolved | Unresolved | Escalated | Follow-up Needed"
    )
    issue_resolution: str = dspy.OutputField(
        desc="1-2 sentences explaining how the issue was handled or why it was not resolved"
    )
    call_summary: str = dspy.OutputField(
        desc="2-3 sentence plain-language summary of the full call"
    )
    keywords: str = dspy.OutputField(
        desc="Up to 5 English keywords describing the main topics, comma-separated"
    )
    call_category: str = dspy.OutputField(
        desc="Exactly one of: Inquiry | Complaint | Technical Support | Billing | Sales | Feedback"
    )
    service: str = dspy.OutputField(
        desc="The single main service or product discussed (e.g. AC Installation, Refrigerator Repair)"
    )
    agent_attitude: str = dspy.OutputField(
        desc="Exactly one of: Friendly | Neutral | Rude"
    )
    customer_satisfaction: str = dspy.OutputField(
        desc="Exactly one of: Satisfied | Neutral | Dissatisfied"
    )
    language: str = dspy.OutputField(
        desc="Exactly one of: Arabic | English | Mixed"
    )
    call_score: str = dspy.OutputField(
        desc=(
            "Integer 0-100. Based on 7 questions: "
            "greeted_professionally, identified_customer_need, provided_accurate_info, "
            "maintained_professional_tone, offered_complete_solution, "
            "confirmed_resolution, proper_closing. "
            "Score = round(passed_count / 7 * 100)"
        )
    )
    score_breakdown: str = dspy.OutputField(
        desc=(
            "JSON string with Pass/Fail for each of the 7 questions. Format: "
            '{"greeted_professionally":{"result":"Pass","note":"..."},'
            '"identified_customer_need":{"result":"Pass","note":"..."},'
            '"provided_accurate_info":{"result":"Pass","note":"..."},'
            '"maintained_professional_tone":{"result":"Pass","note":"..."},'
            '"offered_complete_solution":{"result":"Pass","note":"..."},'
            '"confirmed_resolution":{"result":"Fail","note":"..."},'
            '"proper_closing":{"result":"Pass","note":"..."}}'
        )
    )
