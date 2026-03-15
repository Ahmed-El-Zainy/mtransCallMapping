"""signatures.py — DSPy Signatures for each pipeline step."""
from __future__ import annotations
import dspy


class ArabicRefinement(dspy.Signature):
    """
    You are an expert Egyptian Arabic call center transcript editor for Miraco Company.
    Clean and refine the raw transcript into professional Egyptian dialect Arabic.
    Every output line must start with Agent: or Customer: followed by the text.
    Never add new information. Never add timestamps.
    Use natural Egyptian colloquial Arabic — never Modern Standard Arabic (فصحى).
    """
    raw_transcript: str = dspy.InputField(
        desc="Raw call transcript from STT — may contain errors, repetition, missing speaker labels"
    )
    context_info: str = dspy.InputField(
        desc="Optional call metadata: Call ID, Agent Name, Date, Duration, Call Type",
        default="",
    )
    refined_transcript: str = dspy.OutputField(
        desc=(
            "Cleaned Egyptian-Arabic transcript. "
            "Each line starts with exactly 'Agent: ' or 'Customer: ' followed by the text. "
            "No timestamps. Egyptian dialect only. No preamble — start directly with line 1."
        )
    )


class EnglishRefinement(dspy.Signature):
    """
    You are a professional call center transcript editor for Miraco Company.
    Translate and refine the Arabic/mixed call transcript into professional English.
    Every output line must start with Agent: or Customer: followed by the text.
    Translate Arabic naturally — never word-for-word. Preserve product names exactly.
    Never add new information. Never add timestamps.
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
            "Each line starts with exactly 'Agent: ' or 'Customer: ' followed by the text. "
            "No timestamps. Natural translation, not literal. No preamble — start directly with line 1."
        )
    )


class CallAnalysis(dspy.Signature):
    """
    You are an expert call center quality analyst for Miraco Company.
    Analyse the call transcript and extract all required fields accurately.
    Score the call based on how well the agent answered the customer's questions,
    satisfied their needs, and maintained a professional conversation throughout.
    All text classification fields must use only the specified allowed values.
    """
    raw_transcript: str = dspy.InputField(
        desc="Raw or refined call transcript to analyse"
    )
    context_info: str = dspy.InputField(
        desc="Optional call metadata to improve analysis accuracy",
        default="",
    )
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
            "Integer 0-10 quality score based on 10 evaluation questions. "
            "Score = round(passed_count / 10 * 10). "
            "Questions cover: greeting, understanding customer need, answering every question asked, "
            "providing accurate information, satisfying the customer, professional tone, "
            "empathy, product knowledge, offering alternatives, and proper closing."
        )
    )
    score_breakdown: str = dspy.OutputField(
        desc=(
            'JSON string with Pass/Fail and note for each of 10 quality questions. '
            'Keys: greeted_professionally, understood_customer_need, answered_all_questions, '
            'provided_accurate_info, satisfied_customer_need, maintained_professional_tone, '
            'showed_empathy, demonstrated_product_knowledge, offered_alternatives_if_needed, proper_closing. '
            'Format: {"greeted_professionally":{"result":"Pass","note":"short reason"}, ...}'
        )
    )
