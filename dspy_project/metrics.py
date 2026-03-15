"""
metrics.py — Scoring functions for DSPy optimizers. Zero extra LLM calls.

Score = 0.0 to 1.0 (higher is better).
"""
from __future__ import annotations
import json, re
from typing import Any

_ARABIC_RE  = re.compile(r'[\u0600-\u06ff]')
_LABEL_RE   = re.compile(r'^(Agent|Customer)\s*:', re.MULTILINE)
_TS_RE      = re.compile(r'\[\d{2}:\d{2}:\d{2}[\.,]\d{3}\]')   # detect unwanted timestamps
_INTRO_WORDS = ["here is", "here's", "below", "certainly", "sure", "إليك", "فيما يلي"]

VALID_OUTCOMES      = {'Resolved', 'Unresolved', 'Escalated', 'Follow-up Needed'}
VALID_CATEGORIES    = {'Inquiry', 'Complaint', 'Technical Support', 'Billing', 'Sales', 'Feedback'}
VALID_ATTITUDES     = {'Friendly', 'Neutral', 'Rude'}
VALID_SATISFACTIONS = {'Satisfied', 'Neutral', 'Dissatisfied'}
VALID_LANGUAGES     = {'Arabic', 'English', 'Mixed'}

# Updated 10-question framework
SCORE_KEYS = {
    'greeted_professionally',
    'understood_customer_need',
    'answered_all_questions',
    'provided_accurate_info',
    'satisfied_customer_need',
    'maintained_professional_tone',
    'showed_empathy',
    'demonstrated_product_knowledge',
    'offered_alternatives_if_needed',
    'proper_closing',
}
N_QUESTIONS = len(SCORE_KEYS)   # 10


def _nonblank(text: str) -> list[str]:
    return [l for l in text.splitlines() if l.strip()]

def _label_ratio(text: str) -> float:
    """Fraction of non-blank lines that start with Agent: or Customer:"""
    lines = _nonblank(text)
    if not lines: return 0.0
    return sum(1 for l in lines if _LABEL_RE.match(l.strip())) / len(lines)

def _no_timestamps(text: str) -> float:
    """1.0 if no timestamps found in output, 0.0 if timestamps present."""
    return 0.0 if _TS_RE.search(text) else 1.0

def _length_score(original: str, refined: str) -> float:
    """Line count preserved within ±30% of original."""
    orig_n = len(_nonblank(original))
    ref_n  = len(_nonblank(refined))
    if orig_n == 0: return 0.0
    return max(0.0, 1.0 - abs(ref_n / orig_n - 1.0) / 0.30)

def _arabic_ratio(text: str) -> float:
    chars = re.sub(r'\s', '', text)
    return len(_ARABIC_RE.findall(text)) / max(len(chars), 1)

def _no_intro(text: str) -> float:
    first = _nonblank(text)[0].lower() if _nonblank(text) else ''
    return 0.0 if any(w in first for w in _INTRO_WORDS) else 1.0

def _parse_breakdown(raw: str) -> dict:
    try:
        return json.loads(
            raw.strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip()
        )
    except Exception:
        return {}


# ── Metric 1 — Arabic refinement ─────────────────────────────────────────────

def arabic_metric(example: Any, pred: Any, trace=None) -> float:
    """
    Dimensions:
      label_ratio     0.40 — every line has Agent: or Customer:
      no_timestamps   0.20 — output must NOT contain timestamps
      length_score    0.20 — line count preserved ±30%
      arabic_ratio    0.10 — text is actually Arabic
      no_intro        0.10 — no preamble sentence
    """
    original = getattr(example, 'raw_transcript', '') or ''
    refined  = getattr(pred,    'refined_transcript', '') or ''
    if not refined.strip(): return 0.0

    return round(
        _label_ratio(refined)           * 0.40 +
        _no_timestamps(refined)         * 0.20 +
        _length_score(original, refined)* 0.20 +
        _arabic_ratio(refined)          * 0.10 +
        _no_intro(refined)              * 0.10,
        4,
    )


# ── Metric 2 — English refinement ────────────────────────────────────────────

def english_metric(example: Any, pred: Any, trace=None) -> float:
    """
    Dimensions:
      label_ratio     0.40 — every line has Agent: or Customer:
      no_timestamps   0.20 — output must NOT contain timestamps
      length_score    0.20 — line count preserved ±30%
      not_arabic      0.10 — output is English
      no_intro        0.10 — no preamble sentence
    """
    original = getattr(example, 'raw_transcript', '') or ''
    refined  = getattr(pred,    'refined_transcript', '') or ''
    if not refined.strip(): return 0.0

    return round(
        _label_ratio(refined)            * 0.40 +
        _no_timestamps(refined)          * 0.20 +
        _length_score(original, refined) * 0.20 +
        (1.0 - min(_arabic_ratio(refined), 1.0)) * 0.10 +
        _no_intro(refined)               * 0.10,
        4,
    )


# ── Metric 3 — Call analysis ──────────────────────────────────────────────────

def analysis_metric(example: Any, pred: Any, trace=None) -> float:
    """
    Dimensions:
      classification  0.30 — outcome/category/attitude/satisfaction/language valid
      score_valid     0.20 — call_score is integer 0-10
      breakdown       0.30 — score_breakdown has all 10 keys with Pass/Fail values
      text_quality    0.10 — main_subject/summary/resolution non-empty
      keywords        0.10 — at least 3 keywords present
    """
    # ── Classification ────────────────────────────────────────────────────────
    outcome  = getattr(pred, 'call_outcome',          '') or ''
    category = getattr(pred, 'call_category',         '') or ''
    attitude = getattr(pred, 'agent_attitude',        '') or ''
    sat      = getattr(pred, 'customer_satisfaction', '') or ''
    lang     = getattr(pred, 'language',              '') or ''

    class_score = (
        (1.0 if outcome.strip()  in VALID_OUTCOMES      else 0.0) +
        (1.0 if category.strip() in VALID_CATEGORIES    else 0.0) +
        (1.0 if attitude.strip() in VALID_ATTITUDES     else 0.0) +
        (1.0 if sat.strip()      in VALID_SATISFACTIONS else 0.0) +
        (1.0 if lang.strip()     in VALID_LANGUAGES     else 0.0)
    ) / 5.0

    # ── call_score valid: integer 0-10 ────────────────────────────────────────
    try:
        score_int   = int(str(getattr(pred, 'call_score', '') or '').strip())
        score_valid = 1.0 if 0 <= score_int <= 10 else 0.0
    except (ValueError, TypeError):
        score_valid = 0.0

    # ── score_breakdown: all 10 keys with Pass/Fail ───────────────────────────
    bd = _parse_breakdown(getattr(pred, 'score_breakdown', '') or '')
    if bd:
        keys_present = sum(1 for k in SCORE_KEYS if k in bd)
        pass_fail_ok = sum(
            1 for k in SCORE_KEYS
            if k in bd and bd[k].get('result') in ('Pass', 'Fail')
        )
        breakdown_score = (keys_present / N_QUESTIONS * 0.5) + (pass_fail_ok / N_QUESTIONS * 0.5)
    else:
        breakdown_score = 0.0

    # ── Text quality ──────────────────────────────────────────────────────────
    subject    = getattr(pred, 'main_subject',    '') or ''
    summary    = getattr(pred, 'call_summary',    '') or ''
    resolution = getattr(pred, 'issue_resolution','') or ''
    text_score = (
        (1.0 if len(subject.strip())    > 10 else 0.0) +
        (1.0 if len(summary.strip())    > 30 else 0.0) +
        (1.0 if len(resolution.strip()) > 20 else 0.0)
    ) / 3.0

    # ── Keywords ──────────────────────────────────────────────────────────────
    kw_list  = [k.strip() for k in (getattr(pred, 'keywords', '') or '').split(',') if k.strip()]
    kw_score = min(len(kw_list) / 3.0, 1.0)

    return round(
        class_score     * 0.30 +
        score_valid     * 0.20 +
        breakdown_score * 0.30 +
        text_score      * 0.10 +
        kw_score        * 0.10,
        4,
    )


# ── Combined pipeline metric ──────────────────────────────────────────────────

def pipeline_metric(example: Any, pred: Any, trace=None) -> float:
    return round(
        arabic_metric(example,   pred, trace) * 0.30 +
        english_metric(example,  pred, trace) * 0.30 +
        analysis_metric(example, pred, trace) * 0.40,
        4,
    )
