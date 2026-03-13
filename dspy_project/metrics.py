"""
metrics.py — Scoring functions for DSPy optimizers.

DSPy optimizers (BootstrapFewShot, MIPROv2, etc.) need a metric function:
  metric(example, prediction, trace=None) -> float   0.0 – 1.0

We define one metric per signature, plus a combined pipeline metric.

Metrics are deterministic (no extra LLM calls) so optimization is fast.
"""

from __future__ import annotations

import json
import re
from typing import Any
from pathlib import Path
import sys
import os

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Helpers ───────────────────────────────────────────────────────────────────

_ARABIC_RE   = re.compile(r'[\u0600-\u06ff]')
_LABEL_RE    = re.compile(r'^(Agent|Customer)\s*:', re.MULTILINE)
_TS_RE       = re.compile(r'\[\d{2}:\d{2}:\d{2}\.\d{3}\]')
_INTRO_WORDS = ['here is', 'here\'s', 'below', 'certainly', 'sure', 'إليك', 'فيما يلي']

VALID_OUTCOMES       = {'Resolved', 'Unresolved', 'Escalated', 'Follow-up Needed'}
VALID_CATEGORIES     = {'Inquiry', 'Complaint', 'Technical Support', 'Billing', 'Sales', 'Feedback'}
VALID_ATTITUDES      = {'Friendly', 'Neutral', 'Rude'}
VALID_SATISFACTIONS  = {'Satisfied', 'Neutral', 'Dissatisfied'}
VALID_LANGUAGES      = {'Arabic', 'English', 'Mixed'}
SCORE_KEYS           = {
    'greeted_professionally', 'identified_customer_need', 'provided_accurate_info',
    'maintained_professional_tone', 'offered_complete_solution',
    'confirmed_resolution', 'proper_closing',
}


def _nonblank(text: str) -> list[str]:
    return [l for l in text.splitlines() if l.strip()]


def _label_ratio(text: str) -> float:
    lines = _nonblank(text)
    if not lines:
        return 0.0
    hits = sum(1 for l in lines if _LABEL_RE.match(l.strip()))
    return hits / len(lines)


def _ts_preservation(original: str, refined: str) -> float:
    orig_ts = _TS_RE.findall(original)
    if not orig_ts:
        return 1.0
    ref_ts = _TS_RE.findall(refined)
    return sum(1 for t in orig_ts if t in ref_ts) / len(orig_ts)


def _length_score(original: str, refined: str) -> float:
    orig_n = len(_nonblank(original))
    ref_n  = len(_nonblank(refined))
    if orig_n == 0:
        return 0.0
    ratio = ref_n / orig_n
    return max(0.0, 1.0 - abs(ratio - 1.0) / 0.25)


def _arabic_ratio(text: str) -> float:
    chars = re.sub(r'\s', '', text)
    if not chars:
        return 0.0
    return len(_ARABIC_RE.findall(text)) / len(chars)


def _no_intro(text: str) -> float:
    first = _nonblank(text)[0].lower() if _nonblank(text) else ''
    return 0.0 if any(w in first for w in _INTRO_WORDS) else 1.0


def _parse_score_breakdown(raw: str) -> dict:
    try:
        clean = raw.strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip()
        return json.loads(clean)
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# METRIC 1 — Arabic refinement
# ══════════════════════════════════════════════════════════════════════════════

def arabic_metric(example: Any, pred: Any, trace=None) -> float:
    """
    Score a predicted Arabic refinement.

    Dimensions:
      label_ratio    (0.35)  — every line has Agent: or Customer:
      ts_preservation(0.20)  — timestamps from original preserved
      length_score   (0.20)  — line count within ±25% of original
      arabic_ratio   (0.15)  — text is actually Arabic
      no_intro       (0.10)  — no preamble sentence
    """
    original = getattr(example, 'raw_transcript', '') or ''
    refined  = getattr(pred,    'refined_transcript', '') or ''

    if not refined.strip():
        return 0.0

    lr  = _label_ratio(refined)
    ts  = _ts_preservation(original, refined)
    ln  = _length_score(original, refined)
    ar  = _arabic_ratio(refined)
    ni  = _no_intro(refined)

    score = lr*0.35 + ts*0.20 + ln*0.20 + ar*0.15 + ni*0.10
    return round(score, 4)


# ══════════════════════════════════════════════════════════════════════════════
# METRIC 2 — English refinement
# ══════════════════════════════════════════════════════════════════════════════

def english_metric(example: Any, pred: Any, trace=None) -> float:
    """
    Score a predicted English refinement.

    Dimensions:
      label_ratio    (0.40)  — every line has Agent: or Customer:
      ts_preservation(0.20)  — timestamps from original preserved
      length_score   (0.20)  — line count within ±25% of original
      no_intro       (0.10)  — no preamble
      not_arabic     (0.10)  — output is English (low Arabic char ratio)
    """
    original = getattr(example, 'raw_transcript', '') or ''
    refined  = getattr(pred,    'refined_transcript', '') or ''

    if not refined.strip():
        return 0.0

    lr      = _label_ratio(refined)
    ts      = _ts_preservation(original, refined)
    ln      = _length_score(original, refined)
    ni      = _no_intro(refined)
    not_ar  = 1.0 - min(_arabic_ratio(refined), 1.0)

    score = lr*0.40 + ts*0.20 + ln*0.20 + ni*0.10 + not_ar*0.10
    return round(score, 4)


# ══════════════════════════════════════════════════════════════════════════════
# METRIC 3 — Call analysis
# ══════════════════════════════════════════════════════════════════════════════

def analysis_metric(example: Any, pred: Any, trace=None) -> float:
    """
    Score the call analysis prediction.

    Dimensions:
      classification (0.40)  — outcome/category/attitude/satisfaction/language valid
      keywords       (0.10)  — 1-5 keywords present
      score_valid    (0.20)  — call_score is 0-100 integer
      breakdown      (0.20)  — score_breakdown has all 7 Pass/Fail keys
      text_quality   (0.10)  — main_subject, summary, resolution non-empty
    """
    # ── Classification fields ─────────────────────────────────────────────────
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

    # ── Keywords ──────────────────────────────────────────────────────────────
    kw_str   = getattr(pred, 'keywords', '') or ''
    kw_list  = [k.strip() for k in kw_str.split(',') if k.strip()]
    kw_score = min(len(kw_list) / 3.0, 1.0)   # at least 3 keywords = full marks

    # ── Call score ────────────────────────────────────────────────────────────
    raw_score = getattr(pred, 'call_score', '') or ''
    try:
        score_int   = int(str(raw_score).strip())
        score_valid = 1.0 if 0 <= score_int <= 100 else 0.0
    except (ValueError, TypeError):
        score_valid = 0.0

    # ── Score breakdown ───────────────────────────────────────────────────────
    breakdown_str = getattr(pred, 'score_breakdown', '') or ''
    breakdown     = _parse_score_breakdown(breakdown_str)
    if breakdown:
        keys_present  = sum(1 for k in SCORE_KEYS if k in breakdown)
        has_pass_fail = sum(
            1 for k in SCORE_KEYS
            if k in breakdown and breakdown[k].get('result') in ('Pass', 'Fail')
        )
        breakdown_score = (keys_present / 7 * 0.5) + (has_pass_fail / 7 * 0.5)
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

    score = (
        class_score     * 0.40 +
        kw_score        * 0.10 +
        score_valid     * 0.20 +
        breakdown_score * 0.20 +
        text_score      * 0.10
    )
    return round(score, 4)


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED PIPELINE METRIC
# ══════════════════════════════════════════════════════════════════════════════

def pipeline_metric(example: Any, pred: Any, trace=None) -> float:
    """
    Combined metric for the full pipeline.
    Weighted average of all three section metrics.
    """
    ar = arabic_metric(example, pred, trace)
    en = english_metric(example, pred, trace)
    an = analysis_metric(example, pred, trace)
    return round(ar * 0.30 + en * 0.30 + an * 0.40, 4)
