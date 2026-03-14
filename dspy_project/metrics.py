"""metrics.py — Scoring functions for DSPy optimizers. Zero extra LLM calls."""
from __future__ import annotations
import json, re
from typing import Any

_ARABIC_RE  = re.compile(r'[\u0600-\u06ff]')
_LABEL_RE   = re.compile(r'^(Agent|Customer)\s*:', re.MULTILINE)
_TS_RE      = re.compile(r'\[\d{2}:\d{2}:\d{2}\.\d{3}\]')
_INTRO_WORDS = ["here is","here's","below","certainly","sure","إليك","فيما يلي"]

VALID_OUTCOMES      = {'Resolved','Unresolved','Escalated','Follow-up Needed'}
VALID_CATEGORIES    = {'Inquiry','Complaint','Technical Support','Billing','Sales','Feedback'}
VALID_ATTITUDES     = {'Friendly','Neutral','Rude'}
VALID_SATISFACTIONS = {'Satisfied','Neutral','Dissatisfied'}
VALID_LANGUAGES     = {'Arabic','English','Mixed'}
SCORE_KEYS          = {
    'greeted_professionally','identified_customer_need','provided_accurate_info',
    'maintained_professional_tone','offered_complete_solution',
    'confirmed_resolution','proper_closing',
}

def _nonblank(text: str) -> list[str]:
    return [l for l in text.splitlines() if l.strip()]

def _label_ratio(text: str) -> float:
    lines = _nonblank(text)
    return sum(1 for l in lines if _LABEL_RE.match(l.strip())) / max(len(lines),1)

def _ts_preservation(original: str, refined: str) -> float:
    orig_ts = _TS_RE.findall(original)
    if not orig_ts: return 1.0
    ref_ts = _TS_RE.findall(refined)
    return sum(1 for t in orig_ts if t in ref_ts) / len(orig_ts)

def _length_score(original: str, refined: str) -> float:
    orig_n = len(_nonblank(original))
    ref_n  = len(_nonblank(refined))
    if orig_n == 0: return 0.0
    return max(0.0, 1.0 - abs(ref_n/orig_n - 1.0) / 0.25)

def _arabic_ratio(text: str) -> float:
    chars = re.sub(r'\s','',text)
    return len(_ARABIC_RE.findall(text)) / max(len(chars),1)

def _no_intro(text: str) -> float:
    first = _nonblank(text)[0].lower() if _nonblank(text) else ''
    return 0.0 if any(w in first for w in _INTRO_WORDS) else 1.0

def _parse_breakdown(raw: str) -> dict:
    try:
        return json.loads(raw.strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip())
    except Exception:
        return {}

def arabic_metric(example: Any, pred: Any, trace=None) -> float:
    original = getattr(example,'raw_transcript','') or ''
    refined  = getattr(pred,   'refined_transcript','') or ''
    if not refined.strip(): return 0.0
    return round(
        _label_ratio(refined)*0.35 + _ts_preservation(original,refined)*0.20 +
        _length_score(original,refined)*0.20 + _arabic_ratio(refined)*0.15 +
        _no_intro(refined)*0.10, 4)

def english_metric(example: Any, pred: Any, trace=None) -> float:
    original = getattr(example,'raw_transcript','') or ''
    refined  = getattr(pred,   'refined_transcript','') or ''
    if not refined.strip(): return 0.0
    return round(
        _label_ratio(refined)*0.40 + _ts_preservation(original,refined)*0.20 +
        _length_score(original,refined)*0.20 + _no_intro(refined)*0.10 +
        (1.0 - min(_arabic_ratio(refined),1.0))*0.10, 4)

def analysis_metric(example: Any, pred: Any, trace=None) -> float:
    outcome  = getattr(pred,'call_outcome','') or ''
    category = getattr(pred,'call_category','') or ''
    attitude = getattr(pred,'agent_attitude','') or ''
    sat      = getattr(pred,'customer_satisfaction','') or ''
    lang     = getattr(pred,'language','') or ''
    class_score = (
        (1.0 if outcome.strip()  in VALID_OUTCOMES      else 0.0) +
        (1.0 if category.strip() in VALID_CATEGORIES    else 0.0) +
        (1.0 if attitude.strip() in VALID_ATTITUDES     else 0.0) +
        (1.0 if sat.strip()      in VALID_SATISFACTIONS else 0.0) +
        (1.0 if lang.strip()     in VALID_LANGUAGES     else 0.0)
    ) / 5.0
    kw_list     = [k.strip() for k in (getattr(pred,'keywords','') or '').split(',') if k.strip()]
    kw_score    = min(len(kw_list)/3.0, 1.0)
    try:
        score_int   = int(str(getattr(pred,'call_score','') or '').strip())
        score_valid = 1.0 if 0 <= score_int <= 100 else 0.0
    except (ValueError,TypeError):
        score_valid = 0.0
    bd = _parse_breakdown(getattr(pred,'score_breakdown','') or '')
    if bd:
        keys_ok = sum(1 for k in SCORE_KEYS if k in bd)
        pf_ok   = sum(1 for k in SCORE_KEYS if k in bd and bd[k].get('result') in ('Pass','Fail'))
        breakdown_score = (keys_ok/7*0.5) + (pf_ok/7*0.5)
    else:
        breakdown_score = 0.0
    subject    = getattr(pred,'main_subject','') or ''
    summary    = getattr(pred,'call_summary','') or ''
    resolution = getattr(pred,'issue_resolution','') or ''
    text_score = (
        (1.0 if len(subject.strip())    > 10 else 0.0) +
        (1.0 if len(summary.strip())    > 30 else 0.0) +
        (1.0 if len(resolution.strip()) > 20 else 0.0)
    ) / 3.0
    return round(class_score*0.40 + kw_score*0.10 + score_valid*0.20 + breakdown_score*0.20 + text_score*0.10, 4)

def pipeline_metric(example: Any, pred: Any, trace=None) -> float:
    return round(arabic_metric(example,pred,trace)*0.30 + english_metric(example,pred,trace)*0.30 + analysis_metric(example,pred,trace)*0.40, 4)
