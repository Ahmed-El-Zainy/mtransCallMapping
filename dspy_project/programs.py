"""
programs.py — DSPy Programs (the actual LLM modules).

A DSPy Program wraps one or more Predict/ChainOfThought/ReAct calls.
The optimizer tunes the prompts INSIDE these programs automatically.

Three programs:
  ArabicRefiner   — wraps ArabicRefinement signature
  EnglishRefiner  — wraps EnglishRefinement signature
  CallAnalyser    — wraps CallAnalysis with ChainOfThought for better reasoning
  FullPipeline    — runs all three in sequence, returns unified result dict

Compiled (optimized) versions are saved to:
  dspy_module/compiled/<program>_<optimizer>.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import dspy

from dspy_module.signatures import ArabicRefinement, EnglishRefinement, CallAnalysis

COMPILED_DIR = Path(__file__).parent / "compiled"
COMPILED_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# PROGRAM 1 — Arabic Refiner
# ══════════════════════════════════════════════════════════════════════════════

class ArabicRefiner(dspy.Module):
    """
    DSPy module for Egyptian-Arabic transcript refinement.

    Uses ChainOfThought to encourage the model to reason about
    speaker identification before writing the refined output.
    """

    def __init__(self):
        super().__init__()
        self.refine = dspy.ChainOfThought(ArabicRefinement)

    def forward(self, raw_transcript: str, context_info: str = "") -> dspy.Prediction:
        return self.refine(
            raw_transcript = raw_transcript,
            context_info   = context_info,
        )

    # ── Serialization ─────────────────────────────────────────────────────────
    def save_compiled(self, name: str = "arabic_refiner") -> Path:
        path = COMPILED_DIR / f"{name}.json"
        self.save(str(path))
        return path

    @classmethod
    def load_compiled(cls, name: str = "arabic_refiner") -> "ArabicRefiner":
        path = COMPILED_DIR / f"{name}.json"
        program = cls()
        if path.exists():
            program.load(str(path))
        return program


# ══════════════════════════════════════════════════════════════════════════════
# PROGRAM 2 — English Refiner
# ══════════════════════════════════════════════════════════════════════════════

class EnglishRefiner(dspy.Module):
    """
    DSPy module for professional English transcript refinement/translation.
    """

    def __init__(self):
        super().__init__()
        self.refine = dspy.ChainOfThought(EnglishRefinement)

    def forward(self, raw_transcript: str, context_info: str = "") -> dspy.Prediction:
        return self.refine(
            raw_transcript = raw_transcript,
            context_info   = context_info,
        )

    def save_compiled(self, name: str = "english_refiner") -> Path:
        path = COMPILED_DIR / f"{name}.json"
        self.save(str(path))
        return path

    @classmethod
    def load_compiled(cls, name: str = "english_refiner") -> "EnglishRefiner":
        path = COMPILED_DIR / f"{name}.json"
        program = cls()
        if path.exists():
            program.load(str(path))
        return program


# ══════════════════════════════════════════════════════════════════════════════
# PROGRAM 3 — Call Analyser
# ══════════════════════════════════════════════════════════════════════════════

class CallAnalyser(dspy.Module):
    """
    DSPy module for full call analysis.

    Uses ChainOfThought — the model first reasons through the call
    before producing each classification output, improving accuracy
    on constrained fields (category, attitude, score).
    """

    def __init__(self):
        super().__init__()
        self.analyse = dspy.ChainOfThought(CallAnalysis)

    def forward(self, raw_transcript: str, context_info: str = "") -> dspy.Prediction:
        pred = self.analyse(
            raw_transcript = raw_transcript,
            context_info   = context_info,
        )

        # Post-process: ensure call_score is clamped to 0-100
        if hasattr(pred, 'call_score'):
            try:
                pred.call_score = str(max(0, min(100, int(str(pred.call_score).strip()))))
            except (ValueError, TypeError):
                pred.call_score = "0"

        # Post-process: ensure keywords max 5
        if hasattr(pred, 'keywords'):
            kws = [k.strip() for k in (pred.keywords or '').split(',') if k.strip()]
            pred.keywords = ', '.join(kws[:5])

        return pred

    def to_analysis_dict(self, pred: dspy.Prediction) -> dict:
        """Convert DSPy prediction to the analysis dict expected by run_pipeline."""
        breakdown_raw = getattr(pred, 'score_breakdown', '{}') or '{}'
        try:
            clean = (breakdown_raw.strip()
                     .removeprefix('```json').removeprefix('```')
                     .removesuffix('```').strip())
            breakdown = json.loads(clean)
        except Exception:
            breakdown = {}

        kw_str  = getattr(pred, 'keywords', '') or ''
        kw_list = [k.strip() for k in kw_str.split(',') if k.strip()][:5]

        try:
            score = int(str(getattr(pred, 'call_score', 0)).strip())
        except (ValueError, TypeError):
            score = 0

        return {
            'main_subject':          getattr(pred, 'main_subject',          ''),
            'call_outcome':          getattr(pred, 'call_outcome',          ''),
            'issue_resolution':      getattr(pred, 'issue_resolution',      ''),
            'call_summary':          getattr(pred, 'call_summary',          ''),
            'keywords':              kw_list,
            'call_category':         getattr(pred, 'call_category',         ''),
            'service':               getattr(pred, 'service',               ''),
            'agent_attitude':        getattr(pred, 'agent_attitude',        ''),
            'customer_satisfaction': getattr(pred, 'customer_satisfaction', ''),
            'language':              getattr(pred, 'language',              ''),
            'call_score':            score,
            'score_breakdown':       breakdown,
        }

    def save_compiled(self, name: str = "call_analyser") -> Path:
        path = COMPILED_DIR / f"{name}.json"
        self.save(str(path))
        return path

    @classmethod
    def load_compiled(cls, name: str = "call_analyser") -> "CallAnalyser":
        path = COMPILED_DIR / f"{name}.json"
        program = cls()
        if path.exists():
            program.load(str(path))
        return program


# ══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE — all three programs in sequence
# ══════════════════════════════════════════════════════════════════════════════

class FullPipeline(dspy.Module):
    """
    Complete refiner pipeline as a single DSPy module.

    Runs:
      1. ArabicRefiner   → transcript_arabic
      2. EnglishRefiner  → transcript_english
      3. CallAnalyser    → all analysis fields

    The optimizer can tune all three sub-programs simultaneously.
    """

    def __init__(self):
        super().__init__()
        self.arabic_refiner  = ArabicRefiner()
        self.english_refiner = EnglishRefiner()
        self.call_analyser   = CallAnalyser()

    def forward(
        self,
        raw_transcript: str,
        context_info:   str = "",
    ) -> dspy.Prediction:
        arabic_pred  = self.arabic_refiner(raw_transcript, context_info)
        english_pred = self.english_refiner(raw_transcript, context_info)
        analysis_pred = self.call_analyser(raw_transcript, context_info)

        analysis = self.call_analyser.to_analysis_dict(analysis_pred)

        return dspy.Prediction(
            raw_transcript         = raw_transcript,
            context_info           = context_info,
            refined_transcript     = getattr(arabic_pred,  'refined_transcript', ''),
            # re-use field name for english (caller differentiates by context)
            transcript_arabic      = getattr(arabic_pred,  'refined_transcript', ''),
            transcript_english     = getattr(english_pred, 'refined_transcript', ''),
            # analysis fields (flat, so metrics can access them directly)
            main_subject           = analysis.get('main_subject',          ''),
            call_outcome           = analysis.get('call_outcome',          ''),
            issue_resolution       = analysis.get('issue_resolution',      ''),
            call_summary           = analysis.get('call_summary',          ''),
            keywords               = ', '.join(analysis.get('keywords', [])),
            call_category          = analysis.get('call_category',         ''),
            service                = analysis.get('service',               ''),
            agent_attitude         = analysis.get('agent_attitude',        ''),
            customer_satisfaction  = analysis.get('customer_satisfaction', ''),
            language               = analysis.get('language',              ''),
            call_score             = str(analysis.get('call_score', 0)),
            score_breakdown        = json.dumps(analysis.get('score_breakdown', {})),
        )

    def to_pipeline_dict(self, pred: dspy.Prediction, original_transcription: str, context_info: str) -> dict:
        """Convert FullPipeline prediction to the run_pipeline() output dict."""
        try:
            breakdown = json.loads(getattr(pred, 'score_breakdown', '{}') or '{}')
        except Exception:
            breakdown = {}

        kw_str  = getattr(pred, 'keywords', '') or ''
        kw_list = [k.strip() for k in kw_str.split(',') if k.strip()][:5]

        try:
            score = int(str(getattr(pred, 'call_score', 0)).strip())
        except (ValueError, TypeError):
            score = 0

        return {
            'original_transcription': original_transcription,
            'context_info':           context_info,
            'transcript_arabic':      getattr(pred, 'transcript_arabic',     ''),
            'transcript_english':     getattr(pred, 'transcript_english',    ''),
            'main_subject':           getattr(pred, 'main_subject',          ''),
            'call_outcome':           getattr(pred, 'call_outcome',          ''),
            'issue_resolution':       getattr(pred, 'issue_resolution',      ''),
            'call_summary':           getattr(pred, 'call_summary',          ''),
            'keywords':               kw_list,
            'call_category':          getattr(pred, 'call_category',         ''),
            'service':                getattr(pred, 'service',               ''),
            'agent_attitude':         getattr(pred, 'agent_attitude',        ''),
            'customer_satisfaction':  getattr(pred, 'customer_satisfaction', ''),
            'language':               getattr(pred, 'language',              ''),
            'call_score':             score,
            'score_breakdown':        breakdown,
        }

    def save_compiled(self, name: str = "full_pipeline") -> Path:
        path = COMPILED_DIR / f"{name}.json"
        self.save(str(path))
        return path

    @classmethod
    def load_compiled(cls, name: str = "full_pipeline") -> "FullPipeline":
        path = COMPILED_DIR / f"{name}.json"
        program = cls()
        if path.exists():
            program.load(str(path))
        return program
