"""programs.py — DSPy Programs (ArabicRefiner, EnglishRefiner, CallAnalyser, FullPipeline)."""
from __future__ import annotations
import json
from pathlib import Path
import dspy
from signatures import ArabicRefinement, EnglishRefinement, CallAnalysis

COMPILED_DIR = Path(__file__).parent / "compiled"
COMPILED_DIR.mkdir(exist_ok=True)


class ArabicRefiner(dspy.Module):
    def __init__(self):
        super().__init__()
        self.refine = dspy.ChainOfThought(ArabicRefinement)

    def forward(self, raw_transcript: str, context_info: str = "") -> dspy.Prediction:
        return self.refine(raw_transcript=raw_transcript, context_info=context_info)

    def save_compiled(self, name: str = "arabic_refiner") -> Path:
        path = COMPILED_DIR / f"{name}.json"
        self.save(str(path))
        return path

    @classmethod
    def load_compiled(cls, name: str = "arabic_refiner") -> "ArabicRefiner":
        path = COMPILED_DIR / f"{name}.json"
        prog = cls()
        if path.exists():
            prog.load(str(path))
        return prog


class EnglishRefiner(dspy.Module):
    def __init__(self):
        super().__init__()
        self.refine = dspy.ChainOfThought(EnglishRefinement)

    def forward(self, raw_transcript: str, context_info: str = "") -> dspy.Prediction:
        return self.refine(raw_transcript=raw_transcript, context_info=context_info)

    def save_compiled(self, name: str = "english_refiner") -> Path:
        path = COMPILED_DIR / f"{name}.json"
        self.save(str(path))
        return path

    @classmethod
    def load_compiled(cls, name: str = "english_refiner") -> "EnglishRefiner":
        path = COMPILED_DIR / f"{name}.json"
        prog = cls()
        if path.exists():
            prog.load(str(path))
        return prog


class CallAnalyser(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyse = dspy.ChainOfThought(CallAnalysis)

    def forward(self, raw_transcript: str, context_info: str = "") -> dspy.Prediction:
        pred = self.analyse(raw_transcript=raw_transcript, context_info=context_info)
        if hasattr(pred, 'call_score'):
            try:
                pred.call_score = str(max(0, min(10, int(str(pred.call_score).strip()))))
            except (ValueError, TypeError):
                pred.call_score = "0"
        if hasattr(pred, 'keywords'):
            kws = [k.strip() for k in (pred.keywords or '').split(',') if k.strip()]
            pred.keywords = ', '.join(kws[:5])
        return pred

    def to_analysis_dict(self, pred: dspy.Prediction) -> dict:
        
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
        prog = cls()
        if path.exists():
            prog.load(str(path))
        return prog


class FullPipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.arabic_refiner  = ArabicRefiner()
        self.english_refiner = EnglishRefiner()
        self.call_analyser   = CallAnalyser()

    def forward(self, raw_transcript: str, context_info: str = "") -> dspy.Prediction:
        arabic_pred   = self.arabic_refiner(raw_transcript, context_info)
        english_pred  = self.english_refiner(raw_transcript, context_info)
        analysis_pred = self.call_analyser(raw_transcript, context_info)
        analysis      = self.call_analyser.to_analysis_dict(analysis_pred)
        return dspy.Prediction(
            raw_transcript        = raw_transcript,
            context_info          = context_info,
            transcript_arabic     = getattr(arabic_pred,  'refined_transcript', ''),
            transcript_english    = getattr(english_pred, 'refined_transcript', ''),
            **{k: (', '.join(v) if k=='keywords' and isinstance(v,list)
                   else (str(v) if k=='call_score' else
                         (json.dumps(v) if k=='score_breakdown' and isinstance(v,dict) else v)))
               for k, v in analysis.items()},
        )

    def save_compiled(self, name: str = "full_pipeline") -> Path:
        path = COMPILED_DIR / f"{name}.json"
        self.save(str(path))
        return path

    @classmethod
    def load_compiled(cls, name: str = "full_pipeline") -> "FullPipeline":
        path = COMPILED_DIR / f"{name}.json"
        prog = cls()
        if path.exists():
            prog.load(str(path))
        return prog
