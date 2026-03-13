"""
optimizer.py — DSPy prompt optimization.

Optimizers available:
  BootstrapFewShot    Fast, no LLM calls to optimize — just runs examples
  BootstrapFewShotWithRandomSearch  Tries multiple random seeds, picks best
  MIPROv2             Powerful — generates candidate instructions + few-shots
                      Uses LLM calls during optimization (costs money)

Usage:
  # CLI
  python -m dspy_module.optimizer --program arabic --optimizer bootstrap
  python -m dspy_module.optimizer --program all    --optimizer mipro

  # From code
  from dspy_module.optimizer import optimize_program, evaluate_program
  result = optimize_program('arabic', optimizer='bootstrap')
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import dspy
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch

from lm_setup   import configure_dspy
from programs   import ArabicRefiner, EnglishRefiner, CallAnalyser
from metrics    import arabic_metric, english_metric, analysis_metric
from trainset   import get_arabic_trainset, get_english_trainset, get_analysis_trainset

RESULTS_FILE = ROOT / "dspy_module" / "optimization_results.json"


# ── Evaluate a program against a validation set ───────────────────────────────

def evaluate_program(program, valset: list, metric_fn) -> dict:
    """
    Run program on every example in valset and score with metric_fn.
    Returns dict with scores and average.
    """
    scores     = []
    per_example = []

    for ex in valset:
        try:
            pred  = program(**{k: getattr(ex, k) for k in ex.inputs()})
            score = metric_fn(ex, pred)
            scores.append(score)
            per_example.append({
                "input_preview": str(getattr(ex, 'raw_transcript', ''))[:80] + '…',
                "score": score,
            })
        except Exception as e:
            per_example.append({"input_preview": "ERROR", "score": 0.0, "error": str(e)})
            scores.append(0.0)

    return {
        "avg_score":   round(sum(scores) / max(len(scores), 1), 4),
        "min_score":   round(min(scores, default=0), 4),
        "max_score":   round(max(scores, default=0), 4),
        "n_examples":  len(valset),
        "per_example": per_example,
    }


# ── Optimize one program ──────────────────────────────────────────────────────

def optimize_program(
    program_name: str,           # "arabic" | "english" | "analysis"
    optimizer:    str = "bootstrap",
    max_bootstrapped_demos: int = 3,
    num_candidates: int = 8,
    verbose: bool = True,
) -> dict:
    """
    Optimize a DSPy program and save the compiled version.

    Returns a result dict with before/after scores.
    """
    configure_dspy()

    # ── Select program + data + metric ────────────────────────────────────────
    CONFIGS = {
        "arabic":  (ArabicRefiner,  get_arabic_trainset,  arabic_metric),
        "english": (EnglishRefiner, get_english_trainset, english_metric),
        "analysis":(CallAnalyser,   get_analysis_trainset, analysis_metric),
    }

    if program_name not in CONFIGS:
        raise ValueError(f"Unknown program: '{program_name}'. Choose from: {list(CONFIGS)}")

    ProgramClass, get_data, metric_fn = CONFIGS[program_name]
    trainset, valset = get_data()

    if verbose:
        print(f"\n{'═'*60}")
        print(f"  Optimizing: {program_name.upper()}")
        print(f"  Optimizer:  {optimizer}")
        print(f"  Train:      {len(trainset)} examples")
        print(f"  Val:        {len(valset)} examples")
        print(f"{'═'*60}")

    # ── Baseline (uncompiled) ─────────────────────────────────────────────────
    baseline = ProgramClass()
    if verbose: print("\n  [1/3] Evaluating baseline…")
    before = evaluate_program(baseline, valset, metric_fn)
    if verbose: print(f"        Baseline avg score: {before['avg_score']:.3f}")

    # ── Select optimizer ──────────────────────────────────────────────────────
    t0 = time.time()

    if optimizer == "bootstrap":
        teleprompter = BootstrapFewShot(
            metric               = metric_fn,
            max_bootstrapped_demos = max_bootstrapped_demos,
            max_labeled_demos    = max_bootstrapped_demos,
        )
    elif optimizer == "random_search":
        teleprompter = BootstrapFewShotWithRandomSearch(
            metric               = metric_fn,
            max_bootstrapped_demos = max_bootstrapped_demos,
            num_candidate_programs = num_candidates,
        )
    elif optimizer == "mipro":
        try:
            from dspy.teleprompt import MIPROv2
        except ImportError:
            from dspy.teleprompt import MIPRO as MIPROv2
        teleprompter = MIPROv2(
            metric        = metric_fn,
            auto          = "light",
            num_candidates = num_candidates,
            verbose       = verbose,
        )
    else:
        raise ValueError(f"Unknown optimizer: '{optimizer}'. Choose: bootstrap | random_search | mipro")

    # ── Compile (optimize) ────────────────────────────────────────────────────
    if verbose: print(f"\n  [2/3] Running {optimizer} optimization…")
    try:
        compiled = teleprompter.compile(ProgramClass(), trainset=trainset)
    except Exception as exc:
        if verbose: print(f"  ⚠️  Optimizer error: {exc}")
        compiled = baseline   # fall back to baseline

    elapsed = round(time.time() - t0, 1)

    # ── Evaluate compiled ─────────────────────────────────────────────────────
    if verbose: print(f"\n  [3/3] Evaluating compiled program…")
    after = evaluate_program(compiled, valset, metric_fn)

    improvement = round(after['avg_score'] - before['avg_score'], 4)
    if verbose:
        print(f"\n  {'─'*50}")
        print(f"  Before: {before['avg_score']:.3f}")
        print(f"  After:  {after['avg_score']:.3f}")
        print(f"  Δ gain: {improvement:+.3f}  ({'✅ improved' if improvement > 0 else '⚠️  no improvement'})")
        print(f"  Time:   {elapsed}s")

    # ── Save compiled program ─────────────────────────────────────────────────
    save_name = f"{program_name}_{optimizer}"
    if hasattr(compiled, 'save_compiled'):
        saved_path = compiled.save_compiled(save_name)
    else:
        saved_path = ProgramClass.load_compiled.__func__.__globals__.get(
            'COMPILED_DIR', Path('.')
        ) / f"{save_name}.json"
        compiled.save(str(saved_path))

    if verbose:
        print(f"  Saved:  {saved_path}")

    result = {
        "program":     program_name,
        "optimizer":   optimizer,
        "timestamp":   __import__('datetime').datetime.utcnow().isoformat(),
        "elapsed_s":   elapsed,
        "before":      before,
        "after":       after,
        "improvement": improvement,
        "compiled_path": str(saved_path),
    }

    # ── Persist results ───────────────────────────────────────────────────────
    _append_result(result)
    return result


def _append_result(result: dict) -> None:
    existing = []
    if RESULTS_FILE.exists():
        try:
            existing = json.loads(RESULTS_FILE.read_text())
        except Exception:
            pass
    existing.append(result)
    existing = existing[-100:]
    RESULTS_FILE.write_text(json.dumps(existing, indent=2, ensure_ascii=False))


def load_results() -> list[dict]:
    if not RESULTS_FILE.exists():
        return []
    try:
        return json.loads(RESULTS_FILE.read_text())
    except Exception:
        return []


def get_best_compiled(program_name: str) -> str | None:
    """Return the compiled JSON path with the best after-score for a program."""
    results = [r for r in load_results() if r.get("program") == program_name]
    if not results:
        return None
    best = max(results, key=lambda r: r.get("after", {}).get("avg_score", 0))
    return best.get("compiled_path")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize DSPy programs")
    parser.add_argument("--program",   default="arabic",
                        choices=["arabic","english","analysis","all"],
                        help="Which program to optimize")
    parser.add_argument("--optimizer", default="bootstrap",
                        choices=["bootstrap","random_search","mipro"],
                        help="Which DSPy optimizer to use")
    parser.add_argument("--demos",    type=int, default=3,
                        help="Max bootstrapped demos")
    parser.add_argument("--candidates",type=int, default=8,
                        help="Candidate programs for random_search/mipro")
    parser.add_argument("--show-results", action="store_true",
                        help="Show past optimization results")
    args = parser.parse_args()

    configure_dspy()

    if args.show_results:
        results = load_results()
        if not results:
            print("No results yet. Run without --show-results to optimize.")
        else:
            print(f"\n{'─'*60}")
            print(f"{'Program':<12} {'Optimizer':<15} {'Before':>7} {'After':>7} {'Δ':>7}  {'Time':>6}")
            print(f"{'─'*60}")
            for r in results[-20:]:
                b = r.get('before',{}).get('avg_score',0)
                a = r.get('after',{}).get('avg_score',0)
                d = r.get('improvement', a-b)
                t = r.get('elapsed_s', 0)
                print(f"{r['program']:<12} {r['optimizer']:<15} {b:>7.3f} {a:>7.3f} {d:>+7.3f}  {t:>5.1f}s")
        sys.exit(0)

    programs = ["arabic","english","analysis"] if args.program == "all" else [args.program]
    for prog in programs:
        optimize_program(
            program_name            = prog,
            optimizer               = args.optimizer,
            max_bootstrapped_demos  = args.demos,
            num_candidates          = args.candidates,
        )
