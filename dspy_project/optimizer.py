"""
optimizer.py — DSPy prompt optimization.

CLI:
  python optimizer.py --program arabic --optimizer bootstrap
  python optimizer.py --program all    --optimizer mipro
  python optimizer.py --show-results
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import dspy
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch

import lm_setup  # triggers configure_dspy() once at module level
from programs  import ArabicRefiner, EnglishRefiner, CallAnalyser
from metrics   import arabic_metric, english_metric, analysis_metric
from trainset  import get_arabic_trainset, get_english_trainset, get_analysis_trainset

RESULTS_FILE = Path(__file__).parent / "optimization_results.json"


def evaluate_program(program, valset: list, metric_fn) -> dict:
    scores, per_example = [], []
    for ex in valset:
        try:
            pred  = program(**{k: getattr(ex, k) for k in ex.inputs()})
            score = metric_fn(ex, pred)
            scores.append(score)
            per_example.append({"input_preview": str(getattr(ex,'raw_transcript',''))[:80]+'…', "score": score})
        except Exception as e:
            per_example.append({"input_preview": "ERROR", "score": 0.0, "error": str(e)})
            scores.append(0.0)
    return {
        "avg_score":   round(sum(scores) / max(len(scores),1), 4),
        "min_score":   round(min(scores, default=0), 4),
        "max_score":   round(max(scores, default=0), 4),
        "n_examples":  len(valset),
        "per_example": per_example,
    }


def optimize_program(
    program_name: str,
    optimizer: str = "bootstrap",
    max_bootstrapped_demos: int = 3,
    num_candidates: int = 8,
    verbose: bool = True,
) -> dict:
    # DSPy configured at import via lm_setup
    CONFIGS = {
        "arabic":   (ArabicRefiner,  get_arabic_trainset,   arabic_metric),
        "english":  (EnglishRefiner, get_english_trainset,  english_metric),
        "analysis": (CallAnalyser,   get_analysis_trainset, analysis_metric),
    }
    if program_name not in CONFIGS:
        raise ValueError(f"Unknown program '{program_name}'. Choose: {list(CONFIGS)}")

    ProgramClass, get_data, metric_fn = CONFIGS[program_name]
    trainset, valset = get_data()

    if verbose:
        print(f"\n{'═'*60}\n  Optimizing: {program_name.upper()}\n  Optimizer: {optimizer}")
        print(f"  Train: {len(trainset)}  Val: {len(valset)}\n{'═'*60}")

    baseline = ProgramClass()
    if verbose: print("\n  [1/3] Evaluating baseline…")
    before = evaluate_program(baseline, valset, metric_fn)
    if verbose: print(f"        Baseline: {before['avg_score']:.3f}")

    t0 = time.time()

    if optimizer == "bootstrap":
        tp = BootstrapFewShot(
            metric=metric_fn,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_bootstrapped_demos,
        )
    elif optimizer == "random_search":
        tp = BootstrapFewShotWithRandomSearch(
            metric=metric_fn,
            max_bootstrapped_demos=max_bootstrapped_demos,
            num_candidate_programs=num_candidates,
        )
    elif optimizer == "mipro":
        try:
            from dspy.teleprompt import MIPROv2
        except ImportError:
            from dspy.teleprompt import MIPRO as MIPROv2
        tp = MIPROv2(metric=metric_fn, auto="light", num_candidates=num_candidates, verbose=verbose)
    else:
        raise ValueError(f"Unknown optimizer '{optimizer}'. Choose: bootstrap | random_search | mipro")

    if verbose: print(f"\n  [2/3] Running {optimizer}…")
    try:
        compiled = tp.compile(ProgramClass(), trainset=trainset)
    except Exception as exc:
        if verbose: print(f"  ⚠️  {exc}")
        compiled = baseline

    elapsed = round(time.time() - t0, 1)

    if verbose: print(f"\n  [3/3] Evaluating compiled…")
    after = evaluate_program(compiled, valset, metric_fn)
    improvement = round(after['avg_score'] - before['avg_score'], 4)

    if verbose:
        print(f"\n  Before: {before['avg_score']:.3f}  After: {after['avg_score']:.3f}")
        print(f"  Δ:      {improvement:+.3f}  Time: {elapsed}s")

    # Save compiled
    save_name = f"{program_name}_{optimizer}"
    if hasattr(compiled, 'save_compiled'):
        saved_path = compiled.save_compiled(save_name)
    else:
        from programs import COMPILED_DIR
        saved_path = COMPILED_DIR / f"{save_name}.json"
        compiled.save(str(saved_path))

    if verbose: print(f"  Saved: {saved_path}")

    result = {
        "program": program_name, "optimizer": optimizer,
        "timestamp": __import__('datetime').datetime.utcnow().isoformat(),
        "elapsed_s": elapsed, "before": before, "after": after,
        "improvement": improvement, "compiled_path": str(saved_path),
    }
    _append_result(result)
    return result


def _append_result(result: dict) -> None:
    existing = []
    if RESULTS_FILE.exists():
        try: existing = json.loads(RESULTS_FILE.read_text())
        except Exception: pass
    existing.append(result)
    RESULTS_FILE.write_text(json.dumps(existing[-100:], indent=2, ensure_ascii=False))


def load_results() -> list[dict]:
    if not RESULTS_FILE.exists(): return []
    try: return json.loads(RESULTS_FILE.read_text())
    except Exception: return []


def get_best_compiled(program_name: str) -> str | None:
    results = [r for r in load_results() if r.get("program") == program_name]
    if not results: return None
    best = max(results, key=lambda r: r.get("after",{}).get("avg_score", 0))
    return best.get("compiled_path")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize DSPy programs")
    parser.add_argument("--program",   default="arabic", choices=["arabic","english","analysis","all"])
    parser.add_argument("--optimizer", default="bootstrap", choices=["bootstrap","random_search","mipro"])
    parser.add_argument("--demos",     type=int, default=3)
    parser.add_argument("--candidates",type=int, default=8)
    parser.add_argument("--show-results", action="store_true")
    args = parser.parse_args()

    # DSPy configured at import via lm_setup
    if args.show_results:
        results = load_results()
        if not results:
            print("No results yet.")
        else:
            print(f"\n{'─'*60}")
            print(f"{'Program':<12} {'Optimizer':<16} {'Before':>7} {'After':>7} {'Δ':>7}  {'Time':>6}")
            print(f"{'─'*60}")
            for r in results[-20:]:
                b = r.get('before',{}).get('avg_score',0)
                a = r.get('after',{}).get('avg_score',0)
                d = r.get('improvement', a-b)
                t = r.get('elapsed_s', 0)
                print(f"{r['program']:<12} {r['optimizer']:<16} {b:>7.3f} {a:>7.3f} {d:>+7.3f}  {t:>5.1f}s")
        sys.exit(0)

    programs = ["arabic","english","analysis"] if args.program == "all" else [args.program]
    for prog in programs:
        optimize_program(prog, args.optimizer, args.demos, args.candidates)
