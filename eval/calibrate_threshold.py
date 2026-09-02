"""Sweeps candidate `min_top1_score` confidence-gate thresholds against a
`harness.py` results file and reports the false-positive/false-negative
rate at each candidate, to find the cutoff that best separates "should
answer" from "should decline" questions (per CLAUDE.md's eval spec).

Pure post-processing over already-collected data — no live API calls, so
it can be re-run freely against any `harness.py` output without re-querying
the pipeline.

Calibration scope: only records where the pipeline actually ran a vector
search (`top1_score` is not None) are usable — the confidence gate's job is
purely "is retrieval confident enough to attempt an answer," a different
concern from off-limits topics (which short-circuit before retrieval) or
generation/verification correctness (a separate concern from retrieval
confidence). Each usable record is bucketed by its *expected* behavior
label, independent of whether the eventual judged answer was correct — a
record where retrieval found plausible-looking-but-wrong-category content
still tells us whether retrieval confidence cleared the bar, which is all
this gate decides.

Usage:
    python eval/calibrate_threshold.py [--results PATH]
"""

import argparse
import json
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent


def load_results(path: Path) -> list[dict[str, Any]]:
    payload: dict[str, Any] = json.loads(path.read_text())
    results: list[dict[str, Any]] = payload["results"]
    return results


def usable_records(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in results if r.get("top1_score") is not None]


def candidate_thresholds(records: list[dict[str, Any]], score_key: str) -> list[float]:
    """Midpoints between consecutive distinct observed scores, plus the
    boundaries — the only points where a sweep's classification decisions
    can actually change, so this covers every distinct outcome without an
    arbitrary fixed-step grid."""
    scores = sorted({r[score_key] for r in records})
    if not scores:
        return []
    midpoints = [(a + b) / 2 for a, b in zip(scores, scores[1:], strict=False)]
    return [scores[0] - 0.01, *midpoints, scores[-1] + 0.01]


def evaluate_threshold(
    records: list[dict[str, Any]], score_key: str, threshold: float
) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for r in records:
        predicted_answer = r[score_key] >= threshold
        expects_answer = r["expected_behavior"] == "answer"
        if predicted_answer and expects_answer:
            tp += 1
        elif predicted_answer and not expects_answer:
            fp += 1
        elif not predicted_answer and expects_answer:
            fn += 1
        else:
            tn += 1
    total = len(records)
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": (tp + tn) / total if total else 0.0,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "false_negative_rate": fn / (fn + tp) if (fn + tp) else 0.0,
    }


def sweep(records: list[dict[str, Any]], score_key: str) -> list[dict[str, Any]]:
    thresholds = candidate_thresholds(records, score_key)
    return [evaluate_threshold(records, score_key, t) for t in thresholds]


def recommend(sweep_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Best accuracy; ties broken toward the HIGHER threshold — this project
    treats a false positive (answering when it shouldn't) as more costly
    than a false negative (an unnecessary decline), since ungrounded
    answers are the hallucination-risk failure mode CLAUDE.md prioritizes
    guarding against."""
    best_accuracy = max(r["accuracy"] for r in sweep_results)
    tied = [r for r in sweep_results if r["accuracy"] == best_accuracy]
    return max(tied, key=lambda r: r["threshold"])


def print_report(
    name: str, records: list[dict[str, Any]], sweep_results: list[dict[str, Any]]
) -> None:
    print(f"\n=== {name} calibration ({len(records)} usable records) ===")
    answer_scores = [r for r in records if r["expected_behavior"] == "answer"]
    decline_scores = [r for r in records if r["expected_behavior"] == "decline"]
    print(f"  should-answer n={len(answer_scores)}, should-decline n={len(decline_scores)}")
    print(f"  {'threshold':>10} {'acc':>6} {'FP':>4} {'FN':>4} {'FPR':>6} {'FNR':>6}")
    for r in sweep_results:
        print(
            f"  {r['threshold']:>10.4f} {r['accuracy']:>6.1%} {r['fp']:>4} {r['fn']:>4} "
            f"{r['false_positive_rate']:>6.1%} {r['false_negative_rate']:>6.1%}"
        )
    best = recommend(sweep_results)
    print(
        f"  Recommended: {best['threshold']:.4f} "
        f"(accuracy={best['accuracy']:.1%}, FP={best['fp']}, FN={best['fn']})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(EVAL_DIR / "results" / "latest.json"))
    args = parser.parse_args()

    results = load_results(Path(args.results))
    records = usable_records(results)
    if not records:
        print("No records with a top1_score found — nothing to calibrate.")
        return

    top1_sweep = sweep(records, "top1_score")
    print_report("min_top1_score", records, top1_sweep)

    gap_sweep = sweep(records, "score_gap")
    print_report("min_score_gap", records, gap_sweep)
    print(
        "\nNote: min_score_gap is a secondary signal (CLAUDE.md: 'ideally the score gap "
        "between top results'). Only adopt a non-zero value here if its accuracy clearly "
        "beats leaving it permissive — otherwise min_top1_score alone is doing the "
        "separating work and a second threshold just adds a tunable knob with no payoff."
    )


if __name__ == "__main__":
    main()
