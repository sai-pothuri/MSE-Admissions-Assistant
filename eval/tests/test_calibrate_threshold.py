import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibrate_threshold import (  # noqa: E402
    candidate_thresholds,
    evaluate_threshold,
    recommend,
    sweep,
    usable_records,
)


def _record(expected_behavior, top1_score, score_gap=0.0):
    return {
        "expected_behavior": expected_behavior,
        "top1_score": top1_score,
        "score_gap": score_gap,
    }


def test_usable_records_excludes_records_without_a_score():
    records = [
        _record("answer", 0.5),
        {"expected_behavior": "decline", "top1_score": None, "score_gap": None},
    ]

    assert usable_records(records) == [records[0]]


def test_candidate_thresholds_returns_midpoints_plus_boundaries():
    records = [_record("answer", 0.2), _record("decline", 0.4), _record("answer", 0.6)]

    candidates = candidate_thresholds(records, "top1_score")

    assert candidates[0] < 0.2
    assert candidates[-1] > 0.6
    assert any(abs(c - 0.3) < 1e-9 for c in candidates)
    assert any(abs(c - 0.5) < 1e-9 for c in candidates)


def test_candidate_thresholds_empty_for_no_records():
    assert candidate_thresholds([], "top1_score") == []


def test_evaluate_threshold_perfectly_separates_clean_data():
    records = [
        _record("answer", 0.6),
        _record("answer", 0.7),
        _record("decline", 0.1),
        _record("decline", 0.2),
    ]

    result = evaluate_threshold(records, "top1_score", threshold=0.4)

    assert result == {
        "threshold": 0.4,
        "tp": 2,
        "fp": 0,
        "tn": 2,
        "fn": 0,
        "accuracy": 1.0,
        "false_positive_rate": 0.0,
        "false_negative_rate": 0.0,
    }


def test_evaluate_threshold_counts_false_positives_and_negatives():
    # At threshold 0.5: the decline@0.6 record is a false positive (gate
    # would let it through), and the answer@0.4 record is a false negative
    # (gate would incorrectly block it).
    records = [_record("answer", 0.4), _record("decline", 0.6)]

    result = evaluate_threshold(records, "top1_score", threshold=0.5)

    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["accuracy"] == 0.0


def test_sweep_finds_a_threshold_with_perfect_accuracy_when_scores_are_separable():
    records = [
        _record("answer", 0.6),
        _record("answer", 0.7),
        _record("decline", 0.1),
        _record("decline", 0.2),
    ]

    results = sweep(records, "top1_score")

    assert any(r["accuracy"] == 1.0 for r in results)


def test_recommend_breaks_ties_toward_the_higher_threshold():
    """Two thresholds both give perfect accuracy on this gap-separated data
    — recommend() should prefer the more conservative (higher) one, since a
    false positive (answering when it shouldn't) is the costlier failure
    mode for this project."""
    records = [_record("answer", 0.5), _record("decline", 0.1)]

    results = sweep(records, "top1_score")
    best = recommend(results)

    # Every threshold strictly between 0.1 and 0.5 achieves perfect
    # accuracy here — recommend() must pick the largest of those.
    perfect = [r["threshold"] for r in results if r["accuracy"] == 1.0]
    assert best["threshold"] == max(perfect)
