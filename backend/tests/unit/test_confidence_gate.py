from app.config.thresholds import ConfidenceConfig
from app.services.guardrails.confidence_gate import evaluate

CONFIG = ConfidenceConfig(min_top1_score=0.4, min_score_gap=0.02)


def test_passes_when_above_both_thresholds():
    result = evaluate(top1_score=0.6, score_gap=0.05, config=CONFIG)
    assert result.passed is True
    assert result.reason is None


def test_fails_when_top1_score_below_threshold():
    result = evaluate(top1_score=0.3, score_gap=0.05, config=CONFIG)
    assert result.passed is False
    assert result.reason == "top1_score_below_threshold"


def test_fails_when_score_gap_below_threshold():
    result = evaluate(top1_score=0.6, score_gap=0.01, config=CONFIG)
    assert result.passed is False
    assert result.reason == "score_gap_below_threshold"


def test_passes_at_exact_threshold_boundary():
    result = evaluate(top1_score=0.4, score_gap=0.02, config=CONFIG)
    assert result.passed is True
