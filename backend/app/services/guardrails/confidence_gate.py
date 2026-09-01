from dataclasses import dataclass

from app.config.thresholds import ConfidenceConfig


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reason: str | None = None


def evaluate(top1_score: float, score_gap: float, config: ConfidenceConfig) -> GateResult:
    """Pure decision logic, no I/O — the caller is responsible for computing
    top1_score and score_gap from actual search results (see
    `app.services.retrieval.vector_search.top1_and_gap`)."""
    if top1_score < config.min_top1_score:
        return GateResult(passed=False, reason="top1_score_below_threshold")
    if score_gap < config.min_score_gap:
        return GateResult(passed=False, reason="score_gap_below_threshold")
    return GateResult(passed=True)
