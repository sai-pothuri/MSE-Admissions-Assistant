from dataclasses import dataclass


@dataclass(frozen=True)
class ConfidenceConfig:
    min_top1_score: float
    min_score_gap: float


# PROVISIONAL defaults, not yet calibrated against the eval set (Phase 3).
# Picked from a handful of real queries against the live knowledge base:
# clearly relevant questions scored top-1 ~0.45-0.60, clearly irrelevant
# ones ~0.19-0.33. min_score_gap is left permissive (effectively a no-op)
# since 4 samples aren't enough to calibrate a secondary signal — Phase 3's
# eval harness sets the real values for both.
CONFIDENCE = ConfidenceConfig(min_top1_score=0.35, min_score_gap=0.0)

TOP_K = 5

# Categories where the numerical verification pass runs on the generated
# answer before it's returned (CLAUDE.md: reserved for high-stakes
# categories, not applied universally).
NUMERICAL_VERIFICATION_CATEGORIES: frozenset[str] = frozenset({"tuition", "deadlines"})
