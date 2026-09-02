from dataclasses import dataclass


@dataclass(frozen=True)
class ConfidenceConfig:
    min_top1_score: float
    min_score_gap: float


# Calibrated in Phase 3 against the 76-question eval set (eval/questions.jsonl)
# via `eval/calibrate_threshold.py`, sweeping every observed top1_score/
# score_gap value against each question's expected answer/decline label.
#
# min_top1_score=0.40: the should-answer and should-decline score
# distributions substantially overlap in this corpus (the lowest
# should-answer score, ~0.40, sits below several should-decline scores in
# the ~0.43-0.62 range) — no threshold gets both false positives and false
# negatives to zero. 0.40 is the highest threshold that still keeps false
# negatives at zero (every genuinely answerable eval question clears it),
# chosen deliberately over the sweep's raw-accuracy-maximizing value
# (~0.43, which trades one blocked answerable question for a few fewer
# gate-passes). Reasoning: a blocked question has no fallback — the user
# just gets a decline. A gate false-positive does have a fallback — the
# generation system prompt's own "if the context doesn't clearly support
# an answer, say so explicitly" instruction, which the eval observed
# reliably catching out-of-scope questions that made it past the gate.
# So minimizing false negatives is the higher-value target here.
#
# min_score_gap=0.0 (unchanged/no-op): calibrated separately and it does
# not separate the two classes as well as top1_score alone (best accuracy
# ~68% vs. top1's ~84%) — adopting a non-zero cutoff here would only add a
# second tunable knob that trades away answerable questions for no
# real gain.
CONFIDENCE = ConfidenceConfig(min_top1_score=0.40, min_score_gap=0.0)

TOP_K = 5

# Categories where the numerical verification pass runs on the generated
# answer before it's returned (CLAUDE.md: reserved for high-stakes
# categories, not applied universally).
NUMERICAL_VERIFICATION_CATEGORIES: frozenset[str] = frozenset({"tuition", "deadlines"})
