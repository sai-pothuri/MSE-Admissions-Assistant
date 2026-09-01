from app.config.taxonomy import CATEGORIES
from app.services.classification import ClassifyFn, match_label, resolve_classify_fn

SYSTEM_PROMPT = (
    "Classify the following excerpt from a CMU Master of Software Engineering (MSE) "
    "program document into exactly one of these categories: "
    f"{', '.join(CATEGORIES)}.\n\n"
    "Respond with only the category name, nothing else."
)


def classify_chunk(text: str, classify_fn: ClassifyFn | None = None) -> str:
    """Classifies a single chunk's text into the category taxonomy — one
    Claude call per chunk, per CLAUDE.md. Falls back to 'other' if the
    model returns anything outside the known categories."""
    resolved_classify_fn = resolve_classify_fn(classify_fn, SYSTEM_PROMPT)
    raw = resolved_classify_fn(text)
    return match_label(raw, CATEGORIES) or "other"
