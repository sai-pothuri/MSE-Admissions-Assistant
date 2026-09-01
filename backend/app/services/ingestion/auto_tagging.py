from collections.abc import Callable

from app.clients.anthropic_client import classify, get_anthropic_client
from app.config.settings import get_settings
from app.config.taxonomy import CATEGORIES

ClassifyFn = Callable[[str], str]

SYSTEM_PROMPT = (
    "Classify the following excerpt from a CMU Master of Software Engineering (MSE) "
    "program document into exactly one of these categories: "
    f"{', '.join(CATEGORIES)}.\n\n"
    "Respond with only the category name, nothing else."
)


def _default_classify_fn(text: str) -> str:
    settings = get_settings()
    return classify(
        get_anthropic_client(), settings.anthropic_classification_model, SYSTEM_PROMPT, text
    )


def classify_chunk(text: str, classify_fn: ClassifyFn | None = None) -> str:
    """Classifies a single chunk's text into the category taxonomy — one
    Claude call per chunk, per CLAUDE.md. Falls back to 'other' if the
    model returns anything outside the known categories."""
    resolved_classify_fn = classify_fn or _default_classify_fn
    raw = resolved_classify_fn(text).strip().lower()
    return raw if raw in CATEGORIES else "other"
