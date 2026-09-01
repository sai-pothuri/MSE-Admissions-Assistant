import re
from collections.abc import Callable

from app.clients.anthropic_client import classify, get_anthropic_client
from app.config.settings import get_settings

ClassifyFn = Callable[[str], str]


def default_classify_fn(system_prompt: str) -> ClassifyFn:
    """Returns a classify_fn bound to a specific system prompt, backed by the
    real Anthropic classification model. Shared by pre_classifier.py,
    query_classifier.py, and auto_tagging.py so a change to the underlying
    call (retry logic, token budget, etc.) only needs to happen once."""

    def _call(user_text: str) -> str:
        settings = get_settings()
        return classify(
            get_anthropic_client(),
            settings.anthropic_classification_model,
            system_prompt,
            user_text,
        )

    return _call


def resolve_classify_fn(classify_fn: ClassifyFn | None, system_prompt: str) -> ClassifyFn:
    return classify_fn if classify_fn is not None else default_classify_fn(system_prompt)


def normalize_label(raw: str) -> str:
    return raw.strip().strip(".:,;").lower()


def match_label(raw: str, candidates: list[str]) -> str | None:
    """Robust label matching against a fixed set of candidate labels: exact
    match first, falling back to a candidate appearing as a whole word
    within the (normalized) response. Tolerates preambles/punctuation the
    model might add despite being asked for a bare label (e.g. "Topic:
    admissions_predictions." or a truncated/padded response), while the
    word-boundary check prevents short candidates from false-matching
    inside unrelated words (e.g. "other" inside "another")."""
    normalized = normalize_label(raw)
    if normalized in candidates:
        return normalized
    for candidate in candidates:
        if re.search(rf"\b{re.escape(candidate)}\b", normalized):
            return candidate
    return None
