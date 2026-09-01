from collections.abc import Callable
from dataclasses import dataclass

from app.clients.anthropic_client import classify, get_anthropic_client
from app.config.settings import get_settings
from app.config.taxonomy import CATEGORIES

ClassifyFn = Callable[[str], str]

SYSTEM_PROMPT = (
    "Classify the following prospective-student question about the CMU Master of "
    "Software Engineering (MSE) program into exactly one of these categories: "
    f"{', '.join(CATEGORIES)}.\n\n"
    "Respond with only the category name, nothing else."
)


@dataclass(frozen=True)
class QueryClassifyResult:
    category: str  # always one of CATEGORIES — falls back to "other"


def _default_classify_fn(question: str) -> str:
    settings = get_settings()
    return classify(
        get_anthropic_client(), settings.anthropic_classification_model, SYSTEM_PROMPT, question
    )


def classify_query(question: str, classify_fn: ClassifyFn | None = None) -> QueryClassifyResult:
    """Classifies the query into the same category taxonomy used for
    ingestion tagging, so it can be used as a Qdrant payload filter."""
    resolved_classify_fn = classify_fn or _default_classify_fn
    raw = resolved_classify_fn(question).strip().lower()
    category = raw if raw in CATEGORIES else "other"
    return QueryClassifyResult(category=category)
