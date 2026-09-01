from dataclasses import dataclass

from app.config.taxonomy import CATEGORIES
from app.services.classification import ClassifyFn, match_label, resolve_classify_fn

SYSTEM_PROMPT = (
    "Classify the following prospective-student question about the CMU Master of "
    "Software Engineering (MSE) program into exactly one of these categories: "
    f"{', '.join(CATEGORIES)}.\n\n"
    "Respond with only the category name, nothing else."
)


@dataclass(frozen=True)
class QueryClassifyResult:
    category: str  # always one of CATEGORIES — falls back to "other"


def classify_query(question: str, classify_fn: ClassifyFn | None = None) -> QueryClassifyResult:
    """Classifies the query into the same category taxonomy used for
    ingestion tagging, so it can be used as a Qdrant payload filter."""
    resolved_classify_fn = resolve_classify_fn(classify_fn, SYSTEM_PROMPT)
    raw = resolved_classify_fn(question)
    category = match_label(raw, CATEGORIES) or "other"
    return QueryClassifyResult(category=category)
