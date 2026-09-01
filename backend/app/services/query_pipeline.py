from app.clients.anthropic_client import get_anthropic_client
from app.config.thresholds import CONFIDENCE, NUMERICAL_VERIFICATION_CATEGORIES
from app.models.schemas import QueryResponse
from app.services.generation.generator import generate_answer, referenced_results
from app.services.guardrails import confidence_gate, fallback, numerical_verification
from app.services.guardrails.pre_classifier import check as pre_classify
from app.services.guardrails.query_classifier import classify_query
from app.services.retrieval.vector_search import search, top1_and_gap


def answer_query(question: str) -> QueryResponse:
    """The full per-query guardrail pipeline: pre-classify -> classify
    category -> filtered search -> confidence gate -> generate -> (for
    high-stakes categories) numerical verification. Each stage can
    short-circuit to a fallback response without reaching generation.

    May raise `CollectionNotReadyError` (from `search`) — the caller (the
    /query route) is responsible for translating that into an HTTP error."""
    pre_result = pre_classify(question)
    if not pre_result.allowed:
        assert pre_result.redirect_message is not None
        return fallback.off_limits_response(pre_result.redirect_message)

    category = classify_query(question).category
    results = search(question, category=category)

    if not results:
        return fallback.low_confidence_response()

    top1_score, score_gap = top1_and_gap(results)
    gate_result = confidence_gate.evaluate(top1_score, score_gap, CONFIDENCE)
    if not gate_result.passed:
        return fallback.low_confidence_response()

    answer, citations = generate_answer(get_anthropic_client(), question, results)

    if category in NUMERICAL_VERIFICATION_CATEGORIES:
        cited_texts = [r.chunk.text for r in referenced_results(answer, results)]
        verification = numerical_verification.verify(answer, cited_texts)
        if not verification.passed:
            return fallback.verification_failed_response()

    return QueryResponse(answer=answer, citations=citations)
