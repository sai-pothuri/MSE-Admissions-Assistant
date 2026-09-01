from concurrent.futures import ThreadPoolExecutor

import anthropic

from app.clients.anthropic_client import get_anthropic_client
from app.config.thresholds import CONFIDENCE, NUMERICAL_VERIFICATION_CATEGORIES
from app.models.schemas import QueryResponse
from app.services.generation.generator import generate_answer
from app.services.guardrails import confidence_gate, fallback, numerical_verification
from app.services.guardrails.pre_classifier import PreClassifyResult
from app.services.guardrails.pre_classifier import check as pre_classify
from app.services.guardrails.query_classifier import classify_query
from app.services.retrieval.vector_search import search, top1_and_gap


def _classify(question: str) -> tuple[PreClassifyResult, str | None]:
    """Runs pre-classification and query-category classification
    concurrently, since neither depends on the other's output — this saves
    one round trip of latency on the common (non-off-limits) path. The
    trade-off: an off-limits question also pays for the now-unnecessary
    category call, which is acceptable since off-limits questions are the
    rare case.

    The two calls are handled asymmetrically on failure: pre-classification
    is safety-critical (deciding whether to proceed at all), so its failure
    propagates to the caller as an error rather than failing open. Category
    classification is not safety-critical — it degrades to unfiltered
    search (category=None) rather than failing the whole request."""
    with ThreadPoolExecutor(max_workers=2) as executor:
        pre_future = executor.submit(pre_classify, question)
        category_future = executor.submit(classify_query, question)

        pre_result = pre_future.result()

        try:
            category: str | None = category_future.result().category
        except anthropic.APIError:
            category = None

    return pre_result, category


def answer_query(question: str) -> QueryResponse:
    """The full per-query guardrail pipeline: pre-classify -> classify
    category -> filtered search (falling back to unfiltered if the filter
    yields nothing) -> confidence gate -> generate -> (for high-stakes
    categories) numerical verification. Each stage can short-circuit to a
    fallback response without reaching generation.

    May raise `CollectionNotReadyError` (from `search`) — the caller (the
    /query route) is responsible for translating that into an HTTP error."""
    try:
        pre_result, category = _classify(question)
    except anthropic.APIError:
        return fallback.service_unavailable_response()

    if not pre_result.allowed:
        assert pre_result.redirect_message is not None
        return fallback.off_limits_response(pre_result.redirect_message)

    results = search(question, category=category)
    if not results and category is not None:
        # The classified category may be a poor fit (e.g. "other", or a
        # genuine misclassification) — retry unfiltered rather than
        # declining outright just because the filtered search came up empty.
        results = search(question, category=None)

    if not results:
        return fallback.low_confidence_response()

    top1_score, score_gap = top1_and_gap(results)
    gate_result = confidence_gate.evaluate(top1_score, score_gap, CONFIDENCE)
    if not gate_result.passed:
        return fallback.low_confidence_response()

    answer, citations = generate_answer(get_anthropic_client(), question, results)

    if category in NUMERICAL_VERIFICATION_CATEGORIES:
        # Verify against everything the model was actually given as
        # context, not just the subset a citation-display heuristic
        # detects as "referenced" — a correct answer that paraphrases its
        # source instead of naming the file verbatim shouldn't be rejected.
        cited_texts = [r.chunk.text for r in results]
        verification = numerical_verification.verify(answer, cited_texts)
        if not verification.passed:
            return fallback.verification_failed_response()

    return QueryResponse(answer=answer, citations=citations)
