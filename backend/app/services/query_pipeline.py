from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

import anthropic

from app.clients.anthropic_client import get_anthropic_client
from app.config.thresholds import CONFIDENCE, NUMERICAL_VERIFICATION_CATEGORIES
from app.models.schemas import QueryResponse
from app.services.generation.generator import generate_answer
from app.services.guardrails import confidence_gate, fallback, numerical_verification
from app.services.guardrails.pre_classifier import PreClassifyResult
from app.services.guardrails.pre_classifier import check as pre_classify
from app.services.guardrails.query_classifier import classify_query
from app.services.retrieval.vector_search import SearchResult, embed, search, top1_and_gap

Stage = Literal[
    "service_unavailable",
    "pre_classify_rejected",
    "no_results",
    "confidence_gate_rejected",
    "verification_failed",
    "answered",
]


@dataclass(frozen=True)
class PipelineTrace:
    """Diagnostics for a single `answer_query` run — not part of the public
    `/query` response, but needed by the eval harness (Phase 3) to calibrate
    the confidence gate and report per-stage outcomes without duplicating
    the pipeline logic."""

    stage: Stage
    category: str | None
    matched_topic: str | None
    top1_score: float | None
    score_gap: float | None
    results: list[SearchResult]


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
    response, _trace = answer_query_traced(question)
    return response


def answer_query_traced(question: str) -> tuple[QueryResponse, PipelineTrace]:
    """Same pipeline as `answer_query`, but also returns a `PipelineTrace`
    with the intermediate scores/category/short-circuit stage — used by the
    eval harness (Phase 3) to calibrate the confidence gate and report
    outcomes per pipeline stage. `answer_query` is a thin wrapper around
    this that discards the trace, so the two never drift apart."""
    try:
        pre_result, category = _classify(question)
    except anthropic.APIError:
        trace = PipelineTrace(
            stage="service_unavailable",
            category=None,
            matched_topic=None,
            top1_score=None,
            score_gap=None,
            results=[],
        )
        return fallback.service_unavailable_response(), trace

    if not pre_result.allowed:
        assert pre_result.redirect_message is not None
        trace = PipelineTrace(
            stage="pre_classify_rejected",
            category=category,
            matched_topic=pre_result.matched_topic,
            top1_score=None,
            score_gap=None,
            results=[],
        )
        return fallback.off_limits_response(pre_result.redirect_message), trace

    query_vector = embed(question)
    results = search(question, category=category, query_vector=query_vector)
    if category is not None:
        # The classified category may be a poor fit (e.g. "other", or a
        # genuine misclassification) — an empty filtered search is the
        # obvious case, but a *non-empty* filtered search can also be
        # confidently wrong (real chunks under the wrong category filter,
        # scoring fine on their own terms while missing the actually
        # relevant content). So always check unfiltered too and keep
        # whichever has the higher top1 score, rather than only falling
        # back when the filter returns nothing. Reusing `query_vector`
        # means this costs one extra (cheap, local) Qdrant lookup, not a
        # second Voyage embedding call.
        filtered_top1 = top1_and_gap(results)[0] if results else 0.0
        unfiltered_results = search(question, category=None, query_vector=query_vector)
        if unfiltered_results and top1_and_gap(unfiltered_results)[0] > filtered_top1:
            results = unfiltered_results

    if not results:
        trace = PipelineTrace(
            stage="no_results",
            category=category,
            matched_topic=None,
            top1_score=None,
            score_gap=None,
            results=[],
        )
        return fallback.low_confidence_response(), trace

    top1_score, score_gap = top1_and_gap(results)
    gate_result = confidence_gate.evaluate(top1_score, score_gap, CONFIDENCE)
    if not gate_result.passed:
        trace = PipelineTrace(
            stage="confidence_gate_rejected",
            category=category,
            matched_topic=None,
            top1_score=top1_score,
            score_gap=score_gap,
            results=results,
        )
        return fallback.low_confidence_response(), trace

    answer, citations = generate_answer(get_anthropic_client(), question, results)

    if category in NUMERICAL_VERIFICATION_CATEGORIES:
        # Verify against everything the model was actually given as
        # context, not just the subset a citation-display heuristic
        # detects as "referenced" — a correct answer that paraphrases its
        # source instead of naming the file verbatim shouldn't be rejected.
        cited_texts = [r.chunk.text for r in results]
        verification = numerical_verification.verify(answer, cited_texts)
        if not verification.passed:
            trace = PipelineTrace(
                stage="verification_failed",
                category=category,
                matched_topic=None,
                top1_score=top1_score,
                score_gap=score_gap,
                results=results,
            )
            return fallback.verification_failed_response(), trace

    trace = PipelineTrace(
        stage="answered",
        category=category,
        matched_topic=None,
        top1_score=top1_score,
        score_gap=score_gap,
        results=results,
    )
    return QueryResponse(answer=answer, citations=citations), trace
