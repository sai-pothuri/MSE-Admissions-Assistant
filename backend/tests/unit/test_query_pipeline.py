from unittest.mock import MagicMock

import anthropic
import httpx2
import pytest

from app.models.schemas import ChunkPayload, Citation, QueryResponse
from app.services import query_pipeline
from app.services.guardrails.pre_classifier import PreClassifyResult
from app.services.guardrails.query_classifier import QueryClassifyResult
from app.services.retrieval.vector_search import SearchResult


@pytest.fixture(autouse=True)
def _stub_embed(monkeypatch):
    # None of these tests exercise embedding itself — `search` is what's
    # under test/mocked per-case — so stub it out everywhere to avoid a
    # real Voyage API call from the unconditional `embed(question)` in
    # `answer_query_traced`.
    monkeypatch.setattr(query_pipeline, "embed", lambda question: [0.1, 0.2])


def _result(source_file="faq.pdf", page=1, text="Tuition is $26,250.", category="tuition"):
    chunk = ChunkPayload(text=text, source_file=source_file, page_number=page, category=category)
    return SearchResult(chunk=chunk, score=0.9)


def _api_error():
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIError("boom", request, body=None)


def _raising(exc):
    def _fn(*_args, **_kwargs):
        raise exc

    return _fn


def _set_gate(monkeypatch, passed):
    monkeypatch.setattr(
        query_pipeline.confidence_gate, "evaluate", lambda *a, **k: MagicMock(passed=passed)
    )


def _set_generate_answer(monkeypatch, answer, source_file="faq.pdf", page=1):
    monkeypatch.setattr(
        query_pipeline,
        "generate_answer",
        lambda client, q, results: (answer, [Citation(source_file=source_file, page_number=page)]),
    )


def test_answer_query_short_circuits_on_off_limits(monkeypatch):
    monkeypatch.setattr(
        query_pipeline,
        "pre_classify",
        lambda q: PreClassifyResult(allowed=False, redirect_message="Contact the office."),
    )
    # classify_query still runs concurrently with pre_classify (there's no
    # data dependency between them), so it's not asserted not-called here —
    # only that we never reach retrieval once pre_classify says "blocked".
    monkeypatch.setattr(
        query_pipeline, "classify_query", lambda q: QueryClassifyResult(category="admissions")
    )
    search_mock = MagicMock()
    monkeypatch.setattr(query_pipeline, "search", search_mock)

    response = query_pipeline.answer_query("What are my chances of getting in?")

    assert response == QueryResponse(answer="Contact the office.", citations=[])
    search_mock.assert_not_called()


def test_answer_query_returns_service_unavailable_when_pre_classify_errors(monkeypatch):
    monkeypatch.setattr(query_pipeline, "pre_classify", _raising(_api_error()))
    monkeypatch.setattr(
        query_pipeline, "classify_query", lambda q: QueryClassifyResult(category="admissions")
    )
    search_mock = MagicMock()
    monkeypatch.setattr(query_pipeline, "search", search_mock)

    response = query_pipeline.answer_query("A question.")

    assert "temporarily unable" in response.answer
    search_mock.assert_not_called()


def test_answer_query_degrades_to_unfiltered_search_when_classify_query_errors(monkeypatch):
    monkeypatch.setattr(query_pipeline, "pre_classify", lambda q: PreClassifyResult(allowed=True))
    monkeypatch.setattr(query_pipeline, "classify_query", _raising(_api_error()))
    search_mock = MagicMock(return_value=[_result()])
    monkeypatch.setattr(query_pipeline, "search", search_mock)
    _set_gate(monkeypatch, passed=True)
    _set_generate_answer(monkeypatch, "An answer.")

    response = query_pipeline.answer_query("A question.")

    assert response.answer == "An answer."
    search_mock.assert_called_once_with("A question.", category=None, query_vector=[0.1, 0.2])


def test_answer_query_falls_back_to_unfiltered_search_when_filtered_search_is_empty(monkeypatch):
    monkeypatch.setattr(query_pipeline, "pre_classify", lambda q: PreClassifyResult(allowed=True))
    monkeypatch.setattr(
        query_pipeline, "classify_query", lambda q: QueryClassifyResult(category="other")
    )
    calls = []

    def fake_search(q, category=None, query_vector=None):
        calls.append(category)
        return [] if category == "other" else [_result(category="curriculum")]

    monkeypatch.setattr(query_pipeline, "search", fake_search)
    _set_gate(monkeypatch, passed=True)
    _set_generate_answer(monkeypatch, "An answer.")

    response = query_pipeline.answer_query("A question that misclassified as other.")

    assert response.answer == "An answer."
    assert calls == ["other", None]


def test_answer_query_prefers_unfiltered_search_when_it_scores_higher_than_filtered(monkeypatch):
    """Regression test: a category-filtered search that returns *some*
    results is not proof they're the *right* results — e.g. the question
    got misclassified into a real but wrong category. If unfiltered search
    scores higher, its results must win even though the filtered search
    wasn't empty (the bug: the old code only ever fell back on an empty
    filtered result)."""
    monkeypatch.setattr(query_pipeline, "pre_classify", lambda q: PreClassifyResult(allowed=True))
    monkeypatch.setattr(
        query_pipeline, "classify_query", lambda q: QueryClassifyResult(category="other")
    )
    wrong_category_result = _result(category="other", text="Irrelevant chunk.")
    wrong_category_result.score = 0.4
    right_category_result = _result(category="admissions", text="Class size is 25-30.")
    right_category_result.score = 0.6

    def fake_search(q, category=None, query_vector=None):
        return [wrong_category_result] if category == "other" else [right_category_result]

    monkeypatch.setattr(query_pipeline, "search", fake_search)
    _set_gate(monkeypatch, passed=True)
    _set_generate_answer(monkeypatch, "Class size is 25-30.")

    _, trace = query_pipeline.answer_query_traced("What is the class size?")

    assert trace.top1_score == 0.6
    assert trace.results == [right_category_result]


def test_answer_query_runs_numerical_verification_on_fallback_results_despite_stale_category(
    monkeypatch,
):
    """Regression test: a query misclassified into a non-high-stakes
    category (e.g. "faculty") whose unfiltered-fallback results actually
    land in "tuition" must still run numerical verification — gating on
    the classified `category` instead of the actual result categories
    would silently skip it."""
    monkeypatch.setattr(query_pipeline, "pre_classify", lambda q: PreClassifyResult(allowed=True))
    monkeypatch.setattr(
        query_pipeline, "classify_query", lambda q: QueryClassifyResult(category="faculty")
    )
    faculty_result = _result(category="faculty", text="Irrelevant faculty chunk.")
    faculty_result.score = 0.4
    tuition_result = _result(category="tuition", text="The MSE tuition rate is $26,250.00.")
    tuition_result.score = 0.6

    def fake_search(q, category=None, query_vector=None):
        return [faculty_result] if category == "faculty" else [tuition_result]

    monkeypatch.setattr(query_pipeline, "search", fake_search)
    _set_gate(monkeypatch, passed=True)
    _set_generate_answer(monkeypatch, "Tuition is $99,999.")
    monkeypatch.setattr(
        query_pipeline.numerical_verification,
        "verify",
        lambda answer, texts: MagicMock(passed=False),
    )

    response = query_pipeline.answer_query("What's the tuition?")

    assert "couldn't verify" in response.answer


def test_answer_query_returns_low_confidence_fallback_when_no_results(monkeypatch):
    monkeypatch.setattr(query_pipeline, "pre_classify", lambda q: PreClassifyResult(allowed=True))
    monkeypatch.setattr(
        query_pipeline, "classify_query", lambda q: QueryClassifyResult(category="tuition")
    )
    monkeypatch.setattr(query_pipeline, "search", lambda q, category=None, query_vector=None: [])
    generate_answer_mock = MagicMock()
    monkeypatch.setattr(query_pipeline, "generate_answer", generate_answer_mock)

    response = query_pipeline.answer_query("What is tuition?")

    assert response.citations == []
    assert "don't have reliable information" in response.answer
    generate_answer_mock.assert_not_called()


def test_answer_query_returns_low_confidence_fallback_when_gate_fails(monkeypatch):
    monkeypatch.setattr(query_pipeline, "pre_classify", lambda q: PreClassifyResult(allowed=True))
    monkeypatch.setattr(
        query_pipeline, "classify_query", lambda q: QueryClassifyResult(category="tuition")
    )
    monkeypatch.setattr(
        query_pipeline, "search", lambda q, category=None, query_vector=None: [_result()]
    )
    _set_gate(monkeypatch, passed=False)
    generate_answer_mock = MagicMock()
    monkeypatch.setattr(query_pipeline, "generate_answer", generate_answer_mock)

    response = query_pipeline.answer_query("What is tuition?")

    assert "don't have reliable information" in response.answer
    generate_answer_mock.assert_not_called()


def test_answer_query_skips_numerical_verification_for_non_high_stakes_category(monkeypatch):
    monkeypatch.setattr(query_pipeline, "pre_classify", lambda q: PreClassifyResult(allowed=True))
    monkeypatch.setattr(
        query_pipeline, "classify_query", lambda q: QueryClassifyResult(category="admissions")
    )
    monkeypatch.setattr(
        query_pipeline,
        "search",
        lambda q, category=None, query_vector=None: [_result(category="admissions")],
    )
    _set_gate(monkeypatch, passed=True)
    _set_generate_answer(monkeypatch, "Requirements are X.")
    verify_mock = MagicMock()
    monkeypatch.setattr(query_pipeline.numerical_verification, "verify", verify_mock)

    response = query_pipeline.answer_query("What are the requirements?")

    assert response.answer == "Requirements are X."
    verify_mock.assert_not_called()


def test_answer_query_runs_numerical_verification_for_tuition_and_passes(monkeypatch):
    monkeypatch.setattr(query_pipeline, "pre_classify", lambda q: PreClassifyResult(allowed=True))
    monkeypatch.setattr(
        query_pipeline, "classify_query", lambda q: QueryClassifyResult(category="tuition")
    )
    monkeypatch.setattr(
        query_pipeline, "search", lambda q, category=None, query_vector=None: [_result()]
    )
    _set_gate(monkeypatch, passed=True)
    _set_generate_answer(monkeypatch, "Tuition is $26,250.")
    monkeypatch.setattr(
        query_pipeline.numerical_verification,
        "verify",
        lambda answer, texts: MagicMock(passed=True),
    )

    response = query_pipeline.answer_query("What is tuition?")

    assert response.answer == "Tuition is $26,250."


def test_answer_query_returns_verification_failed_fallback(monkeypatch):
    monkeypatch.setattr(query_pipeline, "pre_classify", lambda q: PreClassifyResult(allowed=True))
    monkeypatch.setattr(
        query_pipeline, "classify_query", lambda q: QueryClassifyResult(category="tuition")
    )
    monkeypatch.setattr(
        query_pipeline, "search", lambda q, category=None, query_vector=None: [_result()]
    )
    _set_gate(monkeypatch, passed=True)
    _set_generate_answer(monkeypatch, "Tuition is $99,999.")
    monkeypatch.setattr(
        query_pipeline.numerical_verification,
        "verify",
        lambda answer, texts: MagicMock(passed=False),
    )

    response = query_pipeline.answer_query("What is tuition?")

    assert "couldn't verify" in response.answer
    assert response.citations == []


def test_answer_query_traced_reports_answered_stage_with_scores(monkeypatch):
    monkeypatch.setattr(query_pipeline, "pre_classify", lambda q: PreClassifyResult(allowed=True))
    monkeypatch.setattr(
        query_pipeline, "classify_query", lambda q: QueryClassifyResult(category="admissions")
    )
    monkeypatch.setattr(
        query_pipeline,
        "search",
        lambda q, category=None, query_vector=None: [_result(category="admissions")],
    )
    _set_gate(monkeypatch, passed=True)
    _set_generate_answer(monkeypatch, "Requirements are X.")

    response, trace = query_pipeline.answer_query_traced("What are the requirements?")

    assert response.answer == "Requirements are X."
    assert trace.stage == "answered"
    assert trace.category == "admissions"
    assert trace.top1_score == 0.9
    assert trace.results != []


def test_answer_query_traced_reports_confidence_gate_rejected_stage(monkeypatch):
    monkeypatch.setattr(query_pipeline, "pre_classify", lambda q: PreClassifyResult(allowed=True))
    monkeypatch.setattr(
        query_pipeline, "classify_query", lambda q: QueryClassifyResult(category="tuition")
    )
    monkeypatch.setattr(
        query_pipeline, "search", lambda q, category=None, query_vector=None: [_result()]
    )
    _set_gate(monkeypatch, passed=False)

    _, trace = query_pipeline.answer_query_traced("What is tuition?")

    assert trace.stage == "confidence_gate_rejected"
    # Scores must still be reported even though the gate rejected — the eval
    # harness needs them to calibrate the threshold regardless of what the
    # *current* threshold decided.
    assert trace.top1_score == 0.9


def test_answer_query_traced_reports_pre_classify_rejected_stage_with_matched_topic(monkeypatch):
    monkeypatch.setattr(
        query_pipeline,
        "pre_classify",
        lambda q: PreClassifyResult(
            allowed=False, matched_topic="admissions_predictions", redirect_message="Contact us."
        ),
    )
    monkeypatch.setattr(
        query_pipeline, "classify_query", lambda q: QueryClassifyResult(category="admissions")
    )

    _, trace = query_pipeline.answer_query_traced("What are my chances?")

    assert trace.stage == "pre_classify_rejected"
    assert trace.matched_topic == "admissions_predictions"
    assert trace.top1_score is None


def test_answer_query_verifies_against_full_context_not_just_referenced_citations(monkeypatch):
    """Regression test: numerical verification must use everything the
    model was given as context, not the citation-display heuristic — a
    paraphrased answer that doesn't literally name the source file should
    still verify correctly against the real cited chunk text."""
    monkeypatch.setattr(query_pipeline, "pre_classify", lambda q: PreClassifyResult(allowed=True))
    monkeypatch.setattr(
        query_pipeline, "classify_query", lambda q: QueryClassifyResult(category="tuition")
    )
    monkeypatch.setattr(
        query_pipeline,
        "search",
        lambda q, category=None, query_vector=None: [
            _result(text="The MSE tuition rate is $26,250.00 per semester.")
        ],
    )
    _set_gate(monkeypatch, passed=True)
    # Answer paraphrases instead of naming "faq.pdf" verbatim.
    monkeypatch.setattr(
        query_pipeline,
        "generate_answer",
        lambda client, q, results: (
            "According to the tuition schedule, it's $26,250.00 per semester.",
            [],
        ),
    )

    response = query_pipeline.answer_query("What is tuition?")

    assert "$26,250.00" in response.answer
    assert "couldn't verify" not in response.answer
