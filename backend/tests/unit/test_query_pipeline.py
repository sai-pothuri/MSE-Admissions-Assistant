from unittest.mock import MagicMock

from app.models.schemas import ChunkPayload, Citation, QueryResponse
from app.services import query_pipeline
from app.services.guardrails.pre_classifier import PreClassifyResult
from app.services.guardrails.query_classifier import QueryClassifyResult
from app.services.retrieval.vector_search import SearchResult


def _result(source_file="faq.pdf", page=1, text="Tuition is $26,250.", category="tuition"):
    chunk = ChunkPayload(text=text, source_file=source_file, page_number=page, category=category)
    return SearchResult(chunk=chunk, score=0.9)


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
    classify_query_mock = MagicMock()
    search_mock = MagicMock()
    monkeypatch.setattr(query_pipeline, "classify_query", classify_query_mock)
    monkeypatch.setattr(query_pipeline, "search", search_mock)

    response = query_pipeline.answer_query("What are my chances of getting in?")

    assert response == QueryResponse(answer="Contact the office.", citations=[])
    classify_query_mock.assert_not_called()
    search_mock.assert_not_called()


def test_answer_query_returns_low_confidence_fallback_when_no_results(monkeypatch):
    monkeypatch.setattr(query_pipeline, "pre_classify", lambda q: PreClassifyResult(allowed=True))
    monkeypatch.setattr(
        query_pipeline, "classify_query", lambda q: QueryClassifyResult(category="tuition")
    )
    monkeypatch.setattr(query_pipeline, "search", lambda q, category=None: [])
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
    monkeypatch.setattr(query_pipeline, "search", lambda q, category=None: [_result()])
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
        query_pipeline, "search", lambda q, category=None: [_result(category="admissions")]
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
    monkeypatch.setattr(query_pipeline, "search", lambda q, category=None: [_result()])
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
    monkeypatch.setattr(query_pipeline, "search", lambda q, category=None: [_result()])
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
