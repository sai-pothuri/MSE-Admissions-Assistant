import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import harness  # noqa: E402

from app.models.schemas import QueryResponse  # noqa: E402
from app.services.query_pipeline import PipelineTrace  # noqa: E402


def _question(id_="q1", expected_behavior="decline", notes=None):
    return {
        "id": id_,
        "question": "A question.",
        "category_expected": None,
        "type": "out_of_scope",
        "expected_behavior": expected_behavior,
        **({"notes": notes} if notes else {}),
    }


def _trace(stage, category=None):
    return PipelineTrace(
        stage=stage,
        category=category,
        matched_topic=None,
        top1_score=None,
        score_gap=None,
        results=[],
    )


def test_decline_question_passes_when_guardrail_short_circuits(monkeypatch):
    monkeypatch.setattr(
        harness,
        "answer_query_traced",
        lambda q: (
            QueryResponse(answer="Declined.", citations=[]),
            _trace("confidence_gate_rejected"),
        ),
    )

    record = harness.run_question("reference text", _question(expected_behavior="decline"))

    assert record["passed"] is True
    assert record["judged_correct"] is None


def test_decline_question_fails_on_service_unavailable_instead_of_auto_passing(monkeypatch):
    """Regression test: a real Anthropic API failure during
    pre-classification (stage='service_unavailable') is not a guardrail
    decision — it must not be scored as a correct decline, or a genuine
    outage would inflate the decline-side pass rate while answer-type
    questions in the same window are correctly scored as failures."""
    monkeypatch.setattr(
        harness,
        "answer_query_traced",
        lambda q: (
            QueryResponse(answer="temporarily unable...", citations=[]),
            _trace("service_unavailable"),
        ),
    )

    record = harness.run_question("reference text", _question(expected_behavior="decline"))

    assert record["passed"] is False
    assert "service_unavailable" in record["error"]


def test_decline_question_judged_when_generation_still_ran(monkeypatch):
    monkeypatch.setattr(
        harness,
        "answer_query_traced",
        lambda q: (QueryResponse(answer="I don't know.", citations=[]), _trace("answered")),
    )
    monkeypatch.setattr(
        harness, "judge_decline", lambda *a, **k: {"correct": True, "reasoning": "safe"}
    )

    record = harness.run_question("reference text", _question(expected_behavior="decline"))

    assert record["passed"] is True
    assert record["judged_correct"] is True


def test_answer_question_fails_when_pipeline_did_not_answer(monkeypatch):
    monkeypatch.setattr(
        harness,
        "answer_query_traced",
        lambda q: (
            QueryResponse(answer="Declined.", citations=[]),
            _trace("confidence_gate_rejected"),
        ),
    )
    judge_mock = MagicMock()
    monkeypatch.setattr(harness, "judge_answer", judge_mock)

    record = harness.run_question("reference text", _question(expected_behavior="answer"))

    assert record["passed"] is False
    judge_mock.assert_not_called()


def test_run_question_never_raises_on_pipeline_error(monkeypatch):
    def _raise(q):
        raise RuntimeError("boom")

    monkeypatch.setattr(harness, "answer_query_traced", _raise)

    record = harness.run_question("reference text", _question(expected_behavior="answer"))

    assert record["passed"] is False
    assert "boom" in record["error"]
