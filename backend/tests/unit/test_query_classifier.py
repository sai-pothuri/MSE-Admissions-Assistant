from app.services.guardrails.query_classifier import classify_query


def test_classify_query_returns_matching_category():
    result = classify_query("What is the tuition cost?", classify_fn=lambda q: "tuition")
    assert result.category == "tuition"


def test_classify_query_normalizes_case_and_whitespace():
    result = classify_query("...", classify_fn=lambda q: "  Curriculum \n")
    assert result.category == "curriculum"


def test_classify_query_falls_back_to_other_for_unrecognized_label():
    result = classify_query("...", classify_fn=lambda q: "not_a_real_category")
    assert result.category == "other"
