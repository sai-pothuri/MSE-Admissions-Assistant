from app.services.ingestion.auto_tagging import classify_chunk


def test_classify_chunk_returns_matching_category():
    text = "Tuition for the MSE program is $26,250 per semester."
    category = classify_chunk(text, lambda t: "tuition")
    assert category == "tuition"


def test_classify_chunk_normalizes_case_and_whitespace():
    category = classify_chunk("...", lambda t: "  Faculty \n")
    assert category == "faculty"


def test_classify_chunk_falls_back_to_other_for_unrecognized_label():
    category = classify_chunk("...", lambda t: "not_a_real_category")
    assert category == "other"
