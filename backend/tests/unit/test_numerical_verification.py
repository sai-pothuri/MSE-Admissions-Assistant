from app.services.guardrails.numerical_verification import extract_numbers, verify


def test_extract_numbers_finds_currency_percentages_and_plain_numbers():
    text = "Tuition is $26,250.00 per semester, a 5% increase, across 2 semesters."
    numbers = extract_numbers(text)
    assert "$26,250.00" in numbers
    assert "5%" in numbers
    assert "2" in numbers


def test_verify_passes_when_all_numbers_appear_in_cited_text():
    answer = "Tuition is $26,250.00 per semester."
    cited_texts = ["The MSE tuition rate is $26,250.00 per semester for 2025-2026."]
    result = verify(answer, cited_texts)
    assert result.passed is True
    assert result.unverified_numbers == []


def test_verify_fails_when_a_number_is_not_in_cited_text():
    answer = "Tuition is $99,999.00 per semester."
    cited_texts = ["The MSE tuition rate is $26,250.00 per semester."]
    result = verify(answer, cited_texts)
    assert result.passed is False
    assert "$99,999.00" in result.unverified_numbers


def test_verify_passes_when_answer_has_no_numbers():
    answer = "Tuition information is available on the program website."
    result = verify(answer, cited_texts=["Some unrelated source text."])
    assert result.passed is True


def test_verify_checks_across_multiple_cited_chunks():
    answer = "The deadline is January 15 and the fee is $50."
    cited_texts = ["Applications close on January 15.", "A $50 application fee applies."]
    result = verify(answer, cited_texts)
    assert result.passed is True
