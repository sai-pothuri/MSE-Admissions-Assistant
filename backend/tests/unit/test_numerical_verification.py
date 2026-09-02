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


def test_verify_rejects_a_truncated_number_embedded_in_a_longer_correct_one():
    """Regression test: plain substring containment would let "6,250" pass
    just because it's embedded in the correct "$26,250.00" — the truncated/
    wrong figure must be rejected, not silently verified."""
    answer = "Tuition is $6,250 per semester."
    cited_texts = ["The MSE tuition rate is $26,250.00 per semester."]
    result = verify(answer, cited_texts)
    assert result.passed is False
    assert "$6,250" in result.unverified_numbers


def test_verify_rejects_truncated_percentage_embedded_in_a_longer_one():
    answer = "There is a 5% increase this year."
    cited_texts = ["Tuition increased 25% compared to last year."]
    result = verify(answer, cited_texts)
    assert result.passed is False
    assert "5%" in result.unverified_numbers


def test_extract_numbers_does_not_swallow_a_trailing_comma():
    """Regression test: '$7,500, not $10,000' must extract '$7,500' and
    '$10,000' separately, not '$7,500,' with the punctuation comma baked
    into the number — the corrupted token would never match the source
    text's clean '$7,500', false-positiving a correct answer."""
    numbers = extract_numbers("It's worth $7,500, not $10,000.")
    assert "$7,500" in numbers
    assert "$10,000" in numbers
    assert "$7,500," not in numbers


def test_verify_passes_when_number_is_followed_by_a_comma_in_the_answer():
    answer = "The scholarship is worth $7,500, not the figure you mentioned."
    cited_texts = ["Director's Scholarships are awarded for $7,500 each."]
    result = verify(answer, cited_texts)
    assert result.passed is True


def test_extract_numbers_ignores_page_citation_numbers():
    """Regression test: a page number cited as '(Source: file, page 12)' is
    citation metadata, not a factual claim — it won't appear in the cited
    chunk's body text and must not be treated as an unverified fact."""
    numbers = extract_numbers("The rate is $7,500 (Source: FAQ.pdf, page 12).")
    assert "$7,500" in numbers
    assert "12" not in numbers


def test_extract_numbers_ignores_multiple_page_citation_numbers():
    numbers = extract_numbers("Contacts are listed (SCS_S3D_MSE.pdf, pages 9, 49, 50).")
    assert numbers == []


def test_verify_passes_when_only_number_present_is_a_page_citation():
    answer = "Marlana Ivey handles admissions (SCS_S3D_MSE.pdf, page 9)."
    cited_texts = ["Marlana Ivey is the Senior Admissions Officer."]
    result = verify(answer, cited_texts)
    assert result.passed is True
