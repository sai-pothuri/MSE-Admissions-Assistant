from app.services.classification import match_label, normalize_label

CANDIDATES = ["admissions", "curriculum", "tuition", "deadlines", "faculty", "other"]


def test_match_label_exact_match():
    assert match_label("tuition", CANDIDATES) == "tuition"


def test_match_label_tolerates_case_and_whitespace():
    assert match_label("  Tuition \n", CANDIDATES) == "tuition"


def test_match_label_tolerates_preamble_and_punctuation():
    """Regression test: a formatting quirk in the model's response (extra
    words, trailing punctuation) shouldn't cause the label to go
    unrecognized — that's what let a genuinely off-limits question fail
    open in the review finding this fixes."""
    assert match_label("Topic: admissions.", CANDIDATES) == "admissions"
    assert match_label("The category is deadlines", CANDIDATES) == "deadlines"


def test_match_label_does_not_false_match_short_candidate_inside_longer_word():
    """"other" must not match merely because it's a substring of an
    unrelated word like "another" — word-boundary matching is required."""
    assert match_label("this is another topic entirely", CANDIDATES) is None


def test_match_label_returns_none_for_no_match():
    assert match_label("completely unrelated response", CANDIDATES) is None


def test_normalize_label_strips_whitespace_case_and_trailing_punctuation():
    assert normalize_label("  Tuition. \n") == "tuition"
    assert normalize_label("Deadlines:") == "deadlines"
