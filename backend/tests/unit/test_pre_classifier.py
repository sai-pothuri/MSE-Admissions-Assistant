from app.services.guardrails.pre_classifier import (
    OffLimitsConfig,
    OffLimitsTopic,
    check,
    load_offlimits_config,
)

CONFIG = OffLimitsConfig(
    redirect_message="Please contact the program office.",
    topics=[
        OffLimitsTopic(
            name="admissions_predictions",
            description="Predicting an individual's odds of admission.",
            example_phrasings=["What are my chances of getting in?"],
        ),
        OffLimitsTopic(
            name="legal_or_visa_advice",
            description="Individualized legal or visa advice.",
            example_phrasings=["What visa should I apply for?"],
        ),
    ],
)


def test_check_allows_in_scope_question():
    result = check("What are the admission requirements?", CONFIG, classify_fn=lambda q: "none")
    assert result.allowed is True
    assert result.matched_topic is None
    assert result.redirect_message is None


def test_check_blocks_matched_off_limits_topic():
    result = check(
        "What are my chances of getting in with a 3.2 GPA?",
        CONFIG,
        classify_fn=lambda q: "admissions_predictions",
    )
    assert result.allowed is False
    assert result.matched_topic == "admissions_predictions"
    assert result.redirect_message == "Please contact the program office."


def test_check_is_case_insensitive_and_strips_whitespace():
    result = check("...", CONFIG, classify_fn=lambda q: "  Legal_Or_Visa_Advice \n")
    assert result.allowed is False
    assert result.matched_topic == "legal_or_visa_advice"


def test_check_treats_unrecognized_label_as_allowed():
    """A model hallucinating a label outside the configured topic list
    should fail open to 'allowed' rather than block on an unknown reason."""
    result = check("A normal question.", CONFIG, classify_fn=lambda q: "something_unexpected")
    assert result.allowed is True


def test_load_offlimits_config_reads_the_real_yaml_file():
    config = load_offlimits_config()
    assert config.redirect_message
    names = [t.name for t in config.topics]
    assert "admissions_predictions" in names
    assert all(t.example_phrasings for t in config.topics)
