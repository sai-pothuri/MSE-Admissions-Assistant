from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from app.clients.anthropic_client import classify, get_anthropic_client
from app.config.settings import get_settings

OFFLIMITS_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "offlimits.yaml"

ClassifyFn = Callable[[str], str]


@dataclass(frozen=True)
class OffLimitsTopic:
    name: str
    description: str
    example_phrasings: list[str]


@dataclass(frozen=True)
class OffLimitsConfig:
    redirect_message: str
    topics: list[OffLimitsTopic]


@dataclass(frozen=True)
class PreClassifyResult:
    allowed: bool
    matched_topic: str | None = None
    redirect_message: str | None = None


@lru_cache
def load_offlimits_config(path: Path = OFFLIMITS_CONFIG_PATH) -> OffLimitsConfig:
    with path.open() as f:
        raw = yaml.safe_load(f)
    topics = [
        OffLimitsTopic(
            name=t["name"],
            description=t["description"],
            example_phrasings=t["example_phrasings"],
        )
        for t in raw["topics"]
    ]
    return OffLimitsConfig(redirect_message=raw["redirect_message"], topics=topics)


def _build_system_prompt(config: OffLimitsConfig) -> str:
    topic_lines = "\n".join(
        f'- {t.name}: {t.description} (e.g. "{t.example_phrasings[0]}")' for t in config.topics
    )
    return (
        "You are a topic-boundary classifier for a CMU Master of Software Engineering "
        "(MSE) admissions chatbot. Given a prospective student's question, decide whether "
        "it falls into one of these off-limits topics:\n\n"
        f"{topic_lines}\n\n"
        "Respond with only the matching topic name exactly as given above, "
        "or the single word 'none' if it doesn't match any of them."
    )


def _default_classify_fn(prompt_text: str, system_prompt: str) -> str:
    settings = get_settings()
    return classify(
        get_anthropic_client(), settings.anthropic_classification_model, system_prompt, prompt_text
    )


def check(
    query: str,
    config: OffLimitsConfig | None = None,
    classify_fn: ClassifyFn | None = None,
) -> PreClassifyResult:
    """Runs BEFORE retrieval. If the query matches an off-limits topic,
    returns allowed=False with a redirect message — the caller should
    short-circuit and never reach retrieval or generation."""
    resolved_config = config if config is not None else load_offlimits_config()
    system_prompt = _build_system_prompt(resolved_config)
    resolved_classify_fn = classify_fn or (
        lambda text: _default_classify_fn(text, system_prompt)
    )

    raw = resolved_classify_fn(query).strip().lower()
    matched = next((t for t in resolved_config.topics if t.name.lower() == raw), None)

    if matched is None:
        return PreClassifyResult(allowed=True)
    return PreClassifyResult(
        allowed=False,
        matched_topic=matched.name,
        redirect_message=resolved_config.redirect_message,
    )
