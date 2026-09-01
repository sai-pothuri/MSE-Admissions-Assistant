from functools import lru_cache

import anthropic

from app.config.settings import get_settings

CLASSIFY_MAX_TOKENS = 32


@lru_cache
def get_anthropic_client() -> anthropic.Anthropic:
    settings = get_settings()
    return anthropic.Anthropic(api_key=settings.anthropic_api_key)


def classify(client: anthropic.Anthropic, model: str, system_prompt: str, user_text: str) -> str:
    """Shared low-level helper for the guardrail/ingestion classification
    calls (pre-classification, query classification, auto-tagging) — a
    single-label Claude call with a small token budget and thinking
    disabled, since these are short, deterministic label lookups."""
    message = client.messages.create(
        model=model,
        max_tokens=CLASSIFY_MAX_TOKENS,
        system=system_prompt,
        thinking={"type": "disabled"},
        messages=[{"role": "user", "content": user_text}],
    )
    return "".join(block.text for block in message.content if block.type == "text").strip()
