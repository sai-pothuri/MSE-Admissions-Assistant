import anthropic

from app.config.settings import get_settings
from app.models.schemas import Citation
from app.services.generation.prompt_templates import SYSTEM_PROMPT, build_user_message
from app.services.retrieval.vector_search import SearchResult

MAX_TOKENS = 1024


def generate_answer(
    client: anthropic.Anthropic, question: str, results: list[SearchResult]
) -> tuple[str, list[Citation]]:
    settings = get_settings()
    context_chunks: list[dict[str, str | int]] = [
        {
            "source_file": r.chunk.source_file,
            "page_number": r.chunk.page_number,
            "text": r.chunk.text,
        }
        for r in results
    ]

    # Note: this SDK version's Messages API no longer exposes a `temperature`
    # param (CLAUDE.md calls for "low temperature" generation, but that knob
    # has been removed from the API surface) — grounding is enforced entirely
    # through the system prompt instead.
    message = client.messages.create(
        model=settings.anthropic_generation_model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_message(question, context_chunks)}],
    )
    answer = "".join(block.text for block in message.content if block.type == "text")

    citations = [
        Citation(source_file=r.chunk.source_file, page_number=r.chunk.page_number)
        for r in results
    ]
    return answer, citations
