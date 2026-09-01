import anthropic

from app.config.settings import get_settings
from app.models.schemas import Citation
from app.services.generation.prompt_templates import SYSTEM_PROMPT, build_user_message
from app.services.retrieval.vector_search import SearchResult

MAX_TOKENS = 1024


def referenced_results(answer: str, results: list[SearchResult]) -> list[SearchResult]:
    """Chunks whose source_file is actually mentioned in the answer text —
    used both to build citations and to scope numerical verification to the
    sources the model actually drew from, deduped by (source_file, page)."""
    seen: set[tuple[str, int]] = set()
    referenced: list[SearchResult] = []
    for r in results:
        if r.chunk.source_file not in answer:
            continue
        key = (r.chunk.source_file, r.chunk.page_number)
        if key in seen:
            continue
        seen.add(key)
        referenced.append(r)
    return referenced


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
    # through the system prompt instead. Thinking is explicitly disabled so
    # the full MAX_TOKENS budget goes to the visible answer, not adaptive
    # reasoning tokens drawn from the same budget.
    message = client.messages.create(
        model=settings.anthropic_generation_model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        thinking={"type": "disabled"},
        messages=[{"role": "user", "content": build_user_message(question, context_chunks)}],
    )
    answer = "".join(block.text for block in message.content if block.type == "text")

    # Only cite sources the model actually referenced in the answer text
    # (per the prompt's "cite the source file(s) you used" instruction) —
    # a retrieved-but-unused chunk (e.g. on a refusal) shouldn't be cited.
    citations = [
        Citation(source_file=r.chunk.source_file, page_number=r.chunk.page_number)
        for r in referenced_results(answer, results)
    ]

    return answer, citations
