from unittest.mock import MagicMock

from app.models.schemas import ChunkPayload
from app.services.generation.generator import generate_answer
from app.services.retrieval.vector_search import SearchResult


def _fake_message(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    message = MagicMock()
    message.content = [block]
    return message


def test_generate_answer_only_cites_sources_referenced_in_the_answer():
    used = SearchResult(
        chunk=ChunkPayload(
            text="Tuition is $26,250 per semester.",
            source_file="faq.pdf",
            page_number=1,
            category="general",
        ),
        score=0.9,
    )
    unused = SearchResult(
        chunk=ChunkPayload(
            text="Unrelated content.",
            source_file="handbook.pdf",
            page_number=5,
            category="general",
        ),
        score=0.5,
    )
    client = MagicMock()
    client.messages.create.return_value = _fake_message(
        "Tuition is $26,250 per semester. Source: faq.pdf"
    )

    _, citations = generate_answer(client, "What is tuition?", [used, unused])

    assert len(citations) == 1
    assert citations[0].source_file == "faq.pdf"
    assert citations[0].page_number == 1


def test_generate_answer_dedupes_repeated_citations():
    result_a = SearchResult(
        chunk=ChunkPayload(
            text="Chunk A.", source_file="faq.pdf", page_number=1, category="general"
        ),
        score=0.9,
    )
    result_b = SearchResult(
        chunk=ChunkPayload(
            text="Chunk B.", source_file="faq.pdf", page_number=1, category="general"
        ),
        score=0.8,
    )
    client = MagicMock()
    client.messages.create.return_value = _fake_message("Answer. Source: faq.pdf")

    _, citations = generate_answer(client, "A question?", [result_a, result_b])

    assert len(citations) == 1
    assert citations[0].source_file == "faq.pdf"
    assert citations[0].page_number == 1


def test_generate_answer_disables_thinking_so_full_budget_is_visible_text():
    client = MagicMock()
    client.messages.create.return_value = _fake_message("An answer.")

    generate_answer(client, "A question?", [])

    _, kwargs = client.messages.create.call_args
    assert kwargs["thinking"] == {"type": "disabled"}
