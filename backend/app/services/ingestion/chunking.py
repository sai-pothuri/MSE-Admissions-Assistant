from dataclasses import dataclass

from app.services.ingestion.pdf_extraction import ExtractedBlock

# Rough token estimate (~4 chars/token for English text) — good enough for
# sizing chunks at this scale; no tokenizer dependency needed.
CHARS_PER_TOKEN = 4
TARGET_TOKENS = 400
MAX_CHARS = TARGET_TOKENS * CHARS_PER_TOKEN
OVERLAP_TOKENS = 50
OVERLAP_CHARS = OVERLAP_TOKENS * CHARS_PER_TOKEN


@dataclass
class Chunk:
    text: str
    page_number: int


def chunk_blocks(blocks: list[ExtractedBlock]) -> list[Chunk]:
    """Structure-aware chunking: accumulate whole paragraph blocks up to
    ~MAX_CHARS, never splitting a block mid-paragraph. A new chunk starts
    with a character-based overlap carried from the tail of the previous
    chunk, so context isn't lost at chunk boundaries."""
    chunks: list[Chunk] = []
    current_texts: list[str] = []
    current_len = 0
    current_page: int | None = None  # page the chunk's content starts on
    last_page: int | None = None  # page of the most recently appended text

    def flush() -> None:
        nonlocal current_texts, current_len, current_page
        if current_texts:
            assert current_page is not None
            chunks.append(Chunk(text="\n\n".join(current_texts), page_number=current_page))
        current_texts = []
        current_len = 0
        current_page = None

    for block in blocks:
        block_len = len(block.text)

        if current_texts and current_len + block_len > MAX_CHARS:
            tail = current_texts[-1][-OVERLAP_CHARS:] if current_texts else ""
            # The tail is drawn from the most recently appended text, which
            # may be on a later page than the chunk's own start page.
            tail_page = last_page
            flush()
            if tail:
                current_texts = [tail]
                current_len = len(tail)
                current_page = tail_page

        if current_page is None:
            current_page = block.page_number

        current_texts.append(block.text)
        current_len += block_len
        last_page = block.page_number

    flush()
    return chunks
