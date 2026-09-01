from app.services.ingestion.chunking import MAX_CHARS, chunk_blocks
from app.services.ingestion.pdf_extraction import ExtractedBlock


def test_chunk_blocks_merges_small_blocks_into_one_chunk():
    blocks = [
        ExtractedBlock(text="First paragraph.", page_number=1),
        ExtractedBlock(text="Second paragraph.", page_number=1),
    ]
    chunks = chunk_blocks(blocks)
    assert len(chunks) == 1
    assert "First paragraph." in chunks[0].text
    assert "Second paragraph." in chunks[0].text
    assert chunks[0].page_number == 1


def test_chunk_blocks_splits_when_exceeding_max_chars():
    big_block_a = ExtractedBlock(text="A" * (MAX_CHARS - 10), page_number=1)
    big_block_b = ExtractedBlock(text="B" * 500, page_number=2)
    chunks = chunk_blocks([big_block_a, big_block_b])
    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    # chunks[1] leads with the page-1 overlap tail, so it's attributed to
    # page 1 (where its content actually starts), not page 2's block.
    assert chunks[1].page_number == 1


def test_chunk_blocks_never_splits_a_single_block_mid_paragraph():
    huge_block = ExtractedBlock(text="X" * (MAX_CHARS * 3), page_number=1)
    chunks = chunk_blocks([huge_block])
    assert len(chunks) == 1
    assert chunks[0].text == huge_block.text


def test_chunk_blocks_empty_input_returns_no_chunks():
    assert chunk_blocks([]) == []


def test_chunk_blocks_carries_overlap_into_next_chunk():
    big_block_a = ExtractedBlock(text="A" * (MAX_CHARS - 10), page_number=1)
    big_block_b = ExtractedBlock(text="B" * 500, page_number=2)
    chunks = chunk_blocks([big_block_a, big_block_b])
    assert chunks[1].text.startswith("A")
    # The carried-over overlap text is from page 1, so the citation page
    # must follow the tail's origin, not the block that triggered the split.
    assert chunks[1].page_number == 1


def test_chunk_blocks_overlap_page_follows_tail_across_three_pages():
    block_a = ExtractedBlock(text="A" * (MAX_CHARS - 10), page_number=1)
    block_b = ExtractedBlock(text="B" * (MAX_CHARS - 10), page_number=2)
    block_c = ExtractedBlock(text="C" * 500, page_number=3)
    chunks = chunk_blocks([block_a, block_b, block_c])
    assert len(chunks) == 3
    assert chunks[0].page_number == 1
    assert chunks[1].page_number == 1  # leads with page-1 overlap tail
    assert chunks[2].page_number == 2  # leads with page-2 overlap tail
