from pathlib import Path

from app.services.ingestion import pdf_extraction


class _FakePage:
    def __init__(self, blocks):
        self._blocks = blocks

    def get_text(self, kind):
        assert kind == "blocks"
        return self._blocks


class _FakeDoc:
    def __init__(self, pages):
        self._pages = pages

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        return iter(self._pages)


def test_extract_blocks_skips_image_blocks(monkeypatch):
    # (x0, y0, x1, y1, text, block_no, block_type) — block_type 1 is an
    # image block; PyMuPDF still puts a non-empty placeholder in the text
    # slot, so it must be filtered by type, not by text truthiness.
    text_block = (0, 0, 10, 10, "Real paragraph text.\n", 0, 0)
    image_block = (0, 0, 10, 10, "<image placeholder>\n", 1, 1)
    fake_page = _FakePage([text_block, image_block])

    monkeypatch.setattr(pdf_extraction.fitz, "open", lambda path: _FakeDoc([fake_page]))

    blocks = pdf_extraction.extract_blocks(Path("fake.pdf"))

    assert len(blocks) == 1
    assert blocks[0].text == "Real paragraph text."
    assert blocks[0].page_number == 1
