from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class ExtractedBlock:
    text: str
    page_number: int


# PyMuPDF block tuples are (x0, y0, x1, y1, text, block_no, block_type),
# where block_type 0 is text and 1 is image. Image blocks carry a non-empty
# placeholder in the text slot, not real content, so they must be excluded
# explicitly rather than relying on an `if text:` truthiness check.
TEXT_BLOCK_TYPE = 0


def extract_blocks(pdf_path: Path) -> list[ExtractedBlock]:
    blocks: list[ExtractedBlock] = []
    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            for block in page.get_text("blocks"):
                if block[6] != TEXT_BLOCK_TYPE:
                    continue
                text = block[4].strip()
                if text:
                    blocks.append(ExtractedBlock(text=text, page_number=page_index))
    return blocks
