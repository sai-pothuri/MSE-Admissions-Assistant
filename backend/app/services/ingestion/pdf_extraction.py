from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class ExtractedBlock:
    text: str
    page_number: int


def extract_blocks(pdf_path: Path) -> list[ExtractedBlock]:
    blocks: list[ExtractedBlock] = []
    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            for block in page.get_text("blocks"):
                text = block[4].strip()
                if text:
                    blocks.append(ExtractedBlock(text=text, page_number=page_index))
    return blocks
