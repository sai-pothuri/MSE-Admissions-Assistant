import re
from dataclasses import dataclass

# Currency, percentages, and plain numbers (with optional thousands
# separators/decimals) — covers the numeric formats that show up in tuition
# and deadline content. Deliberately literal: no normalization ("$50,000"
# won't match "fifty thousand"), per the low-complexity default in plan.md.
# The thousands-grouped alternative requires each comma to be followed by
# exactly 3 digits — a loose `[\d,]*` char class would also swallow a
# trailing comma that's just punctuation (e.g. extracting "$7,500," instead
# of "$7,500" from "...worth $7,500, not $10,000", which then fails to
# match the source text's "$7,500" and false-positives the whole answer).
NUMBER_PATTERN = re.compile(r"\$?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|\$?\d+(?:\.\d+)?%?")

# The model is instructed to cite "(Source: file, page N)" — page numbers
# are citation metadata, not factual claims, and won't literally appear in
# the cited chunk's body text, so the whole parenthetical citation must be
# excluded before extraction (not just the page number) — a source
# filename can itself contain a digit (e.g. "SCS_S3D_MSE.pdf"), which would
# otherwise leak through as an "unverified number" too.
PAGE_CITATION_PATTERN = re.compile(r"\([^()]*\bpages?\s+\d+(?:\s*,\s*\d+)*[^()]*\)", re.IGNORECASE)


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    unverified_numbers: list[str]


def extract_numbers(text: str) -> list[str]:
    text_without_page_citations = PAGE_CITATION_PATTERN.sub("", text)
    return NUMBER_PATTERN.findall(text_without_page_citations)


def _appears_as_whole_number(number: str, source: str) -> bool:
    """Plain substring containment would let a truncated/wrong number pass
    just because it's embedded in a longer correct one (e.g. "6,250" is a
    substring of "$26,250.00"). Require it not be immediately adjacent to
    another digit on either side, so it can't be a fragment of a larger
    numeral."""
    pattern = rf"(?<!\d){re.escape(number)}(?!\d)"
    return re.search(pattern, source) is not None


def verify(answer: str, cited_texts: list[str]) -> VerificationResult:
    """Every number in `answer` must literally appear, as a whole number
    (not embedded in a longer one), in at least one of `cited_texts` (the
    chunks the answer was generated from)."""
    numbers = extract_numbers(answer)
    combined_source = "\n".join(cited_texts)
    unverified = [n for n in numbers if not _appears_as_whole_number(n, combined_source)]
    return VerificationResult(passed=not unverified, unverified_numbers=unverified)
