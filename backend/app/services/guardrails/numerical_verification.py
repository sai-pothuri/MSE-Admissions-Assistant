import re
from dataclasses import dataclass

# Currency, percentages, and plain numbers (with optional thousands
# separators/decimals) — covers the numeric formats that show up in tuition
# and deadline content. Deliberately literal: no normalization ("$50,000"
# won't match "fifty thousand"), per the low-complexity default in plan.md.
NUMBER_PATTERN = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    unverified_numbers: list[str]


def extract_numbers(text: str) -> list[str]:
    return NUMBER_PATTERN.findall(text)


def verify(answer: str, cited_texts: list[str]) -> VerificationResult:
    """Every number in `answer` must literally appear in at least one of
    `cited_texts` (the chunks the answer was generated from)."""
    numbers = extract_numbers(answer)
    combined_source = "\n".join(cited_texts)
    unverified = [n for n in numbers if n not in combined_source]
    return VerificationResult(passed=not unverified, unverified_numbers=unverified)
