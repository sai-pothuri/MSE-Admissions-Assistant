import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.services.admin.git_utils import KNOWLEDGE_BASE_DIR, REPO_ROOT

# Field separator / record separator (ASCII 0x1f/0x1e) — unambiguous
# delimiters for parsing `git log` output, since commit messages/authors
# can contain almost any other character.
_FIELD_SEP = "\x1f"
_RECORD_SEP = "\x1e"
_LOG_FORMAT = f"%H{_FIELD_SEP}%aI{_FIELD_SEP}%an{_FIELD_SEP}%B{_RECORD_SEP}"


@dataclass(frozen=True)
class EditLogEntry:
    commit_hash: str
    timestamp: str  # ISO 8601, from git's %aI
    author: str
    action: str
    file: str


def _parse_trailers(body: str) -> dict[str, str]:
    trailers = {}
    for line in body.splitlines():
        key, sep, value = line.partition(":")
        if sep:
            trailers[key.strip()] = value.strip()
    return trailers


def parse_git_log(
    repo_root: Path = REPO_ROOT, path: Path = KNOWLEDGE_BASE_DIR
) -> list[EditLogEntry]:
    """Parses the commit history for `path` into structured entries, using
    the `Action:`/`File:` trailers `file_manager.py` writes on every
    commit. Newest first (git's default log order)."""
    head_check = subprocess.run(
        ["git", "rev-parse", "--verify", "-q", "HEAD"], cwd=repo_root, capture_output=True
    )
    if head_check.returncode != 0:
        # A repo with zero commits ever made — `git log` itself would exit
        # non-zero here rather than returning an empty result.
        return []

    result = subprocess.run(
        ["git", "log", f"--format={_LOG_FORMAT}", "--", str(path)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    entries = []
    for record in result.stdout.split(_RECORD_SEP):
        record = record.strip("\n")
        if not record:
            continue
        commit_hash, timestamp, author, body = record.split(_FIELD_SEP, 3)
        trailers = _parse_trailers(body)
        entries.append(
            EditLogEntry(
                commit_hash=commit_hash,
                timestamp=timestamp,
                author=author,
                action=trailers.get("Action", "unknown"),
                file=trailers.get("File", ""),
            )
        )
    return entries
