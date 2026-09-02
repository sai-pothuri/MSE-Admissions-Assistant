import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from app.services.admin.git_utils import REPO_ROOT, commit_paths

MANUAL_EDIT_LOG_PATH = REPO_ROOT / "data" / "manual_edit_log.jsonl"


@dataclass(frozen=True)
class ManualEditEntry:
    action: str  # "retag" | "promote"
    detail: str
    timestamp: float


def append_entry(
    action: str,
    detail: str,
    log_path: Path = MANUAL_EDIT_LOG_PATH,
    repo_root: Path = REPO_ROOT,
) -> ManualEditEntry:
    """Records a change that touches Qdrant only (a re-tag or a staging
    promotion) — git history on `data/knowledge_base/` doesn't see these,
    but CLAUDE.md's auditability goal still wants them logged, so this
    appends a line and commits it, keeping the whole edit history
    git-tracked rather than splitting it across git + an untracked file."""
    entry = ManualEditEntry(action=action, detail=detail, timestamp=time.time())
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps(asdict(entry)) + "\n")
    message = f"[edit-log] {action}: {detail}\n\nAction: {action}\nDetail: {detail}"
    commit_paths(repo_root, [log_path], message)
    return entry


def read_entries(log_path: Path = MANUAL_EDIT_LOG_PATH) -> list[ManualEditEntry]:
    if not log_path.exists():
        return []
    entries = []
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(ManualEditEntry(**json.loads(line)))
    return entries
