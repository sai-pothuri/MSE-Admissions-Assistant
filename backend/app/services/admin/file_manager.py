from dataclasses import dataclass
from pathlib import Path

from app.services.admin.git_utils import (
    KNOWLEDGE_BASE_DIR,
    REPO_ROOT,
    commit_paths,
    resolve_safe_pdf_path,
)


@dataclass(frozen=True)
class CommitInfo:
    commit_hash: str
    action: str
    filename: str


def _commit(repo_root: Path, target: Path, action: str, filename: str) -> CommitInfo:
    message = f"[{action}] {filename}\n\nAction: {action}\nFile: {filename}"
    commit_hash = commit_paths(repo_root, [target], message)
    return CommitInfo(commit_hash=commit_hash, action=action, filename=filename)


def upload_file(
    filename: str,
    content: bytes,
    knowledge_base_dir: Path = KNOWLEDGE_BASE_DIR,
    repo_root: Path = REPO_ROOT,
) -> CommitInfo:
    knowledge_base_dir.mkdir(parents=True, exist_ok=True)
    target = resolve_safe_pdf_path(knowledge_base_dir, filename)
    if target.exists():
        raise FileExistsError(f"'{filename}' already exists — use replace_file instead.")
    target.write_bytes(content)
    return _commit(repo_root, target, "upload", filename)


def replace_file(
    filename: str,
    content: bytes,
    knowledge_base_dir: Path = KNOWLEDGE_BASE_DIR,
    repo_root: Path = REPO_ROOT,
) -> CommitInfo:
    target = resolve_safe_pdf_path(knowledge_base_dir, filename)
    if not target.exists():
        raise FileNotFoundError(f"'{filename}' does not exist — use upload_file instead.")
    target.write_bytes(content)
    return _commit(repo_root, target, "replace", filename)


def delete_file(
    filename: str,
    knowledge_base_dir: Path = KNOWLEDGE_BASE_DIR,
    repo_root: Path = REPO_ROOT,
) -> CommitInfo:
    target = resolve_safe_pdf_path(knowledge_base_dir, filename)
    if not target.exists():
        raise FileNotFoundError(f"'{filename}' does not exist.")
    target.unlink()
    return _commit(repo_root, target, "delete", filename)


def list_files(knowledge_base_dir: Path = KNOWLEDGE_BASE_DIR) -> list[str]:
    if not knowledge_base_dir.exists():
        return []
    return sorted(p.name for p in knowledge_base_dir.glob("*.pdf"))
