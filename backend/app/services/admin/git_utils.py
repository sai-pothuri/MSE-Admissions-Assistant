import subprocess
from pathlib import Path

# backend/app/services/admin/git_utils.py -> repo root is four parents up.
REPO_ROOT = Path(__file__).resolve().parents[4]
KNOWLEDGE_BASE_DIR = REPO_ROOT / "data" / "knowledge_base" / "general"


class UnsafeFilenameError(ValueError):
    pass


def resolve_safe_pdf_path(knowledge_base_dir: Path, filename: str) -> Path:
    """Resolves `filename` against `knowledge_base_dir`, rejecting anything
    that would escape it (e.g. `../../etc/passwd`) or isn't a `.pdf` —
    every place a client-controlled filename turns into a filesystem path
    (upload/replace/delete, chunk preview) must go through this, not just
    join the path directly. Checking the *resolved* path is a descendant
    of the directory is the robust defense here — a character-allowlist
    regex on the filename alone is easy to get subtly wrong."""
    if not filename.lower().endswith(".pdf"):
        raise UnsafeFilenameError(f"'{filename}' must be a .pdf file")
    base = knowledge_base_dir.resolve()
    target = (knowledge_base_dir / filename).resolve()
    if not target.is_relative_to(base) or target == base:
        raise UnsafeFilenameError(f"'{filename}' is not a valid filename")
    return target


def run_git(*args: str, cwd: Path) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def commit_paths(repo_root: Path, paths: list[Path], message: str) -> str:
    """Stages exactly `paths` (never `git add -A`, which could sweep in
    unrelated uncommitted changes elsewhere in the repo) and commits them.
    Returns the new commit hash — the current HEAD if `paths` turned out
    to be byte-identical to what's already committed (e.g. `replace_file`
    called with content matching the existing file): `git commit` would
    otherwise fail on "nothing to commit" and raise, when "nothing
    changed" isn't actually an error condition here."""
    path_args = [str(p) for p in paths]
    run_git("add", *path_args, cwd=repo_root)
    if not run_git("status", "--porcelain", "--", *path_args, cwd=repo_root):
        return run_git("rev-parse", "HEAD", cwd=repo_root)
    run_git("commit", "-m", message, cwd=repo_root)
    return run_git("rev-parse", "HEAD", cwd=repo_root)
