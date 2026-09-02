import subprocess
from pathlib import Path

# backend/app/services/admin/git_utils.py -> repo root is four parents up.
REPO_ROOT = Path(__file__).resolve().parents[4]
KNOWLEDGE_BASE_DIR = REPO_ROOT / "data" / "knowledge_base" / "general"


def run_git(*args: str, cwd: Path) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def commit_paths(repo_root: Path, paths: list[Path], message: str) -> str:
    """Stages exactly `paths` (never `git add -A`, which could sweep in
    unrelated uncommitted changes elsewhere in the repo) and commits them.
    Returns the new commit hash."""
    run_git("add", *[str(p) for p in paths], cwd=repo_root)
    run_git("commit", "-m", message, cwd=repo_root)
    return run_git("rev-parse", "HEAD", cwd=repo_root)
