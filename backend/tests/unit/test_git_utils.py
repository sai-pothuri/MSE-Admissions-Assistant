import subprocess

from app.services.admin.git_utils import commit_paths, run_git


def _init_repo(path):
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)


def test_commit_paths_creates_a_commit_with_the_given_message(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "file.txt").write_text("hello")

    commit_hash = commit_paths(tmp_path, [tmp_path / "file.txt"], "Add file.txt")

    log = run_git("log", "-1", "--format=%H %s", cwd=tmp_path)
    assert log.startswith(commit_hash)
    assert "Add file.txt" in log


def test_commit_paths_only_stages_the_given_paths(tmp_path):
    """Regression test: must never `git add -A` — an unrelated uncommitted
    file elsewhere in the repo must not get swept into this commit."""
    _init_repo(tmp_path)
    (tmp_path / "intended.txt").write_text("intended change")
    (tmp_path / "unrelated.txt").write_text("someone else's in-progress work")

    commit_paths(tmp_path, [tmp_path / "intended.txt"], "Add intended.txt")

    status = run_git("status", "--porcelain", cwd=tmp_path)
    assert "unrelated.txt" in status
    assert "intended.txt" not in status


def test_commit_paths_stages_a_deletion(tmp_path):
    _init_repo(tmp_path)
    target = tmp_path / "file.txt"
    target.write_text("hello")
    commit_paths(tmp_path, [target], "Add file.txt")

    target.unlink()
    commit_hash = commit_paths(tmp_path, [target], "Delete file.txt")

    log = run_git("log", "-1", "--format=%H %s", cwd=tmp_path)
    assert log.startswith(commit_hash)
    files_changed = run_git("show", "--stat", "--format=", commit_hash, cwd=tmp_path)
    assert "file.txt" in files_changed
