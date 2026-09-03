import subprocess

import pytest

from app.services.admin.git_utils import (
    UnsafeFilenameError,
    commit_paths,
    resolve_safe_pdf_path,
    run_git,
)


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


def test_commit_paths_is_a_no_op_when_content_is_byte_identical(tmp_path):
    """Regression test: re-committing byte-identical content (e.g.
    replace_file called with the same bytes already on disk) must not
    raise — `git commit` fails on "nothing to commit", which isn't
    actually an error condition here."""
    _init_repo(tmp_path)
    target = tmp_path / "file.txt"
    target.write_text("hello")
    first_hash = commit_paths(tmp_path, [target], "Add file.txt")

    target.write_text("hello")  # identical content
    second_hash = commit_paths(tmp_path, [target], "Re-add file.txt")

    assert second_hash == first_hash
    log_count = run_git("rev-list", "--count", "HEAD", cwd=tmp_path)
    assert log_count == "1"


def test_resolve_safe_pdf_path_accepts_a_normal_filename(tmp_path):
    kb_dir = tmp_path / "knowledge_base"
    kb_dir.mkdir()

    resolved = resolve_safe_pdf_path(kb_dir, "handbook.pdf")

    assert resolved == (kb_dir / "handbook.pdf").resolve()


@pytest.mark.parametrize(
    "filename",
    [
        "../../etc/passwd.pdf",
        "../secret.pdf",
        "../../../../etc/cron.d/evil.pdf",
        "subdir/../../escape.pdf",
    ],
)
def test_resolve_safe_pdf_path_rejects_path_traversal(tmp_path, filename):
    kb_dir = tmp_path / "knowledge_base"
    kb_dir.mkdir()

    with pytest.raises(UnsafeFilenameError):
        resolve_safe_pdf_path(kb_dir, filename)


def test_resolve_safe_pdf_path_rejects_an_absolute_path(tmp_path):
    kb_dir = tmp_path / "knowledge_base"
    kb_dir.mkdir()

    with pytest.raises(UnsafeFilenameError):
        resolve_safe_pdf_path(kb_dir, "/etc/passwd.pdf")


def test_resolve_safe_pdf_path_rejects_a_non_pdf_extension(tmp_path):
    kb_dir = tmp_path / "knowledge_base"
    kb_dir.mkdir()

    with pytest.raises(UnsafeFilenameError):
        resolve_safe_pdf_path(kb_dir, "not-a-pdf.txt")


def test_resolve_safe_pdf_path_rejects_the_directory_itself(tmp_path):
    kb_dir = tmp_path / "knowledge_base"
    kb_dir.mkdir()

    with pytest.raises(UnsafeFilenameError):
        resolve_safe_pdf_path(kb_dir, ".")
