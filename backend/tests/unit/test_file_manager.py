import subprocess

import pytest

from app.services.admin import file_manager


@pytest.fixture
def repo(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    kb_dir = tmp_path / "data" / "knowledge_base" / "general"
    return tmp_path, kb_dir


def test_upload_file_writes_content_and_commits(repo):
    repo_root, kb_dir = repo

    commit = file_manager.upload_file(
        "handbook.pdf", b"%PDF-1.4 fake", knowledge_base_dir=kb_dir, repo_root=repo_root
    )

    assert (kb_dir / "handbook.pdf").read_bytes() == b"%PDF-1.4 fake"
    assert commit.action == "upload"
    assert commit.filename == "handbook.pdf"
    log = subprocess.run(
        ["git", "log", "-1", "--format=%s%n%b"], cwd=repo_root, capture_output=True, text=True
    ).stdout
    assert "[upload] handbook.pdf" in log
    assert "Action: upload" in log
    assert "File: handbook.pdf" in log


def test_upload_file_raises_if_file_already_exists(repo):
    repo_root, kb_dir = repo
    file_manager.upload_file("handbook.pdf", b"v1", knowledge_base_dir=kb_dir, repo_root=repo_root)

    with pytest.raises(FileExistsError):
        file_manager.upload_file(
            "handbook.pdf", b"v2", knowledge_base_dir=kb_dir, repo_root=repo_root
        )


def test_replace_file_overwrites_content_and_commits(repo):
    repo_root, kb_dir = repo
    file_manager.upload_file("handbook.pdf", b"v1", knowledge_base_dir=kb_dir, repo_root=repo_root)

    commit = file_manager.replace_file(
        "handbook.pdf", b"v2", knowledge_base_dir=kb_dir, repo_root=repo_root
    )

    assert (kb_dir / "handbook.pdf").read_bytes() == b"v2"
    assert commit.action == "replace"


def test_replace_file_with_identical_content_does_not_raise(repo):
    """Regression test: replacing a file with byte-identical content used
    to crash (git has nothing to commit) instead of succeeding as a no-op."""
    repo_root, kb_dir = repo
    file_manager.upload_file("handbook.pdf", b"v1", knowledge_base_dir=kb_dir, repo_root=repo_root)

    commit = file_manager.replace_file(
        "handbook.pdf", b"v1", knowledge_base_dir=kb_dir, repo_root=repo_root
    )

    assert (kb_dir / "handbook.pdf").read_bytes() == b"v1"
    assert commit.action == "replace"


def test_replace_file_raises_if_file_does_not_exist(repo):
    repo_root, kb_dir = repo

    with pytest.raises(FileNotFoundError):
        file_manager.replace_file(
            "missing.pdf", b"v1", knowledge_base_dir=kb_dir, repo_root=repo_root
        )


def test_delete_file_removes_content_and_commits(repo):
    repo_root, kb_dir = repo
    file_manager.upload_file("handbook.pdf", b"v1", knowledge_base_dir=kb_dir, repo_root=repo_root)

    commit = file_manager.delete_file(
        "handbook.pdf", knowledge_base_dir=kb_dir, repo_root=repo_root
    )

    assert not (kb_dir / "handbook.pdf").exists()
    assert commit.action == "delete"


def test_delete_file_raises_if_file_does_not_exist(repo):
    repo_root, kb_dir = repo

    with pytest.raises(FileNotFoundError):
        file_manager.delete_file("missing.pdf", knowledge_base_dir=kb_dir, repo_root=repo_root)


def test_list_files_returns_sorted_pdf_names(repo):
    repo_root, kb_dir = repo
    file_manager.upload_file("b.pdf", b"v1", knowledge_base_dir=kb_dir, repo_root=repo_root)
    file_manager.upload_file("a.pdf", b"v1", knowledge_base_dir=kb_dir, repo_root=repo_root)

    assert file_manager.list_files(knowledge_base_dir=kb_dir) == ["a.pdf", "b.pdf"]


def test_list_files_returns_empty_list_when_directory_does_not_exist(tmp_path):
    assert file_manager.list_files(knowledge_base_dir=tmp_path / "nonexistent") == []
