import subprocess

from app.services.admin import file_manager
from app.services.admin.git_log import parse_git_log


def _init_repo(path):
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test Admin"], cwd=path, check=True)


def test_parse_git_log_returns_structured_entries_newest_first(tmp_path):
    _init_repo(tmp_path)
    kb_dir = tmp_path / "data" / "knowledge_base" / "general"
    file_manager.upload_file("a.pdf", b"v1", knowledge_base_dir=kb_dir, repo_root=tmp_path)
    file_manager.upload_file("b.pdf", b"v1", knowledge_base_dir=kb_dir, repo_root=tmp_path)

    entries = parse_git_log(repo_root=tmp_path, path=kb_dir)

    assert len(entries) == 2
    assert entries[0].file == "b.pdf"
    assert entries[0].action == "upload"
    assert entries[0].author == "Test Admin"
    assert entries[1].file == "a.pdf"


def test_parse_git_log_captures_replace_and_delete_actions(tmp_path):
    _init_repo(tmp_path)
    kb_dir = tmp_path / "data" / "knowledge_base" / "general"
    file_manager.upload_file("a.pdf", b"v1", knowledge_base_dir=kb_dir, repo_root=tmp_path)
    file_manager.replace_file("a.pdf", b"v2", knowledge_base_dir=kb_dir, repo_root=tmp_path)
    file_manager.delete_file("a.pdf", knowledge_base_dir=kb_dir, repo_root=tmp_path)

    entries = parse_git_log(repo_root=tmp_path, path=kb_dir)

    actions = [e.action for e in entries]
    assert actions == ["delete", "replace", "upload"]


def test_parse_git_log_returns_empty_list_when_no_history(tmp_path):
    _init_repo(tmp_path)
    kb_dir = tmp_path / "data" / "knowledge_base" / "general"
    kb_dir.mkdir(parents=True)

    assert parse_git_log(repo_root=tmp_path, path=kb_dir) == []
