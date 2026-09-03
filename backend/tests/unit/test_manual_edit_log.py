import subprocess

from app.services.admin import manual_edit_log


def _init_repo(path):
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)


def test_append_entry_writes_and_commits(tmp_path):
    _init_repo(tmp_path)
    log_path = tmp_path / "data" / "manual_edit_log.jsonl"

    entry = manual_edit_log.append_entry(
        "retag", "chunk-1 -> tuition", log_path=log_path, repo_root=tmp_path
    )

    assert entry.action == "retag"
    assert log_path.exists()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=tmp_path, capture_output=True, text=True
    ).stdout
    assert status.strip() == ""  # nothing left uncommitted
    log_message = subprocess.run(
        ["git", "log", "-1", "--format=%s%n%b"], cwd=tmp_path, capture_output=True, text=True
    ).stdout
    assert "[edit-log] retag: chunk-1 -> tuition" in log_message


def test_append_entry_appends_without_overwriting_prior_entries(tmp_path):
    _init_repo(tmp_path)
    log_path = tmp_path / "data" / "manual_edit_log.jsonl"

    manual_edit_log.append_entry("retag", "first", log_path=log_path, repo_root=tmp_path)
    manual_edit_log.append_entry("promote", "second", log_path=log_path, repo_root=tmp_path)

    entries = manual_edit_log.read_entries(log_path=log_path)
    assert [e.detail for e in entries] == ["first", "second"]
    assert [e.action for e in entries] == ["retag", "promote"]


def test_read_entries_returns_empty_list_when_log_does_not_exist(tmp_path):
    assert manual_edit_log.read_entries(log_path=tmp_path / "nonexistent.jsonl") == []
