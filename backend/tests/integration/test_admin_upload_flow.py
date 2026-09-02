import io

import pytest
from fastapi.testclient import TestClient

from app.api.routes import admin_chunks, admin_files, admin_staging
from app.config.settings import get_settings
from app.main import app
from app.services.admin.file_manager import CommitInfo
from app.services.admin.reindex import ReindexConfirmationRequiredError, ReindexWarning
from app.services.admin.retag import ChunkRecord
from app.services.admin.staging import PromotionResult


@pytest.fixture
def client():
    client = TestClient(app)
    client.post("/admin/login", json={"password": get_settings().admin_password})
    return client


def _pdf_upload(filename="handbook.pdf"):
    return {"file": (filename, io.BytesIO(b"%PDF-1.4 fake content"), "application/pdf")}


def test_upload_commits_and_indexes_into_staging(client, monkeypatch):
    monkeypatch.setattr(
        admin_files.file_manager,
        "upload_file",
        lambda filename, content: CommitInfo(
            commit_hash="abc123", action="upload", filename=filename
        ),
    )
    monkeypatch.setattr(admin_files, "index_file", lambda path, collection: 4)

    response = client.post("/admin/files/upload", files=_pdf_upload())

    assert response.status_code == 201
    body = response.json()
    assert body["commit_hash"] == "abc123"
    assert body["chunks_indexed"] == 4
    assert body["collection"] == get_settings().qdrant_collection_staging


def test_upload_rejects_a_filename_that_already_exists(client, monkeypatch):
    def raise_exists(filename, content):
        raise FileExistsError(f"'{filename}' already exists")

    monkeypatch.setattr(admin_files.file_manager, "upload_file", raise_exists)

    response = client.post("/admin/files/upload", files=_pdf_upload())

    assert response.status_code == 409


def test_list_chunks_for_a_file(client, monkeypatch):
    monkeypatch.setattr(
        admin_chunks.retag,
        "list_chunks",
        lambda collection, filename: [
            ChunkRecord(
                point_id="p1",
                text="Tuition is $26,250.",
                source_file=filename,
                page_number=1,
                category="tuition",
                auto_tagged=True,
            )
        ],
    )

    response = client.get("/admin/chunks", params={"filename": "handbook.pdf"})

    assert response.status_code == 200
    assert response.json()[0]["category"] == "tuition"


def test_retag_a_chunk(client, monkeypatch):
    called = {}
    monkeypatch.setattr(
        admin_chunks.retag,
        "retag_chunk",
        lambda collection, point_id, category: called.update(
            collection=collection, point_id=point_id, category=category
        ),
    )

    response = client.patch("/admin/chunks/p1", params={"category": "tuition"})

    assert response.status_code == 200
    assert called == {
        "collection": get_settings().qdrant_collection_staging,
        "point_id": "p1",
        "category": "tuition",
    }


def test_retag_rejects_an_invalid_category(client, monkeypatch):
    from app.services.admin.retag import InvalidCategoryError

    def raise_invalid(collection, point_id, category):
        raise InvalidCategoryError(f"'{category}' is not a known category")

    monkeypatch.setattr(admin_chunks.retag, "retag_chunk", raise_invalid)

    response = client.patch("/admin/chunks/p1", params={"category": "not_real"})

    assert response.status_code == 400


def test_replace_requires_confirmation_when_manual_corrections_exist(client, monkeypatch):
    monkeypatch.setattr(
        admin_files.file_manager,
        "replace_file",
        lambda filename, content: CommitInfo(
            commit_hash="def456", action="replace", filename=filename
        ),
    )

    def raise_confirmation_required(path, collection, confirm):
        raise ReindexConfirmationRequiredError(
            ReindexWarning(has_manual_corrections=True, corrected_chunk_count=2)
        )

    monkeypatch.setattr(admin_files.reindex, "reindex_file", raise_confirmation_required)

    response = client.put("/admin/files/handbook.pdf", files=_pdf_upload())

    assert response.status_code == 409
    assert response.json()["detail"]["corrected_chunk_count"] == 2


def test_replace_proceeds_when_confirmed(client, monkeypatch):
    monkeypatch.setattr(
        admin_files.file_manager,
        "replace_file",
        lambda filename, content: CommitInfo(
            commit_hash="def456", action="replace", filename=filename
        ),
    )
    monkeypatch.setattr(admin_files.reindex, "reindex_file", lambda path, collection, confirm: 6)

    response = client.put(
        "/admin/files/handbook.pdf", files=_pdf_upload(), params={"confirm": "true"}
    )

    assert response.status_code == 200
    assert response.json()["chunks_indexed"] == 6


def test_promote_moves_staged_chunks_to_prod(client, monkeypatch):
    monkeypatch.setattr(
        admin_staging.staging,
        "promote_file",
        lambda filename: PromotionResult(source_file=filename, promoted_count=3),
    )

    response = client.post("/admin/staging/promote/handbook.pdf")

    assert response.status_code == 200
    assert response.json()["promoted_count"] == 3


def test_promote_returns_404_when_nothing_is_staged(client, monkeypatch):
    monkeypatch.setattr(
        admin_staging.staging,
        "promote_file",
        lambda filename: PromotionResult(source_file=filename, promoted_count=0),
    )

    response = client.post("/admin/staging/promote/nonexistent.pdf")

    assert response.status_code == 404


def test_delete_removes_the_file_and_chunks_from_both_collections(client, monkeypatch):
    monkeypatch.setattr(
        admin_files.file_manager,
        "delete_file",
        lambda filename: CommitInfo(commit_hash="ghi789", action="delete", filename=filename),
    )
    deleted_from = []
    monkeypatch.setattr(
        admin_files.retag,
        "delete_chunks",
        lambda collection, filename: deleted_from.append(collection),
    )

    response = client.delete("/admin/files/handbook.pdf")

    assert response.status_code == 200
    settings = get_settings()
    assert deleted_from == [settings.qdrant_collection_staging, settings.qdrant_collection_prod]


def test_full_admin_flow_requires_authentication_at_every_step():
    """Regression test: every admin route must reject an unauthenticated
    client, not just one of them."""
    client = TestClient(app)

    assert client.get("/admin/files").status_code == 401
    assert client.post("/admin/files/upload", files=_pdf_upload()).status_code == 401
    assert client.get("/admin/chunks", params={"filename": "x.pdf"}).status_code == 401
    assert client.patch("/admin/chunks/p1", params={"category": "tuition"}).status_code == 401
    assert client.post("/admin/staging/promote/x.pdf").status_code == 401
    assert client.delete("/admin/files/x.pdf").status_code == 401
