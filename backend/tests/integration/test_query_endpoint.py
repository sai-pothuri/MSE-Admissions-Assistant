from fastapi.testclient import TestClient

from app.api.routes import query as query_route
from app.main import app
from app.models.schemas import Citation
from app.services.retrieval.vector_search import CollectionNotReadyError


def test_query_endpoint_returns_answer_and_citations(monkeypatch):
    monkeypatch.setattr(query_route, "search", lambda question: [])
    monkeypatch.setattr(
        query_route,
        "generate_answer",
        lambda client, question, results: (
            "The MSE program requires a bachelor's degree.",
            [Citation(source_file="handbook.pdf", page_number=3)],
        ),
    )
    monkeypatch.setattr(query_route, "get_anthropic_client", lambda: object())

    client = TestClient(app)
    response = client.post("/query", json={"question": "What are the admission requirements?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "The MSE program requires a bachelor's degree."
    assert body["citations"] == [{"source_file": "handbook.pdf", "page_number": 3}]


def test_query_endpoint_returns_503_when_collection_not_ready(monkeypatch):
    def raise_not_ready(question):
        raise CollectionNotReadyError("Collection 'mse_kb_prod' does not exist yet.")

    monkeypatch.setattr(query_route, "search", raise_not_ready)

    client = TestClient(app)
    response = client.post("/query", json={"question": "What is tuition?"})

    assert response.status_code == 503
    assert "does not exist yet" in response.json()["detail"]
