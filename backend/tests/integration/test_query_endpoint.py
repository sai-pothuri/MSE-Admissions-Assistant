from fastapi.testclient import TestClient

from app.api.routes import query as query_route
from app.main import app
from app.models.schemas import Citation


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
