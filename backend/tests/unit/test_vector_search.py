from unittest.mock import MagicMock

import pytest
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.services.retrieval import vector_search


def test_search_raises_when_collection_missing(monkeypatch):
    mock_qdrant = MagicMock()
    mock_qdrant.collection_exists.return_value = False
    monkeypatch.setattr(vector_search, "get_qdrant_client", lambda: mock_qdrant)
    monkeypatch.setattr(vector_search, "get_voyage_client", lambda: MagicMock())

    with pytest.raises(vector_search.CollectionNotReadyError):
        vector_search.search("What is tuition?")

    mock_qdrant.query_points.assert_not_called()


def test_search_queries_when_collection_exists(monkeypatch):
    mock_qdrant = MagicMock()
    mock_qdrant.collection_exists.return_value = True
    mock_qdrant.query_points.return_value.points = []
    monkeypatch.setattr(vector_search, "get_qdrant_client", lambda: mock_qdrant)
    monkeypatch.setattr(vector_search, "get_voyage_client", lambda: MagicMock())
    monkeypatch.setattr(vector_search, "embed_query", lambda client, text: [0.1, 0.2])

    results = vector_search.search("What is tuition?")

    assert results == []
    mock_qdrant.query_points.assert_called_once()


def test_search_omits_filter_when_no_category_given(monkeypatch):
    mock_qdrant = MagicMock()
    mock_qdrant.collection_exists.return_value = True
    mock_qdrant.query_points.return_value.points = []
    monkeypatch.setattr(vector_search, "get_qdrant_client", lambda: mock_qdrant)
    monkeypatch.setattr(vector_search, "get_voyage_client", lambda: MagicMock())
    monkeypatch.setattr(vector_search, "embed_query", lambda client, text: [0.1, 0.2])

    vector_search.search("What is tuition?")

    _, kwargs = mock_qdrant.query_points.call_args
    assert kwargs["query_filter"] is None


def test_search_applies_category_filter_when_given(monkeypatch):
    mock_qdrant = MagicMock()
    mock_qdrant.collection_exists.return_value = True
    mock_qdrant.query_points.return_value.points = []
    monkeypatch.setattr(vector_search, "get_qdrant_client", lambda: mock_qdrant)
    monkeypatch.setattr(vector_search, "get_voyage_client", lambda: MagicMock())
    monkeypatch.setattr(vector_search, "embed_query", lambda client, text: [0.1, 0.2])

    vector_search.search("What is tuition?", category="tuition")

    _, kwargs = mock_qdrant.query_points.call_args
    expected = Filter(must=[FieldCondition(key="category", match=MatchValue(value="tuition"))])
    assert kwargs["query_filter"] == expected
