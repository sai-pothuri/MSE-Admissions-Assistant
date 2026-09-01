from fastapi import APIRouter, HTTPException

from app.clients.anthropic_client import get_anthropic_client
from app.models.schemas import QueryRequest, QueryResponse
from app.services.generation.generator import generate_answer
from app.services.retrieval.vector_search import CollectionNotReadyError, search

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        results = search(request.question)
    except CollectionNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    answer, citations = generate_answer(get_anthropic_client(), request.question, results)
    return QueryResponse(answer=answer, citations=citations)
