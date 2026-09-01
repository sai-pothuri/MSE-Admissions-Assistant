from fastapi import APIRouter

from app.clients.anthropic_client import get_anthropic_client
from app.models.schemas import QueryRequest, QueryResponse
from app.services.generation.generator import generate_answer
from app.services.retrieval.vector_search import search

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    results = search(request.question)
    answer, citations = generate_answer(get_anthropic_client(), request.question, results)
    return QueryResponse(answer=answer, citations=citations)
