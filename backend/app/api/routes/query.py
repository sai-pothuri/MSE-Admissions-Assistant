from fastapi import APIRouter, HTTPException

from app.models.schemas import QueryRequest, QueryResponse
from app.services.query_pipeline import answer_query
from app.services.retrieval.vector_search import CollectionNotReadyError

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        return answer_query(request.question)
    except CollectionNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
