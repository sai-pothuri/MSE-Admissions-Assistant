from pydantic import BaseModel


class ChunkPayload(BaseModel):
    text: str
    source_file: str
    page_number: int
    category: str
    auto_tagged: bool = False


class Citation(BaseModel):
    source_file: str
    page_number: int


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
