from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    text: str
    score: float


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    chunks: int
    pages: int


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks: int
    pages: int


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)


class SourceNode(BaseModel):
    page: int
    filename: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceNode]


class HealthResponse(BaseModel):
    status: str


class ErrorResponse(BaseModel):
    detail: str
    code: str | None = None
