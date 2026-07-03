class RAGException(Exception):
    status_code: int = 500
    detail: str = "Internal server error"

    def __init__(self, detail: str | None = None) -> None:
        if detail:
            self.detail = detail
        super().__init__(self.detail)


class InvalidPDFError(RAGException):
    status_code = 400
    detail = "Invalid PDF file"


class EmptyUploadError(RAGException):
    status_code = 400
    detail = "Empty upload — no files provided"


class LLMUnavailableError(RAGException):
    status_code = 503
    detail = "LLM service is currently unavailable"


class VectorDBUnavailableError(RAGException):
    status_code = 503
    detail = "Vector database is currently unavailable"


class EmbeddingModelError(RAGException):
    status_code = 500
    detail = "Embedding model failed to process input"


class DocumentNotFoundError(RAGException):
    status_code = 404
    detail = "Document not found"
