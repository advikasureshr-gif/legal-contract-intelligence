class DocumentProcessingError(Exception):
    """Base exception for document processing errors."""

    def __init__(self, error_code: str, message: str):
        self.error_code = error_code
        self.message = message
        super().__init__(message)

    def to_dict(self):
        return {
            "status": "error",
            "error_code": self.error_code,
            "message": self.message
        }
