from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class PageData:
    page: int
    original_text: str
    processed_text: str


@dataclass
class DocumentData:
    document_name: str
    status: str
    language: Optional[str]
    pages: list[PageData]

    def to_dict(self):
        return {
            "document_name": self.document_name,
            "status": self.status,
            "language": self.language,
            "pages": [asdict(page) for page in self.pages]
        }
