import os

import fitz  # PyMuPDF
from docx import Document
from langdetect import detect, LangDetectException

from .schemas import DocumentData, PageData
from .preprocessing import clean_text
from .errors import DocumentProcessingError


def process_document(file_path: str) -> DocumentData:
    """
    Main entry point for document processing.

    Supports PDF and DOCX files.

    Returns structured document data containing:
    - document name
    - detected language
    - page number
    - original extracted text
    - processed text
    """

    if not isinstance(file_path, str) or not file_path.strip():
        raise DocumentProcessingError(
            "INVALID_FILE_PATH",
            "A valid document file path is required."
        )

    if not os.path.exists(file_path):
        raise DocumentProcessingError(
            "FILE_NOT_FOUND",
            "The specified document could not be found."
        )

    extension = os.path.splitext(file_path)[1].lower()

    try:
        if extension == ".pdf":
            pages = extract_pdf(file_path)

        elif extension == ".docx":
            pages = extract_docx(file_path)

        else:
            raise DocumentProcessingError(
                "UNSUPPORTED_FILE_TYPE",
                "Only PDF and DOCX files are supported."
            )

    except DocumentProcessingError:
        raise

    except Exception as exc:
        raise DocumentProcessingError(
            "TEXT_EXTRACTION_FAILED",
            f"Unable to extract text from document: {exc}"
        )

    if not pages or not any(page.original_text.strip() for page in pages):
        raise DocumentProcessingError(
            "TEXT_EXTRACTION_FAILED",
            "No text could be extracted from the document."
        )

    # Detect language using all extracted original text
    full_text = "\n".join(
        page.original_text for page in pages
    )

    try:
        language = detect(full_text)
    except LangDetectException:
        language = "unknown"

    return DocumentData(
        document_name=os.path.basename(file_path),
        status="success",
        language=language,
        pages=pages
    )


def extract_pdf(file_path: str) -> list[PageData]:
    """
    Extract text page-by-page from a PDF.
    """

    pages = []

    with fitz.open(file_path) as document:
        for page_number, page in enumerate(document, start=1):
            original = page.get_text()

            pages.append(
                PageData(
                    page=page_number,
                    original_text=original,
                    processed_text=clean_text(original)
                )
            )

    return pages


def extract_docx(file_path: str) -> list[PageData]:
    """
    Extract text from a DOCX document.

    DOCX does not reliably expose physical page boundaries
    through python-docx, so the current version treats the
    document as a single logical page.
    """

    document = Document(file_path)

    text = "\n".join(
        paragraph.text
        for paragraph in document.paragraphs
    )

    return [
        PageData(
            page=1,
            original_text=text,
            processed_text=clean_text(text)
        )
    ]
