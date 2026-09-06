import re
import unicodedata


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode characters while preserving legal meaning.

    NFKC helps normalize visually equivalent characters such as
    full-width characters and compatibility forms.
    """
    return unicodedata.normalize("NFKC", text)


def remove_non_content_artifacts(text: str) -> str:
    """
    Remove common document-extraction artifacts such as page markers
    and isolated form-feed characters.

    Legal content is intentionally preserved.
    """

    # Remove common PDF page-break/form-feed artifacts
    text = text.replace("\f", "\n")

    # Remove common page-number artifacts:
    # Page -2-
    # Page 2
    # -2-
    text = re.sub(
        r"(?im)^\s*page\s*[-–—]?\s*\d+\s*[-–—]?\s*$",
        "",
        text
    )

    text = re.sub(
        r"(?m)^\s*[-–—]\s*\d+\s*[-–—]\s*$",
        "",
        text
    )

    return text


def remove_html_xml_tags(text: str) -> str:
    """
    Remove HTML/XML markup while preserving the text contained
    inside the tags.
    """

    return re.sub(r"<[^>]+>", "", text)


def remove_ocr_garbage(text: str) -> str:
    """
    Remove obvious OCR/extraction garbage without aggressively
    changing legal text.

    This removes:
    - Unicode replacement characters
    - zero-width characters
    - repeated non-text garbage characters

    It intentionally does NOT attempt to correct words or numbers,
    because OCR correction can change legal meaning.
    """

    # Unicode replacement character often indicates failed decoding/OCR
    text = text.replace("\ufffd", "")

    # Remove zero-width / invisible formatting characters
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)

    # Remove isolated decorative OCR symbols at the beginning/end
    # of lines while keeping normal punctuation.
    text = re.sub(
        r"(?m)^\s*[|¦]{2,}\s*$",
        "",
        text
    )

    return text


def remove_control_characters(text: str) -> str:
    """
    Remove control characters while preserving meaningful
    whitespace such as spaces and newlines.
    """

    cleaned = []

    for char in text:
        category = unicodedata.category(char)

        if category.startswith("C"):
            # Keep newline and tab temporarily; tabs are normalized later.
            if char in ("\n", "\t"):
                cleaned.append(char)
        else:
            cleaned.append(char)

    return "".join(cleaned)


def normalize_whitespace(text: str) -> str:
    """
    Normalize spaces, tabs, line endings, and excessive blank lines.
    """

    # Normalize Windows / old Mac line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Convert tabs to spaces
    text = text.replace("\t", " ")

    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split("\n")]

    # Reconstruct text
    text = "\n".join(lines)

    # Collapse multiple spaces
    text = re.sub(r"[ ]{2,}", " ", text)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove leading/trailing whitespace from complete text
    return text.strip()


def clean_text(text: str) -> str:
    """
    Clean and normalize extracted document text.

    Processing stages:
    1. Unicode normalization
    2. Removal of non-content document artifacts
    3. HTML/XML tag removal
    4. Conservative OCR garbage cleanup
    5. Control-character removal
    6. Whitespace normalization

    The function is intentionally conservative because the input
    is legal contract text. Legal numbers, punctuation, dates,
    currency values, percentages, section numbers, and negations
    must be preserved.

    This function returns processed_text only.
    original_text is never modified by this function.
    """

    if not isinstance(text, str):
        raise TypeError("Text must be a string.")

    # 1. Unicode normalization
    text = normalize_unicode(text)

    # 2. Remove non-content artifacts
    text = remove_non_content_artifacts(text)

    # 3. Remove HTML/XML tags
    text = remove_html_xml_tags(text)

    # 4. Remove obvious OCR/extraction garbage
    text = remove_ocr_garbage(text)

    # 5. Remove control characters
    text = remove_control_characters(text)

    # 6. Normalize whitespace
    text = normalize_whitespace(text)

    return text
