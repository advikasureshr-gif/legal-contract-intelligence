import re
import unicodedata


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode characters while preserving legal meaning.
    """
    return unicodedata.normalize("NFKC", text)


def remove_non_content_artifacts(text: str) -> str:
    """
    Remove common document-extraction artifacts while preserving
    line structure.

    Artifact lines are replaced with empty lines rather than
    deleting the line itself.
    """

    # Preserve form-feed as a line boundary
    text = text.replace("\f", "\n")

    # Remove common page-number artifacts while preserving
    # the newline at the end of the line.
    #
    # Examples:
    # Page -2-
    # Page 2
    text = re.sub(
        r"(?im)^[^\S\r\n]*page[^\S\r\n]*[-–—]?[^\S\r\n]*\d+[^\S\r\n]*[-–—]?[^\S\r\n]*(?=\n|$)",
        "",
        text
    )

    # Examples:
    # -2-
    # 2
    text = re.sub(
        r"(?m)^[^\S\r\n]*[-–—]?[^\S\r\n]*\d+[^\S\r\n]*[-–—]?[^\S\r\n]*(?=\n|$)",
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

    Line structure is preserved.
    """

    # Unicode replacement character
    text = text.replace("\ufffd", "")

    # Zero-width / invisible formatting characters
    text = re.sub(
        r"[\u200b\u200c\u200d\ufeff]",
        "",
        text
    )

    # Replace isolated decorative OCR-symbol lines with
    # empty lines rather than deleting the lines.
    text = re.sub(
        r"(?m)^[^\S\r\n]*[|¦]{2,}[^\S\r\n]*(?=\n|$)",
        "",
        text
    )

    return text


def remove_control_characters(text: str) -> str:
    """
    Remove control characters while preserving meaningful
    whitespace such as spaces, tabs, and newlines.
    """

    cleaned = []

    for char in text:
        category = unicodedata.category(char)

        if category.startswith("C"):
            if char in ("\n", "\t"):
                cleaned.append(char)
        else:
            cleaned.append(char)

    return "".join(cleaned)


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace while preserving line boundaries.

    The number of lines is intentionally preserved because the
    clause segmentation stage aligns processed lines with their
    corresponding original lines.
    """

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Convert tabs to spaces
    text = text.replace("\t", " ")

    # Strip whitespace from each line without removing lines
    lines = [line.strip() for line in text.split("\n")]

    # Collapse multiple spaces inside each line
    lines = [
        re.sub(r"[ ]{2,}", " ", line)
        for line in lines
    ]

    # Reconstruct without removing blank lines
    return "\n".join(lines)


def clean_text(text: str) -> str:
    """
    Clean and normalize extracted document text.

    The preprocessing pipeline is conservative because the input
    is legal contract text. Legal numbers, punctuation, dates,
    currency values, percentages, section numbers, and negations
    are preserved.

    Line boundaries are preserved so that original_text and
    processed_text remain aligned for clause segmentation.
    """

    if not isinstance(text, str):
        raise TypeError("Text must be a string.")

    # 1. Unicode normalization
    text = normalize_unicode(text)

    # 2. Remove document artifacts
    text = remove_non_content_artifacts(text)

    # 3. Remove HTML/XML tags
    text = remove_html_xml_tags(text)

    # 4. Remove obvious OCR garbage
    text = remove_ocr_garbage(text)

    # 5. Remove control characters
    text = remove_control_characters(text)

    # 6. Normalize whitespace
    text = normalize_whitespace(text)

    return text