import re


def clean_text(text: str) -> str:
    """
    Clean and normalize extracted document text.

    The function does not change the meaning of the text.
    It is intended to produce processed_text while
    original_text remains unchanged.
    """

    if not isinstance(text, str):
        raise TypeError("Text must be a string.")

    # Normalize Windows and old Mac line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Replace tabs with spaces
    text = text.replace("\t", " ")

    # Remove trailing spaces from each line
    lines = [line.rstrip() for line in text.split("\n")]

    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in lines]

    # Reconstruct the text
    text = "\n".join(lines)

    # Collapse multiple spaces into a single space
    text = re.sub(r"[ ]{2,}", " ", text)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove leading/trailing whitespace from the complete text
    text = text.strip()

    return text
