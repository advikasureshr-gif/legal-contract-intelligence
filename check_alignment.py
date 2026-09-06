from document_processing.processor import extract_pdf

file_path = r"c:\Users\advik\Downloads\SampleContract-Shuttle.pdf"

pages = extract_pdf(file_path)

for page in pages:
    original_lines = len(page.original_text.split("\n"))
    processed_lines = len(page.processed_text.split("\n"))

    print(
        f"Page {page.page}: "
        f"original={original_lines}, "
        f"processed={processed_lines}"
    )

    print(
        f"  original newlines: {page.original_text.count(chr(10))}, "
        f"processed newlines: {page.processed_text.count(chr(10))}"
    )

    if original_lines != processed_lines:
        print("  >>> MISMATCH <<<")

print("\nDone.")