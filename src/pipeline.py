import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from document_processing.processor import process_document
from clause_segmentation.segmenter import segment_contract
from risk.risk_engine import RiskEngine

def analyze_document(file_path: str) -> dict:
    """
    Run the complete legal contract analysis pipeline.

    Pipeline:
        Document
        -> preprocessing
        -> clause segmentation
        -> clause classification
        -> risk classification
    """

    # ---------------------------------------------------------
    # 1. DOCUMENT PROCESSING
    # ---------------------------------------------------------
    document = process_document(file_path)

    document_data = document.to_dict()

    # ---------------------------------------------------------
    # 2. CLAUSE SEGMENTATION
    # ---------------------------------------------------------
    segmentation_input = {
        "document_name": document.document_name,
        "language": document.language or "unknown",
        "pages": [
            {
                "page": page.page,
                "original_text": page.original_text,
                "processed_text": page.processed_text,
            }
            for page in document.pages
        ],
    }

    segmentation_result = segment_contract(segmentation_input)

    if segmentation_result.get("status") != "success":
        return {
            "status": "error",
            "stage": "clause_segmentation",
            "error": segmentation_result,
        }

    clauses = segmentation_result["clauses"]

    # ---------------------------------------------------------
    # 3. LOAD RISK ENGINE
    # ---------------------------------------------------------
    engine = RiskEngine()

    # ---------------------------------------------------------
    # 4. ANALYZE EVERY CLAUSE
    # ---------------------------------------------------------
    analyzed_clauses = []

    for clause in clauses:
        processed_text = clause["processed_text"]

        # The ML models operate on processed text.
        analysis = engine.analyze_clause(processed_text)

        # Preserve segmentation metadata.
        analyzed_clause = {
            "clause_id": clause["clause_id"],
            "section": clause["section"],
            "page": clause["page"],
            "language": clause["language"],
            "original_text": clause["original_text"],
            "processed_text": clause["processed_text"],
            "clause_type": analysis["clause_type"],
            "clause_confidence": analysis["clause_confidence"],
            "risk_label": analysis["risk_label"],
            "risk_probability": analysis["risk_probability"],
            "risk_threshold": analysis["risk_threshold"],
        }

        analyzed_clauses.append(analyzed_clause)

    # ---------------------------------------------------------
    # 5. FINAL OUTPUT
    # ---------------------------------------------------------
    return {
        "status": "success",
        "document_name": document.document_name,
        "language": document.language,
        "clauses": analyzed_clauses,
    }


def main():
    if len(sys.argv) != 2:
        print("Usage: python src/pipeline.py <document_path>")
        sys.exit(1)

    file_path = sys.argv[1]

    try:
        result = analyze_document(file_path)

        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False
            )
        )

    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": str(exc),
                },
                indent=2,
                ensure_ascii=False
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    main()