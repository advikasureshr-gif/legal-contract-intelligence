import unittest

from segmenter import segment_contract


def make_document(text, language="en", page=1):
    return {
        "document_name": "test_contract.txt",
        "language": language,
        "pages": [
            {
                "page": page,
                "text": text,
            }
        ],
    }


def clauses_of(result):
    assert result["status"] == "success"
    return result["clauses"]


def find_clause(result, clause_id):
    for clause in clauses_of(result):
        if clause["clause_id"] == clause_id:
            return clause
    return None


class ClauseSegmentationTests(unittest.TestCase):

    # ==================================================================
    # Basic numbering
    # ==================================================================

    def test_basic_numbered_clauses(self):
        document = make_document(
            "1. TERM\n\n"
            "This Agreement begins on the Effective Date.\n\n"
            "2. PAYMENT\n\n"
            "The Customer shall pay all invoices within thirty days."
        )

        result = segment_contract(document)
        clauses = clauses_of(result)

        self.assertEqual(len(clauses), 2)
        self.assertEqual(clauses[0]["clause_id"], "1")
        self.assertEqual(clauses[1]["clause_id"], "2")

        self.assertIn(
            "This Agreement begins on the Effective Date.",
            clauses[0]["text"],
        )

    # ==================================================================
    # Nested clauses
    # ==================================================================

    def test_nested_numbered_clauses(self):
        document = make_document(
            "1. ESTABLISHMENT\n\n"
            "1.1 Grant and Acceptance. The Company grants the Distributor "
            "the right to distribute the Products.\n\n"
            "1.2 License. The Company grants a license to use the trademarks."
        )

        result = segment_contract(document)
        clauses = clauses_of(result)

        self.assertEqual(len(clauses), 2)

        self.assertEqual(clauses[0]["clause_id"], "1.1")
        self.assertEqual(clauses[1]["clause_id"], "1.2")

        self.assertEqual(
            clauses[0]["section"],
            "ESTABLISHMENT",
        )

    def test_top_level_numeric_with_own_body_remains_clause(self):
        document = make_document(
            "1. TERM\n\n"
            "This is substantive text belonging directly to section one.\n\n"
            "1.1 Initial Term. The initial term is five years."
        )

        result = segment_contract(document)
        clauses = clauses_of(result)

        self.assertEqual(len(clauses), 2)

        self.assertEqual(clauses[0]["clause_id"], "1")
        self.assertIn(
            "substantive text",
            clauses[0]["text"],
        )

        self.assertEqual(clauses[1]["clause_id"], "1.1")

    # ==================================================================
    # Article / Section
    # ==================================================================

    def test_article_detection(self):
        document = make_document(
            "ARTICLE IV\n\n"
            "The parties agree to the following obligations."
        )

        result = segment_contract(document)
        clauses = clauses_of(result)

        self.assertEqual(len(clauses), 1)
        self.assertEqual(
            clauses[0]["clause_id"],
            "ARTICLE IV",
        )

    def test_section_detection(self):
        document = make_document(
            "SECTION 3\n\n"
            "The parties shall maintain confidentiality."
        )

        result = segment_contract(document)
        clauses = clauses_of(result)

        self.assertEqual(len(clauses), 1)
        self.assertEqual(
            clauses[0]["clause_id"],
            "SECTION 3",
        )

    # ==================================================================
    # Heading detection
    # ==================================================================

    def test_all_caps_heading(self):
        document = make_document(
            "CONFIDENTIALITY\n\n"
            "Each party shall protect confidential information."
        )

        result = segment_contract(document)
        clauses = clauses_of(result)

        self.assertEqual(len(clauses), 1)
        self.assertEqual(
            clauses[0]["section"],
            "CONFIDENTIALITY",
        )

        self.assertEqual(
            clauses[0]["clause_id"],
            "UNSPECIFIED-1",
        )

        self.assertIn(
            "Each party shall protect confidential information.",
            clauses[0]["text"],
        )

    def test_title_case_heading(self):
        document = make_document(
            "Governing Law\n\n"
            "This Agreement shall be governed by the laws of Delaware."
        )

        result = segment_contract(document)
        clauses = clauses_of(result)

        self.assertEqual(
            clauses[0]["section"],
            "Governing Law",
        )

    def test_ordinary_short_sentence_not_heading(self):
        document = make_document(
            "The Company agrees.\n\n"
            "The Distributor shall comply with all applicable laws."
        )

        result = segment_contract(document)
        clauses = clauses_of(result)

        self.assertEqual(len(clauses), 1)

        self.assertEqual(
            clauses[0]["section"],
            "",
        )

        self.assertIn(
            "The Company agrees.",
            clauses[0]["text"],
        )

    # ==================================================================
    # Numeric false positives
    # ==================================================================

    def test_decimal_at_line_start_is_not_clause(self):
        document = make_document(
            "The property is located at\n"
            "10.5 Main Street and is leased by the Company."
        )

        result = segment_contract(document)
        clauses = clauses_of(result)

        self.assertEqual(len(clauses), 1)

        self.assertEqual(
            clauses[0]["clause_id"],
            "UNSPECIFIED-1",
        )

        self.assertIn(
            "10.5 Main Street",
            clauses[0]["text"],
        )

    def test_measurement_at_line_start_is_not_clause(self):
        document = make_document(
            "The initial period is\n"
            "5.2 years unless extended by written agreement."
        )

        result = segment_contract(document)
        clauses = clauses_of(result)

        self.assertEqual(len(clauses), 1)

        self.assertIn(
            "5.2 years",
            clauses[0]["text"],
        )

    def test_real_nested_clause_is_detected(self):
        document = make_document(
            "5. GENERAL TERMS\n\n"
            "5.1 Notices. All notices shall be given in writing.\n\n"
            "5.2 Termination Rights. Either party may terminate this Agreement."
        )

        result = segment_contract(document)
        clauses = clauses_of(result)

        self.assertEqual(len(clauses), 2)

        self.assertEqual(
            clauses[0]["clause_id"],
            "5.1",
        )

        self.assertEqual(
            clauses[1]["clause_id"],
            "5.2",
        )

    # ==================================================================
    # Continuation paragraphs
    # ==================================================================

    def test_continuation_paragraphs_stay_in_same_clause(self):
        document = make_document(
            "1.1 Grant. The Company grants the Distributor the right to sell "
            "the Products.\n\n"
            "The Distributor shall use its best efforts to promote the "
            "Products throughout the Territory.\n\n"
            "The Distributor shall also maintain adequate records."
        )

        result = segment_contract(document)
        clauses = clauses_of(result)

        self.assertEqual(len(clauses), 1)

        text = clauses[0]["text"]

        self.assertIn(
            "The Distributor shall use its best efforts",
            text,
        )

        self.assertIn(
            "The Distributor shall also maintain adequate records.",
            text,
        )

    # ==================================================================
    # Page boundaries
    # ==================================================================

    def test_clause_can_span_pages(self):
        document = {
            "document_name": "contract.txt",
            "language": "en",
            "pages": [
                {
                    "page": 1,
                    "text": (
                        "1.1 Grant. The Company grants the Distributor "
                        "the exclusive right to distribute the Products."
                    ),
                },
                {
                    "page": 2,
                    "text": (
                        "The Distributor shall use best efforts to promote "
                        "the Products throughout the Territory."
                    ),
                },
            ],
        }

        result = segment_contract(document)
        clauses = clauses_of(result)

        self.assertEqual(len(clauses), 1)

        self.assertEqual(
            clauses[0]["page"],
            1,
        )

        self.assertIn(
            "The Distributor shall use best efforts",
            clauses[0]["text"],
        )

    def test_page_marker_is_removed(self):
        document = {
            "document_name": "contract.txt",
            "language": "en",
            "pages": [
                {
                    "page": 1,
                    "text": "1. TERM\n\nThis Agreement shall continue."
                },
                {
                    "page": 2,
                    "text": (
                        "Page -2-\n\n"
                        "The parties may renew the Agreement."
                    ),
                },
            ],
        }

        result = segment_contract(document)
        clauses = clauses_of(result)

        self.assertEqual(len(clauses), 1)

        self.assertNotIn(
            "Page -2-",
            clauses[0]["text"],
        )

        self.assertIn(
            "The parties may renew",
            clauses[0]["text"],
        )

    # ==================================================================
    # Lists
    # ==================================================================

    def test_parenthesized_list_items_stay_in_parent_clause(self):
        document = make_document(
            "1.6 Minimum Purchases. The Distributor shall satisfy "
            "the following requirements:\n\n"
            "(A) Purchase not less than the minimum quantity;\n"
            "(B) Submit quarterly forecasts;\n"
            "(C) Maintain adequate inventory levels."
        )

        result = segment_contract(document)
        clauses = clauses_of(result)

        self.assertEqual(len(clauses), 1)

        text = clauses[0]["text"]

        self.assertIn("(A)", text)
        self.assertIn("(B)", text)
        self.assertIn("(C)", text)

    # ==================================================================
    # Recitals
    # ==================================================================

    def test_recitals_are_detected_in_preamble(self):
        document = make_document(
            "A. The Company manufactures Products.\n\n"
            "B. The Distributor wishes to distribute the Products.\n\n"
            "1. TERM\n\n"
            "This Agreement shall remain in effect."
        )

        result = segment_contract(document)
        clauses = clauses_of(result)

        self.assertGreaterEqual(len(clauses), 3)

        self.assertEqual(
            clauses[0]["clause_id"],
            "A",
        )

        self.assertEqual(
            clauses[1]["clause_id"],
            "B",
        )

        self.assertEqual(
            clauses[2]["clause_id"],
            "1",
        )

    # ==================================================================
    # Section propagation
    # ==================================================================

    def test_section_propagation(self):
        document = make_document(
            "TERMINATION\n\n"
            "3.1 Notice. Either party may terminate upon sixty days notice."
        )

        result = segment_contract(document)
        clauses = clauses_of(result)

        self.assertEqual(
            clauses[0]["section"],
            "TERMINATION",
        )

    # ==================================================================
    # Language preservation
    # ==================================================================

    def test_language_is_preserved(self):
        document = make_document(
            "1. Term\n\n"
            "This Agreement remains effective.",
            language="ta",
        )

        result = segment_contract(document)
        clauses = clauses_of(result)

        self.assertEqual(
            clauses[0]["language"],
            "ta",
        )

    # ==================================================================
    # Text preservation
    # ==================================================================

    def test_clause_text_preserves_internal_line_structure(self):
        document = make_document(
            "1. TERM\n\n"
            "This Agreement begins on the Effective Date.\n"
            "It shall remain in force for five years."
        )

        result = segment_contract(document)
        clauses = clauses_of(result)

        text = clauses[0]["text"]

        self.assertIn(
            "This Agreement begins on the Effective Date.",
            text,
        )

        self.assertIn(
            "It shall remain in force for five years.",
            text,
        )

    # ==================================================================
    # Invalid input
    # ==================================================================

    def test_invalid_input_type(self):
        result = segment_contract("not a dictionary")

        self.assertEqual(
            result["status"],
            "error",
        )

        self.assertEqual(
            result["error_code"],
            "CLAUSE_SEGMENTATION_FAILED",
        )

    def test_missing_pages(self):
        result = segment_contract(
            {
                "language": "en",
            }
        )

        self.assertEqual(
            result["status"],
            "error",
        )

    def test_invalid_page_text(self):
        result = segment_contract(
            {
                "language": "en",
                "pages": [
                    {
                        "page": 1,
                        "text": 123,
                    }
                ],
            }
        )

        self.assertEqual(
            result["status"],
            "error",
        )

    # ==================================================================
    # Empty document
    # ==================================================================

    def test_empty_document(self):
        document = {
            "document_name": "empty.txt",
            "language": "en",
            "pages": [],
        }

        result = segment_contract(document)

        self.assertEqual(
            result["status"],
            "success",
        )

        self.assertEqual(
            result["clauses"],
            [],
        )

    # ==================================================================
    # Determinism
    # ==================================================================

    def test_deterministic_output(self):
        document = make_document(
            "1. TERM\n\n"
            "This Agreement remains effective.\n\n"
            "2. PAYMENT\n\n"
            "The Customer shall pay all invoices."
        )

        first = segment_contract(document)
        second = segment_contract(document)

        self.assertEqual(first, second)

    # ==================================================================
    # Ordering
    # ==================================================================

    def test_clause_order_is_preserved(self):
        document = make_document(
            "1. TERM\n\n"
            "The Agreement begins today.\n\n"
            "2. PAYMENT\n\n"
            "The Customer shall pay.\n\n"
            "3. TERMINATION\n\n"
            "Either party may terminate."
        )

        result = segment_contract(document)
        clauses = clauses_of(result)

        self.assertEqual(
            [c["clause_id"] for c in clauses],
            ["1", "2", "3"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)