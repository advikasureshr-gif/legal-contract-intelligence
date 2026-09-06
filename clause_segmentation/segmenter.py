"""
Clause Segmentation Module
===========================

Responsibility
--------------
Take a page-structured legal contract and split it into an ordered list
of clause objects.

This module is responsible ONLY for clause segmentation.

It does NOT:
- classify clauses
- calculate risk
- translate text
- summarize text
- modify legal meaning
- run transformer models

Public API
----------
    segment_contract(document: dict) -> dict

Expected input
--------------
{
    "document_name": "contract.pdf",
    "language": "en",
    "pages": [
        {
            "page": 1,
            "text": "..."
        }
    ]
}

Output
------
{
    "status": "success",
    "clauses": [
        {
            "clause_id": "1.1",
            "section": "ESTABLISHMENT OF DISTRIBUTORSHIP",
            "page": 1,
            "text": "...",
            "language": "en"
        }
    ]
}

Design
------
1. Structural regex detection
2. Rule-based candidate validation
3. Boundary resolution
4. Clause construction

The implementation deliberately avoids ML so that segmentation remains
deterministic and explainable.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ============================================================================
# Constants
# ============================================================================

ERROR_CODE = "CLAUSE_SEGMENTATION_FAILED"


# ============================================================================
# Boundary types
# ============================================================================

class BoundaryType(Enum):
    ARTICLE = "article"
    SECTION = "section"
    NUMERIC = "numeric"
    RECITAL = "recital"
    ALPHA_UPPER_LIST = "alpha_upper_list"
    ALPHA_LOWER_LIST = "alpha_lower_list"
    ROMAN_LIST = "roman_list"
    HEADING = "heading"


class Priority(Enum):
    STRONG = 3
    MEDIUM = 2
    WEAK = 1


# ============================================================================
# Regex patterns
# ============================================================================

PAGE_MARKER_RE = re.compile(
    r"^(?:page\s*-?\s*\d+\s*-?|"
    r"-{1,3}\s*\d+\s*-{1,3})$",
    re.IGNORECASE,
)


ARTICLE_RE = re.compile(
    r"^ARTICLE\s+"
    r"([IVXLCDM]+|\d+)"
    r"(?:\s*[\.:\-]\s*)?"
    r"(.*)$",
    re.IGNORECASE,
)


SECTION_RE = re.compile(
    r"^SECTION\s+"
    r"([A-Za-z0-9IVXLCDM]+(?:\.[A-Za-z0-9IVXLCDM]+)*)"
    r"(?:\s*[\.:\-]\s*)?"
    r"(.*)$",
    re.IGNORECASE,
)


# Numeric clause candidate.
#
# Examples accepted:
#   1. ESTABLISHMENT
#   1.1 Grant and Acceptance
#   2 TERM
#   5.2 Termination Rights
#
# Validation happens later. Regex detection alone does NOT mean boundary.
NUMERIC_RE = re.compile(
    r"^(\d{1,3}(?:\.\d{1,3}){0,5})"
    r"\.?"
    r"\s+"
    r"(\S.*)$"
)


RECITAL_RE = re.compile(
    r"^([A-Z])\.\s+(\S.*)$"
)


PAREN_RE = re.compile(
    r"^\(([A-Za-z]{1,6}|[ivxlcdmIVXLCDM]{1,8})\)"
    r"\s*(.*)$"
)


# Strong heading candidates.
ALL_CAPS_HEADING_RE = re.compile(
    r"^[A-Z][A-Z0-9][A-Z0-9\s,&'()/\-:]+$"
)


# ============================================================================
# Heuristic vocabulary
# ============================================================================

# These are common units/measurement words. They help reject cases such as:
#   5.2 years
#   10.5 miles
#   2.5 million
#
# This is intentionally conservative.
MEASUREMENT_WORDS = {
    "year",
    "years",
    "month",
    "months",
    "week",
    "weeks",
    "day",
    "days",
    "hour",
    "hours",
    "minute",
    "minutes",
    "second",
    "seconds",
    "mile",
    "miles",
    "meter",
    "meters",
    "metre",
    "metres",
    "km",
    "kms",
    "kilometer",
    "kilometers",
    "kilometre",
    "kilometres",
    "ft",
    "feet",
    "foot",
    "inch",
    "inches",
    "percent",
    "percentage",
    "million",
    "billion",
    "thousand",
    "dollars",
    "dollar",
    "usd",
    "rupees",
    "rupee",
    "kg",
    "kgs",
    "gram",
    "grams",
    "mg",
    "litre",
    "litres",
    "liter",
    "liters",
}


# Common address/location indicators.
ADDRESS_WORDS = {
    "street",
    "st.",
    "road",
    "rd.",
    "avenue",
    "ave.",
    "boulevard",
    "blvd.",
    "lane",
    "ln.",
    "drive",
    "dr.",
    "highway",
    "hwy.",
    "parkway",
    "pkwy.",
    "court",
    "ct.",
    "place",
    "pl.",
}


# Common legal words. These are useful when a numeric candidate starts
# with a lowercase word or an otherwise ambiguous token.
LEGAL_START_WORDS = {
    "agreement",
    "acceptance",
    "assignment",
    "confidentiality",
    "company",
    "distributor",
    "distributors",
    "either",
    "employee",
    "employer",
    "grant",
    "insurance",
    "license",
    "licence",
    "liability",
    "payment",
    "party",
    "parties",
    "price",
    "products",
    "renewal",
    "representations",
    "rights",
    "services",
    "termination",
    "term",
    "warranty",
    "whereas",
}


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class _Line:
    """A source line together with its page and position metadata."""

    text: str
    page: int
    index: int


@dataclass
class _Marker:
    """A structural marker candidate."""

    boundary_type: BoundaryType
    clause_id: str
    lead_text: str
    priority: Priority
    raw_text: str


@dataclass
class Clause:
    """
    Internal clause representation.

    text_parts contains paragraphs/lines after the structural identifier.
    """

    clause_id: str
    section: str
    page: int
    language: str
    text_parts: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clause_id": self.clause_id,
            "section": self.section,
            "page": self.page,
            "text": "\n\n".join(
                part for part in self.text_parts if part.strip()
            ),
            "language": self.language,
        }


# ============================================================================
# Basic utilities
# ============================================================================

def _normalize_newlines(text: str) -> str:
    """Normalize newline representation without changing content otherwise."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _is_page_marker(text: str) -> bool:
    return bool(PAGE_MARKER_RE.fullmatch(text.strip()))


def _is_blank(text: str) -> bool:
    return not text.strip()


def _first_word(text: str) -> str:
    match = re.match(r"([A-Za-z][A-Za-z0-9'.-]*)", text.strip())
    return match.group(1).lower() if match else ""


def _is_roman(value: str) -> bool:
    return bool(
        re.fullmatch(
            r"(?i)(?=[MDCLXVI]+$)M{0,4}"
            r"(?:C[MD]|D?C{0,3})?"
            r"(?:X[CL]|L?X{0,3})?"
            r"(?:I[XV]|V?I{0,3})?",
            value,
        )
    )


def _numeric_depth(clause_id: str) -> int:
    return clause_id.count(".") + 1


def _numeric_parent(clause_id: str) -> Optional[str]:
    if "." not in clause_id:
        return None
    return clause_id.rsplit(".", 1)[0]


def _is_top_level_numeric(clause_id: str) -> bool:
    return _numeric_depth(clause_id) == 1


def _has_nested_numeric_prefix(clause_id: str) -> bool:
    return "." in clause_id


# ============================================================================
# Structural detection
# ============================================================================

def _classify_paren_marker(text: str) -> Optional[_Marker]:
    """
    Detect parenthesized list items.

    These are deliberately NOT top-level clauses.
    They are retained inside the current clause.
    """
    match = PAREN_RE.match(text.strip())

    if not match:
        return None

    value = match.group(1)
    lead = match.group(2).strip()

    if value.isalpha() and value.isupper():
        boundary_type = BoundaryType.ALPHA_UPPER_LIST
    elif value.isalpha() and value.islower():
        boundary_type = BoundaryType.ALPHA_LOWER_LIST
    elif _is_roman(value):
        boundary_type = BoundaryType.ROMAN_LIST
    else:
        return None

    return _Marker(
        boundary_type=boundary_type,
        clause_id=f"({value})",
        lead_text=lead,
        priority=Priority.WEAK,
        raw_text=text,
    )


def _classify_marker(text: str) -> Optional[_Marker]:
    """
    Detect structural candidates.

    IMPORTANT:
    Detection is not acceptance. The segmentation engine validates
    candidates before treating them as boundaries.
    """
    stripped = text.strip()

    # ARTICLE IV ...
    match = ARTICLE_RE.fullmatch(stripped)
    if match:
        number = match.group(1)
        lead = match.group(2).strip()

        return _Marker(
            boundary_type=BoundaryType.ARTICLE,
            clause_id=f"ARTICLE {number.upper()}",
            lead_text=lead,
            priority=Priority.STRONG,
            raw_text=text,
        )

    # SECTION 3 ...
    match = SECTION_RE.fullmatch(stripped)
    if match:
        number = match.group(1)
        lead = match.group(2).strip()

        return _Marker(
            boundary_type=BoundaryType.SECTION,
            clause_id=f"SECTION {number}",
            lead_text=lead,
            priority=Priority.STRONG,
            raw_text=text,
        )

    # Numeric candidates.
    match = NUMERIC_RE.fullmatch(stripped)
    if match:
        clause_id = match.group(1)
        lead = match.group(2).strip()

        priority = (
            Priority.STRONG
            if _has_nested_numeric_prefix(clause_id)
            else Priority.MEDIUM
        )

        return _Marker(
            boundary_type=BoundaryType.NUMERIC,
            clause_id=clause_id,
            lead_text=lead,
            priority=priority,
            raw_text=text,
        )

    # Recitals A. B. C. are considered only by contextual validation.
    match = RECITAL_RE.fullmatch(stripped)
    if match:
        letter = match.group(1)
        lead = match.group(2).strip()

        return _Marker(
            boundary_type=BoundaryType.RECITAL,
            clause_id=letter,
            lead_text=lead,
            priority=Priority.MEDIUM,
            raw_text=text,
        )

    # Parenthesized list.
    return _classify_paren_marker(stripped)


# ============================================================================
# Heading detection
# ============================================================================

def _looks_like_heading(
    text: str,
    previous_blank: bool,
    next_blank: bool,
) -> bool:
    """
    Conservative heading detector.

    A heading should normally be:
    - standalone
    - surrounded by paragraph boundaries
    - reasonably short
    - not a normal sentence
    - not obviously a list or numeric expression
    """

    stripped = text.strip()

    if not stripped:
        return False

    if not (previous_blank and next_blank):
        return False

    if len(stripped) > 120:
        return False

    if NUMERIC_RE.fullmatch(stripped):
        return False

    if ARTICLE_RE.fullmatch(stripped):
        return False

    if SECTION_RE.fullmatch(stripped):
        return False

    if RECITAL_RE.fullmatch(stripped):
        return False

    if PAREN_RE.fullmatch(stripped):
        return False

    # Sentence punctuation strongly suggests ordinary prose.
    if stripped.endswith((".", "?", "!", ";")):
        return False

    words = stripped.split()

    if len(words) > 12:
        return False

    # Strongest signal: all caps.
    if ALL_CAPS_HEADING_RE.fullmatch(stripped):
        return True

    # Title Case headings such as:
    #   Term and Expiration
    #   Governing Law
    #   Intellectual Property
    if len(words) >= 2:
        alpha_words = [
            word for word in words
            if any(character.isalpha() for character in word)
        ]

        if not alpha_words:
            return False

        title_like = 0

        for word in alpha_words:
            cleaned = word.strip("\"'`()[]{}:,-")
            if not cleaned:
                continue

            if cleaned[0].isupper():
                title_like += 1

        # Avoid treating ordinary "The Company" prose as a heading.
        if title_like == len(alpha_words) and len(alpha_words) >= 2:
            return True

    return False


# ============================================================================
# Numeric validation
# ============================================================================

def _looks_like_measurement_or_address(lead_text: str) -> bool:
    """
    Reject numeric expressions which happen to look like legal numbering.

    Examples:
        5.2 years
        10.5 Main Street
        2.5 million dollars
    """
    stripped = lead_text.strip()

    if not stripped:
        return True

    first = _first_word(stripped)

    if first in MEASUREMENT_WORDS:
        return True

    # Common numeric address pattern:
    # 10.5 Main Street
    words = stripped.split()

    if len(words) >= 2:
        second_word = words[1].lower().strip(".,;:")

        if second_word in ADDRESS_WORDS:
            return True

    return False


def _looks_like_sentence_fragment(lead_text: str) -> bool:
    """
    Detect obvious ordinary prose fragments.

    This is deliberately conservative because legal clauses can legitimately
    begin with lowercase words after formatting loss.
    """
    stripped = lead_text.strip()

    if not stripped:
        return True

    first = _first_word(stripped)

    # A measurement/address is not a clause title.
    if _looks_like_measurement_or_address(stripped):
        return True

    # Obvious ordinary-language starts.
    if first in {
        "and",
        "but",
        "or",
        "because",
        "although",
        "if",
        "when",
        "while",
        "which",
        "that",
        "than",
        "as",
    }:
        return True

    return False


def _validate_numeric_candidate(
    marker: _Marker,
    previous_numeric_id: Optional[str],
    has_existing_clause: bool,
) -> bool:
    """
    Validate a numeric clause candidate.

    Rules:
    1. Reject empty identifiers/text.
    2. Reject obvious measurements and addresses.
    3. Strongly prefer legal/title-like candidates.
    4. Accept nested numbering when it has structural context.
    5. Accept sequential top-level numbering.
    6. Avoid blindly accepting every decimal-looking line.
    """

    clause_id = marker.clause_id
    lead = marker.lead_text.strip()

    if not clause_id or not lead:
        return False

    if _looks_like_measurement_or_address(lead):
        return False

    if _looks_like_sentence_fragment(lead):
        return False

    first = _first_word(lead)

    # Nested numbering is a very strong legal-document signal.
    if _has_nested_numeric_prefix(clause_id):
        # If we have an existing parent, this is especially reliable.
        if previous_numeric_id:
            parent = _numeric_parent(clause_id)

            if parent == previous_numeric_id:
                return True

        # Still accept nested numbering when the text strongly looks like
        # a heading/title.
        if lead[0].isupper():
            return True

        if first in LEGAL_START_WORDS:
            return True

        return False

    # Top-level numbers.
    #
    # If the text begins with a conventional legal/title word, accept.
    if first in LEGAL_START_WORDS:
        return True

    # Capitalized title-like text is usually a clause heading.
    if lead[0].isupper():
        return True

    # If this is clearly part of an established numbered sequence, accept.
    if previous_numeric_id:
        try:
            previous = int(previous_numeric_id.split(".")[0])
            current = int(clause_id)

            if current == previous + 1:
                return True
        except ValueError:
            pass

    # Without evidence, reject.
    return False


def _validate_marker(
    marker: _Marker,
    previous_numeric_id: Optional[str],
    has_existing_clause: bool,
    preamble: bool,
) -> bool:
    """
    Rule-based acceptance layer.

    This is intentionally separate from regex detection.
    """

    if marker.boundary_type == BoundaryType.NUMERIC:
        return _validate_numeric_candidate(
            marker,
            previous_numeric_id,
            has_existing_clause,
        )

    if marker.boundary_type in {
        BoundaryType.ARTICLE,
        BoundaryType.SECTION,
    }:
        return bool(marker.clause_id)

    if marker.boundary_type == BoundaryType.RECITAL:
        # Multiple recital clauses may occur consecutively in the preamble.
        return preamble

    if marker.boundary_type in {
        BoundaryType.ALPHA_UPPER_LIST,
        BoundaryType.ALPHA_LOWER_LIST,
        BoundaryType.ROMAN_LIST,
    }:
        # Lists are never top-level boundaries.
        return False

    return marker.boundary_type == BoundaryType.HEADING


# ============================================================================
# Segmentation engine
# ============================================================================

class _SegmentationEngine:
    """Deterministic rule-based clause segmentation engine."""

    def __init__(self, language: str):
        self.language = language

        self.clauses: List[Clause] = []

        self.current_section = ""

        self.current_clause: Optional[Clause] = None

        # Current paragraph buffer.
        self.buffer: List[str] = []

        self.buffer_page: Optional[int] = None

        # Used to decide whether a numeric container such as
        # "1. ESTABLISHMENT OF DISTRIBUTORSHIP"
        # should become section context when 1.1 follows it.
        self.pending_container: Optional[Clause] = None

        self.previous_numeric_id: Optional[str] = None

        self.seen_real_clause = False

    # ------------------------------------------------------------------
    # Buffer handling
    # ------------------------------------------------------------------

    def _begin_buffer(self, page: int) -> None:
        if self.buffer_page is None:
            self.buffer_page = page

    def _append_line(self, text: str, page: int) -> None:
        """
        Append a source line while retaining line structure.

        Detection uses stripped text, but stored content does not get
        collapsed into spaces.
        """
        if self.buffer_page is None:
            self.buffer_page = page

        self.buffer.append(text.rstrip())

    def _flush_buffer(self) -> None:
        if not self.buffer:
            return

        # Remove only truly empty edges.
        while self.buffer and not self.buffer[0].strip():
            self.buffer.pop(0)

        while self.buffer and not self.buffer[-1].strip():
            self.buffer.pop()

        if not self.buffer:
            self.buffer_page = None
            return

        paragraph = "\n".join(self.buffer)

        if self.current_clause is None:
            # Unnumbered prose becomes a deterministic fallback clause.
            clause_id = f"UNSPECIFIED-{len(self.clauses) + 1}"

            clause = Clause(
                clause_id=clause_id,
                section=self.current_section,
                page=self.buffer_page or 1,
                language=self.language,
                text_parts=[paragraph],
            )

            self.clauses.append(clause)
            self.current_clause = clause
            self.seen_real_clause = True

        else:
            self.current_clause.text_parts.append(paragraph)

        self.buffer.clear()
        self.buffer_page = None

    # ------------------------------------------------------------------
    # Clause creation
    # ------------------------------------------------------------------

    def _create_clause(
        self,
        marker: _Marker,
        page: int,
    ) -> None:
        """
        Create a new clause from an accepted structural marker.
        """

        self._flush_buffer()

        clause = Clause(
            clause_id=marker.clause_id,
            section=self.current_section,
            page=page,
            language=self.language,
            text_parts=[],
        )

        if marker.lead_text:
            clause.text_parts.append(marker.lead_text)

        self.clauses.append(clause)
        self.current_clause = clause
        self.seen_real_clause = True

        if marker.boundary_type == BoundaryType.NUMERIC:
            self.previous_numeric_id = marker.clause_id

        # A top-level numeric heading may later become section context.
        if (
            marker.boundary_type == BoundaryType.NUMERIC
            and _is_top_level_numeric(marker.clause_id)
        ):
            self.pending_container = clause
        else:
            self.pending_container = None

    # ------------------------------------------------------------------
    # Section handling
    # ------------------------------------------------------------------

    def _apply_heading(self, text: str) -> None:
        """
        Set section context from a standalone heading.

        The heading itself is not emitted as a clause.
        """

        self._flush_buffer()

        self.current_section = text.strip()

        self.current_clause = None
        self.pending_container = None

    def _resolve_pending_container(
        self,
        next_marker: _Marker,
    ) -> None:
        """
        Convert a top-level numeric container into section context when its
        child clause immediately follows.

        Example:

            1. ESTABLISHMENT OF DISTRIBUTORSHIP
            1.1 Grant and Acceptance

        becomes:

            section = "ESTABLISHMENT OF DISTRIBUTORSHIP"
            clause_id = "1.1"

        If the parent has substantive body text, it remains a clause.
        """

        if self.pending_container is None:
            return

        if next_marker.boundary_type != BoundaryType.NUMERIC:
            return

        parent_id = _numeric_parent(next_marker.clause_id)

        if parent_id != self.pending_container.clause_id:
            self.pending_container = None
            return

        parent = self.pending_container

        # A container with only its heading/title should become section
        # context rather than an independent clause.
        if len(parent.text_parts) == 1:
            title = parent.text_parts[0].strip()

            if title:
                self.current_section = title

            if parent in self.clauses:
                self.clauses.remove(parent)

        self.pending_container = None

    # ------------------------------------------------------------------
    # Marker processing
    # ------------------------------------------------------------------

    def _process_marker(
        self,
        marker: _Marker,
        page: int,
        preamble: bool,
    ) -> bool:
        """
        Process an accepted marker.

        Returns True when the marker was handled as a boundary.
        """

        # Parenthesized lists stay inside the current clause.
        if marker.boundary_type in {
            BoundaryType.ALPHA_UPPER_LIST,
            BoundaryType.ALPHA_LOWER_LIST,
            BoundaryType.ROMAN_LIST,
        }:
            return False

        if not _validate_marker(
            marker,
            self.previous_numeric_id,
            self.current_clause is not None,
            preamble,
        ):
            return False

        # Before creating a child numeric clause, resolve its parent.
        self._resolve_pending_container(marker)

        if marker.boundary_type == BoundaryType.HEADING:
            self._apply_heading(marker.lead_text)
            return True

        if marker.boundary_type == BoundaryType.ARTICLE:
            self.current_section = marker.clause_id

        elif marker.boundary_type == BoundaryType.SECTION:
            self.current_section = marker.clause_id

        self._create_clause(marker, page)

        return True

    # ------------------------------------------------------------------
    # Lines and paragraphs
    # ------------------------------------------------------------------

    def process_page(self, page_number: int, text: str) -> None:
        """
        Process one page without treating the page boundary as a clause
        boundary.
        """

        normalized = _normalize_newlines(text)
        lines = normalized.split("\n")

        for position, raw_line in enumerate(lines):
            stripped = raw_line.strip()

            if _is_page_marker(stripped):
                continue

            previous_blank = (
                position == 0
                or _is_blank(lines[position - 1])
            )

            next_blank = (
                position == len(lines) - 1
                or _is_blank(lines[position + 1])
            )

            # Blank line closes the current paragraph but NOT the clause.
            if _is_blank(raw_line):
                self._flush_buffer()
                continue

            marker = _classify_marker(stripped)

            # Parenthesized lists should be appended exactly as part of the
            # current clause.
            if marker and marker.boundary_type in {
                BoundaryType.ALPHA_UPPER_LIST,
                BoundaryType.ALPHA_LOWER_LIST,
                BoundaryType.ROMAN_LIST,
            }:
                if self.current_clause is not None:
                    self._append_line(raw_line, page_number)
                else:
                    self._append_line(raw_line, page_number)
                continue

            # Standalone heading.
            if _looks_like_heading(
                stripped,
                previous_blank,
                next_blank,
            ):
                heading_marker = _Marker(
                    boundary_type=BoundaryType.HEADING,
                    clause_id="HEADING",
                    lead_text=stripped,
                    priority=Priority.MEDIUM,
                    raw_text=raw_line,
                )

                if self._process_marker(
                    heading_marker,
                    page_number,
                    preamble=(
                        not self.seen_real_clause
                        or (
                            marker is not None
                            and marker.boundary_type == BoundaryType.RECITAL
                        )
                    ),
                ):
                    continue

            # Structural marker candidate.
            if marker:
                if self._process_marker(
                    marker,
                    page_number,
                    preamble=(
                        not self.seen_real_clause
                        or (
                            marker is not None
                            and marker.boundary_type == BoundaryType.RECITAL
                        )
                    ),
                ):
                    continue

            # Ordinary continuation/content line.
            self._append_line(raw_line, page_number)

    def finalize(self) -> List[Dict[str, Any]]:
        self._flush_buffer()

        # Clean empty clauses.
        cleaned: List[Clause] = []

        for clause in self.clauses:
            clause.text_parts = [
                part for part in clause.text_parts
                if part.strip()
            ]

            if clause.text_parts:
                cleaned.append(clause)

        self.clauses = cleaned

        # A pending container with no child should remain a legitimate clause.
        return [clause.to_dict() for clause in self.clauses]


# ============================================================================
# Input validation
# ============================================================================

def _validate_document(document: Any) -> Tuple[bool, str]:
    if not isinstance(document, dict):
        return False, "Input document must be a JSON object."

    if "language" not in document:
        return False, "Missing required field: language."

    if not isinstance(document["language"], str):
        return False, "Field 'language' must be a string."

    if "pages" not in document:
        return False, "Missing required field: pages."

    if not isinstance(document["pages"], list):
        return False, "Field 'pages' must be a list."

    for index, page in enumerate(document["pages"]):
        if not isinstance(page, dict):
            return False, f"Page {index} must be an object."

        if "page" not in page:
            return False, f"Page {index} is missing field 'page'."

        if "text" not in page:
            return False, f"Page {index} is missing field 'text'."

        if not isinstance(page["text"], str):
            return False, f"Page {index} field 'text' must be a string."

        if not isinstance(page["page"], int):
            return False, f"Page {index} field 'page' must be an integer."

    return True, ""


# ============================================================================
# Output validation
# ============================================================================

def _validate_output(result: Dict[str, Any]) -> None:
    if result.get("status") != "success":
        return

    clauses = result.get("clauses")

    if not isinstance(clauses, list):
        raise ValueError("Output 'clauses' must be a list.")

    seen_ids = set()

    for clause in clauses:
        required = {
            "clause_id",
            "section",
            "page",
            "text",
            "language",
        }

        if set(clause.keys()) != required:
            raise ValueError(
                "Clause schema mismatch. "
                f"Expected {required}, got {set(clause.keys())}."
            )

        if not isinstance(clause["clause_id"], str):
            raise ValueError("clause_id must be a string.")

        if clause["clause_id"] in seen_ids:
            raise ValueError(
                f"Duplicate clause_id detected: {clause['clause_id']}"
            )

        seen_ids.add(clause["clause_id"])

        if not isinstance(clause["section"], str):
            raise ValueError("section must be a string.")

        if not isinstance(clause["page"], int):
            raise ValueError("page must be an integer.")

        if not isinstance(clause["text"], str):
            raise ValueError("text must be a string.")

        if not isinstance(clause["language"], str):
            raise ValueError("language must be a string.")


# ============================================================================
# Public API
# ============================================================================

def segment_contract(document: Dict[str, Any]) -> Dict[str, Any]:
    """
    Segment a contract into clause objects.

    The function never raises ordinary processing errors to the caller.
    Failures are represented using the project's standard error schema.
    """

    try:
        valid, error_message = _validate_document(document)

        if not valid:
            return {
                "status": "error",
                "error_code": ERROR_CODE,
                "message": error_message,
            }

        language = document["language"]

        engine = _SegmentationEngine(language)

        # Process pages in supplied order.
        for page in document["pages"]:
            engine.process_page(
                page_number=page["page"],
                text=page["text"],
            )

        clauses = engine.finalize()

        result = {
            "status": "success",
            "clauses": clauses,
        }

        _validate_output(result)

        return result

    except Exception as exc:
        return {
            "status": "error",
            "error_code": ERROR_CODE,
            "message": str(exc),
        }


# ============================================================================
# CLI
# ============================================================================

def _main() -> int:
    """
    CLI usage:

        python segmenter.py sample_input.json
    """

    if len(sys.argv) != 2:
        print(
            "Usage: python segmenter.py <input.json>",
            file=sys.stderr,
        )
        return 1

    input_path = sys.argv[1]

    try:
        with open(input_path, "r", encoding="utf-8") as file:
            document = json.load(file)
    except FileNotFoundError:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": ERROR_CODE,
                    "message": f"Input file not found: {input_path}",
                },
                indent=2,
            )
        )
        return 1
    except json.JSONDecodeError as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": ERROR_CODE,
                    "message": f"Invalid JSON: {exc}",
                },
                indent=2,
            )
        )
        return 1

    result = segment_contract(document)

    print(json.dumps(result, indent=2, ensure_ascii=False))

    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(_main())