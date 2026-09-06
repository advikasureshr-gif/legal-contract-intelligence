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
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


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
    r"(?:\s*[.:\-]\s*)?"
    r"(.*)$",
    re.IGNORECASE,
)

SECTION_RE = re.compile(
    r"^SECTION\s+"
    r"([A-Za-z0-9IVXLCDM]+(?:\.[A-Za-z0-9IVXLCDM]+)*)"
    r"(?:\s*[.:\-]\s*)?"
    r"(.*)$",
    re.IGNORECASE,
)

NUMERIC_RE = re.compile(
    r"^(\d{1,3}(?:\.\d{1,3}){0,5})"
    r"\.?"
    r"\s+"
    r"(\S.*)$"
)

STANDALONE_NUMERIC_RE = re.compile(
    r"^(\d{1,3}(?:\.\d{1,3}){0,5})\.?$"
)

RECITAL_RE = re.compile(
    r"^([A-Z])\.\s+(\S.*)$"
)

PAREN_RE = re.compile(
    r"^\(([A-Za-z]{1,6}|[ivxlcdmIVXLCDM]{1,8})\)"
    r"\s*(.*)$"
)

ALL_CAPS_HEADING_RE = re.compile(
    r"^[A-Z][A-Z0-9][A-Z0-9\s,&'()/\-:]+$"
)


# ============================================================================
# Heuristic vocabularies
# ============================================================================

MEASUREMENT_WORDS = {
    "years",
    "year",
    "months",
    "month",
    "days",
    "day",
    "hours",
    "hour",
    "minutes",
    "minute",
    "percent",
    "percentage",
    "million",
    "billion",
    "thousand",
    "dollars",
    "dollar",
}


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


LEGAL_START_WORDS = {
    "agreement",
    "acceptance",
    "acknowledgment",
    "assignment",
    "confidentiality",
    "company",
    "commission",
    "consultant",
    "distributor",
    "distributors",
    "duties",
    "employee",
    "employer",
    "either",
    "grant",
    "insurance",
    "indemnification",
    "independent",
    "inspection",
    "license",
    "licence",
    "liability",
    "laws",
    "modification",
    "nonassignment",
    "notification",
    "payment",
    "party",
    "parties",
    "price",
    "products",
    "rebates",
    "records",
    "renewal",
    "representations",
    "retention",
    "rights",
    "safety",
    "services",
    "subcontracting",
    "term",
    "termination",
    "warranty",
    "work",
}


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class _Marker:
    boundary_type: BoundaryType
    clause_id: str
    lead_text: str
    priority: Priority
    raw_text: str


@dataclass
class Clause:
    clause_id: str
    section: str
    page: int
    language: str
    original_text_parts: List[str]
    processed_text_parts: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clause_id": self.clause_id,
            "section": self.section,
            "page": self.page,
            "original_text": "\n\n".join(
                part
                for part in self.original_text_parts
                if part.strip()
            ),
            "processed_text": "\n\n".join(
                part
                for part in self.processed_text_parts
                if part.strip()
            ),
            "language": self.language,
        }


# ============================================================================
# Basic utilities
# ============================================================================

def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _is_blank(text: str) -> bool:
    return not text.strip()


def _is_page_marker(text: str) -> bool:
    return bool(PAGE_MARKER_RE.fullmatch(text.strip()))


def _first_word(text: str) -> str:
    match = re.match(
        r"([A-Za-z][A-Za-z0-9'.-]*)",
        text.strip(),
    )

    if not match:
        return ""

    return match.group(1).lower()


def _is_roman(value: str) -> bool:
    return bool(
        re.fullmatch(
            r"(?=[MDCLXVI]+$)"
            r"M{0,4}"
            r"(?:C[MD]|D?C{0,3})?"
            r"(?:X[CL]|L?X{0,3})?"
            r"(?:I[XV]|V?I{0,3})?",
            value.upper(),
        )
    )


def _numeric_depth(clause_id: str) -> int:
    return clause_id.count(".") + 1


def _numeric_parent(clause_id: str) -> Optional[str]:
    if "." not in clause_id:
        return None

    return clause_id.rsplit(".", 1)[0]


def _top_level_number(clause_id: str) -> Optional[int]:
    try:
        return int(clause_id.split(".")[0])
    except (ValueError, IndexError):
        return None


def _numeric_components(
    clause_id: str,
) -> Optional[Tuple[int, ...]]:
    try:
        return tuple(
            int(part)
            for part in clause_id.split(".")
        )
    except ValueError:
        return None


def _is_top_level_numeric(clause_id: str) -> bool:
    return _numeric_depth(clause_id) == 1


def _is_standalone_numeric(text: str) -> bool:
    return bool(
        STANDALONE_NUMERIC_RE.fullmatch(
            text.strip()
        )
    )

def _is_standalone_numeric(text: str) -> bool:
    return bool(
        STANDALONE_NUMERIC_RE.fullmatch(
            text.strip()
        )
    )

def _looks_like_non_clause_block(text: str) -> bool:
    """
    Detect common administrative/signature content that should not
    be interpreted as a substantive contract clause.
    """
    normalized = re.sub(
        r"\s+",
        " ",
        text,
    ).strip().upper()

    administrative_phrases = (
        "APPROVED AS TO INSURANCE",
        "APPROVED AS TO FORM",
        "IN WITNESS WHEREOF",
        "DISTRIBUTION:",
        "SIGNATURE",
        "SIGNED",
        "PRINTED",
    )

    return any(
        phrase in normalized
        for phrase in administrative_phrases
    )

# ============================================================================
# Marker detection
# ============================================================================

def _classify_paren_marker(
    text: str,
) -> Optional[_Marker]:

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


def _classify_marker(
    text: str,
) -> Optional[_Marker]:

    stripped = text.strip()

    if not stripped:
        return None

    # ARTICLE
    match = ARTICLE_RE.fullmatch(stripped)

    if match:
        return _Marker(
            boundary_type=BoundaryType.ARTICLE,
            clause_id=match.group(1),
            lead_text=match.group(2).strip(),
            priority=Priority.STRONG,
            raw_text=text,
        )

    # SECTION
    match = SECTION_RE.fullmatch(stripped)

    if match:
        return _Marker(
            boundary_type=BoundaryType.SECTION,
            clause_id=match.group(1),
            lead_text=match.group(2).strip(),
            priority=Priority.STRONG,
            raw_text=text,
        )

    # Numeric
    match = NUMERIC_RE.fullmatch(stripped)

    if match:
        return _Marker(
            boundary_type=BoundaryType.NUMERIC,
            clause_id=match.group(1),
            lead_text=match.group(2).strip(),
            priority=Priority.STRONG,
            raw_text=text,
        )

    # Recital
    match = RECITAL_RE.fullmatch(stripped)

    if match:
        return _Marker(
            boundary_type=BoundaryType.RECITAL,
            clause_id=match.group(1),
            lead_text=match.group(2).strip(),
            priority=Priority.MEDIUM,
            raw_text=text,
        )

    # Parenthesized/list markers such as:
    # (A), (B), (1), 1), 2), etc.
    #
    # These are content inside a clause, not top-level
    # clause boundaries.
    paren_marker = _classify_paren_marker(stripped)

    if paren_marker is not None:
        return None

    return None


# ============================================================================
# Heading detection
# ============================================================================

def _looks_like_heading(
    text: str,
    previous_blank: bool,
    next_blank: bool,
) -> bool:

    stripped = text.strip()

    if not stripped:
        return False

    if not previous_blank or not next_blank:
        return False

    if len(stripped) > 120:
        return False

    # Structural markers take priority.
    if NUMERIC_RE.fullmatch(stripped):
        return False

    if STANDALONE_NUMERIC_RE.fullmatch(stripped):
        return False

    if ARTICLE_RE.fullmatch(stripped):
        return False

    if SECTION_RE.fullmatch(stripped):
        return False

    if RECITAL_RE.fullmatch(stripped):
        return False

    if PAREN_RE.fullmatch(stripped):
        return False

    # A heading should not look like ordinary prose.
    if stripped.endswith(
        (".", "?", "!", ";", ":")
    ):
        return False

    words = stripped.split()

    if not words or len(words) > 12:
        return False

    # All-caps heading.
    if ALL_CAPS_HEADING_RE.fullmatch(stripped):
        return True

    # Title-case heading.
    alpha_words = [
        word
        for word in words
        if any(character.isalpha() for character in word)
    ]

    if not alpha_words:
        return False

    title_like = 0

    for word in alpha_words:
        cleaned = word.strip(
            "\"'`()[]{}:,-"
        )

        if cleaned and cleaned[0].isupper():
            title_like += 1

    return (
        title_like == len(alpha_words)
        and len(alpha_words) >= 2
    )


# ============================================================================
# Candidate validation
# ============================================================================

def _looks_like_measurement_or_address(
    lead_text: str,
) -> bool:

    stripped = lead_text.strip()

    if not stripped:
        return True

    first = _first_word(stripped)

    if first in MEASUREMENT_WORDS:
        return True

    words = stripped.split()

    if len(words) >= 2:

        second = (
            words[1]
            .lower()
            .strip(".,;:")
        )

        if second in ADDRESS_WORDS:
            return True

    return False


def _looks_like_sentence_fragment(
    lead_text: str,
) -> bool:

    stripped = lead_text.strip()

    if not stripped:
        return True

    if _looks_like_measurement_or_address(
        stripped
    ):
        return True

    first = _first_word(stripped)

    return first in {
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
    }


def _is_plausible_clause_title(
    lead_text: str,
) -> bool:

    stripped = lead_text.strip()

    if not stripped:
        return False

    if _looks_like_measurement_or_address(
        stripped
    ):
        return False

    if _looks_like_sentence_fragment(
        stripped
    ):
        return False

    first = _first_word(stripped)

    if first in LEGAL_START_WORDS:
        return True

    # Legal headings normally begin with a capitalized word.
    if stripped[0].isupper():
        return True

    return False


def _validate_numeric_candidate(
    marker: _Marker,
    previous_numeric_id: Optional[str],
    existing_numeric_ids: List[str],
    has_existing_clause: bool,
) -> bool:

    clause_id = marker.clause_id
    lead = marker.lead_text.strip()

    if not clause_id or not lead:
        return False

    if _looks_like_measurement_or_address(
        lead
    ):
        return False

    if _looks_like_sentence_fragment(
        lead
    ):
        return False

    components = _numeric_components(
        clause_id
    )

    if components is None:
        return False

    # Never create the same numeric clause twice.
    # A repeated number may belong to a list, signature block,
    # exhibit, or other non-clause content.
    if clause_id in existing_numeric_ids:
        return False

    # --------------------------------------------------------------
    # Nested numbering
    # --------------------------------------------------------------

    if len(components) > 1:

        top_level = components[0]

        # If this is the first nested clause encountered,
        # require a plausible legal title.
        if not existing_numeric_ids:
            return _is_plausible_clause_title(
                lead
            )

        # Find the most recent numeric clause at the same
        # hierarchical level.
        same_level = [
            item
            for item in existing_numeric_ids
            if _numeric_depth(item)
            == len(components)
        ]

        if same_level:

            previous_same_level = (
                _numeric_components(
                    same_level[-1]
                )
            )

            if previous_same_level:

                # Example:
                # 1.1 -> 1.2 -> 1.3
                #
                # Same parent, increasing child.
                if (
                    previous_same_level[:-1]
                    == components[:-1]
                    and components[-1]
                    > previous_same_level[-1]
                ):
                    return True

        # Also accept a first child such as 1.1 when the
        # corresponding top-level clause exists.
        parent = _numeric_parent(
            clause_id
        )

        if parent in existing_numeric_ids:
            return _is_plausible_clause_title(
                lead
            )

        # More deeply nested clauses:
        #
        # 1.1.1 -> 1.1.2
        #
        # are accepted when they continue the same parent.
        if previous_numeric_id:

            previous_components = (
                _numeric_components(
                    previous_numeric_id
                )
            )

            if previous_components:

                if (
                    len(previous_components)
                    == len(components)
                    and previous_components[:-1]
                    == components[:-1]
                    and components[-1]
                    > previous_components[-1]
                ):
                    return True

        return _is_plausible_clause_title(
            lead
        )

    # --------------------------------------------------------------
    # Top-level numbering
    # --------------------------------------------------------------

    current_number = components[0]

    if first_word := _first_word(lead):

        if first_word in LEGAL_START_WORDS:
            return True

    # A top-level clause title normally begins with a capitalized word.
    if lead and lead[0].isupper():

        # Sequential numbering is particularly strong evidence.
        if previous_numeric_id:

            previous_number = _top_level_number(
                previous_numeric_id
            )

            if (
                previous_number is not None
                and current_number
                == previous_number + 1
            ):
                return True

        # If there are no existing clauses, a plausible title is enough.
        if not has_existing_clause:
            return _is_plausible_clause_title(
                lead
            )

        # A top-level title is still valid even when numbering skips.
        # Real contracts frequently omit or restructure numbering.
        return _is_plausible_clause_title(
            lead
        )

    return False


def _validate_marker(
    marker: _Marker,
    previous_numeric_id: Optional[str],
    existing_numeric_ids: List[str],
    has_existing_clause: bool,
    preamble: bool,
) -> bool:

    if marker.boundary_type == BoundaryType.NUMERIC:

        return _validate_numeric_candidate(
            marker,
            previous_numeric_id,
            existing_numeric_ids,
            has_existing_clause,
        )

    if marker.boundary_type in {
        BoundaryType.ARTICLE,
        BoundaryType.SECTION,
    }:
        return bool(marker.clause_id)

    if marker.boundary_type == BoundaryType.RECITAL:
        return preamble

    # Lists are never top-level boundaries.
    if marker.boundary_type in {
        BoundaryType.ALPHA_UPPER_LIST,
        BoundaryType.ALPHA_LOWER_LIST,
        BoundaryType.ROMAN_LIST,
    }:
        return False

    return marker.boundary_type == BoundaryType.HEADING


# ============================================================================
# Segmentation engine
# ============================================================================

class _SegmentationEngine:

    def __init__(self, language: str):

        self.language = language

        self.clauses: List[Clause] = []

        self.current_section = ""

        self.current_clause: Optional[Clause] = None

        self.original_buffer: List[str] = []
        self.processed_buffer: List[str] = []

        self.buffer_page: Optional[int] = None

        self.pending_container: Optional[Clause] = None

        self.previous_numeric_id: Optional[str] = None

        self.seen_real_clause = False

        # Keep track of accepted numeric IDs independently
        # from the current clause.
        self.numeric_ids: List[str] = []

    # ------------------------------------------------------------------
    # Buffer
    # ------------------------------------------------------------------

    def _append_line(
        self,
        original_text: str,
        processed_text: str,
        page: int,
    ) -> None:

        if self.buffer_page is None:
            self.buffer_page = page

        self.original_buffer.append(
            original_text.rstrip()
        )

        self.processed_buffer.append(
            processed_text.rstrip()
        )

    def _flush_buffer(self) -> None:

        if not self.processed_buffer:
            return

        while (
            self.processed_buffer
            and not self.processed_buffer[0].strip()
        ):
            self.processed_buffer.pop(0)
            self.original_buffer.pop(0)

        while (
            self.processed_buffer
            and not self.processed_buffer[-1].strip()
        ):
            self.processed_buffer.pop()
            self.original_buffer.pop()

        if not self.processed_buffer:

            self.buffer_page = None
            return

        original_text = "\n".join(
            self.original_buffer
        )

        processed_text = "\n".join(
            self.processed_buffer
        )

        if self.current_clause is None:

            clause_id = (
                f"UNSPECIFIED-{len(self.clauses) + 1}"
            )

            clause = Clause(
                clause_id=clause_id,
                section=self.current_section,
                page=self.buffer_page or 1,
                language=self.language,
                original_text_parts=[
                    original_text
                ],
                processed_text_parts=[
                    processed_text
                ],
            )

            self.clauses.append(clause)

            self.current_clause = clause

            self.seen_real_clause = True

        else:

            self.current_clause.original_text_parts.append(
                original_text
            )

            self.current_clause.processed_text_parts.append(
                processed_text
            )

        self.original_buffer.clear()
        self.processed_buffer.clear()
        self.buffer_page = None

    # ------------------------------------------------------------------
    # Clause creation
    # ------------------------------------------------------------------

    def _create_clause(
        self,
        marker: _Marker,
        page: int,
    ) -> None:

        self._flush_buffer()

        # Top-level numeric clauses such as "2. TERM" also
        # establish a new section context.
        section = self.current_section

        if (
            marker.boundary_type == BoundaryType.NUMERIC
            and _is_top_level_numeric(marker.clause_id)
        ):
            section = marker.raw_text.strip()

            # Remove the numeric prefix.
            section = re.sub(
                r"^\d+(?:\.\s*)?",
                "",
                section,
            ).strip()

            # If the heading and clause body are on the same line,
            # keep only the heading.
            if "." in section:
                section = section.split(".", 1)[0].strip()

        clause = Clause(
            clause_id=marker.clause_id,
            section=section,
            page=page,
            language=self.language,
            original_text_parts=[],
            processed_text_parts=[],
        )

        # Preserve the complete structural heading.
        if marker.raw_text.strip():

            clause.original_text_parts.append(
                marker.raw_text.strip()
            )

            clause.processed_text_parts.append(
                marker.raw_text.strip()
            )

        self.clauses.append(clause)

        self.current_clause = clause

        self.seen_real_clause = True

        if marker.boundary_type == BoundaryType.NUMERIC:

            self.previous_numeric_id = (
                marker.clause_id
            )

            if marker.clause_id not in self.numeric_ids:
                self.numeric_ids.append(
                    marker.clause_id
                )

            # A top-level numeric clause may become a
            # section/container if a nested child follows.
            if _is_top_level_numeric(
                marker.clause_id
            ):
                self.pending_container = clause

                # Keep the new top-level heading as the
                # active section for subsequent nested clauses.
                self.current_section = section

            else:
                self.pending_container = None

        else:

            self.pending_container = None

    # ------------------------------------------------------------------
    # Heading handling
    # ------------------------------------------------------------------

    def _apply_heading(
        self,
        text: str,
    ) -> None:

        self._flush_buffer()

        self.current_section = text.strip()

        self.current_clause = None

        self.pending_container = None

    # ------------------------------------------------------------------
    # Pending container
    # ------------------------------------------------------------------

    def _resolve_pending_container(
        self,
        next_marker: _Marker,
    ) -> None:

        if self.pending_container is None:
            return

        if next_marker.boundary_type != BoundaryType.NUMERIC:
            self.pending_container = None
            return

        parent_id = _numeric_parent(
            next_marker.clause_id
        )

        if (
            parent_id
            != self.pending_container.clause_id
        ):
            self.pending_container = None
            return

        parent = self.pending_container

        # A top-level numeric heading followed by a nested
        # child represents a section/container.
        if (
            len(parent.processed_text_parts)
            == 1
        ):

            heading = (
                parent.processed_text_parts[0]
                .strip()
            )

            match = NUMERIC_RE.fullmatch(
                heading
            )

            if match:
                heading = (
                    match.group(2)
                    .strip()
                )

            if heading:
                self.current_section = heading

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

        # Lists remain ordinary content.
        if marker.boundary_type in {
            BoundaryType.ALPHA_UPPER_LIST,
            BoundaryType.ALPHA_LOWER_LIST,
            BoundaryType.ROMAN_LIST,
        }:
            return False

        if not _validate_marker(
            marker,
            self.previous_numeric_id,
            self.numeric_ids,
            self.seen_real_clause,
            preamble,
        ):
            return False

        self._resolve_pending_container(
            marker
        )

        if marker.boundary_type == BoundaryType.HEADING:

            self._apply_heading(
                marker.lead_text
            )

            return True

        # ARTICLE / SECTION labels become section context.
        if marker.boundary_type == BoundaryType.ARTICLE:

            if marker.lead_text:
                self.current_section = (
                    marker.lead_text
                )

        elif marker.boundary_type == BoundaryType.SECTION:

            if marker.lead_text:
                self.current_section = (
                    marker.lead_text
                )

        self._create_clause(
            marker,
            page,
        )

        return True

    # ------------------------------------------------------------------
    # Page processing
    # ------------------------------------------------------------------

    def process_page(
        self,
        page_number: int,
        original_text: str,
        processed_text: str,
    ) -> None:

        original_lines = _normalize_newlines(
            original_text
        ).split("\n")

        processed_lines = _normalize_newlines(
            processed_text
        ).split("\n")

        if len(original_lines) != len(
            processed_lines
        ):
            raise ValueError(
                "Original and processed text must preserve "
                "line alignment."
            )

        position = 0
        total = len(processed_lines)

        while position < total:

            original_line = (
                original_lines[position]
            )

            processed_line = (
                processed_lines[position]
            )

            stripped = processed_line.strip()

            # ----------------------------------------------------------
            # Page marker
            # ----------------------------------------------------------

            if _is_page_marker(stripped):

                position += 1
                continue

            # ----------------------------------------------------------
            # Blank line
            # ----------------------------------------------------------

            if _is_blank(processed_line):

                self._flush_buffer()

                position += 1
                continue

            previous_blank = (
                position == 0
                or _is_blank(
                    processed_lines[position - 1]
                )
            )

            next_blank = (
                position == total - 1
                or _is_blank(
                    processed_lines[position + 1]
                )
            )

            # ----------------------------------------------------------
            # Parenthesized list
            # ----------------------------------------------------------

            marker = _classify_marker(
                stripped
            )

            if marker and marker.boundary_type in {
                BoundaryType.ALPHA_UPPER_LIST,
                BoundaryType.ALPHA_LOWER_LIST,
                BoundaryType.ROMAN_LIST,
            }:

                self._append_line(
                    original_line,
                    processed_line,
                    page_number,
                )

                position += 1
                continue

            # ----------------------------------------------------------
            # Standalone numeric marker
            # ----------------------------------------------------------

            if _is_standalone_numeric(
                stripped
            ):

                match = STANDALONE_NUMERIC_RE.fullmatch(
                    stripped
                )

                numeric_id = (
                    match.group(1)
                    if match
                    else ""
                )

                # A standalone number is only promoted if the
                # following line looks like a real legal heading.
                if position + 1 < total:

                    next_processed = (
                        processed_lines[
                            position + 1
                        ]
                    )

                    next_original = (
                        original_lines[
                            position + 1
                        ]
                    )

                    next_stripped = (
                        next_processed.strip()
                    )

                    if (
                        next_stripped
                        and not _is_page_marker(
                        next_stripped
                        )
                        and _looks_like_heading(
                            next_stripped,
                            previous_blank=True,
                            next_blank=(
                                position + 1 == total - 1
                                or _is_blank(
                                    processed_lines[position + 2]
                                )
                            ),
                        )
                    ):

                        combined = (
                            f"{numeric_id}. "
                            f"{next_stripped}"
                        )

                        combined_marker = (
                            _classify_marker(
                                combined
                            )
                        )

                        if combined_marker:

                            if self._process_marker(
                                combined_marker,
                                page_number,
                                preamble=(
                                    not self.seen_real_clause
                                ),
                            ):

                                # Replace the raw marker with
                                # the actual source lines.
                                if self.current_clause:

                                    raw_combined = (
                                        f"{original_line.strip()}\n"
                                        f"{next_original.strip()}"
                                    )

                                    self.current_clause.original_text_parts[
                                        -1
                                    ] = raw_combined

                                    self.current_clause.processed_text_parts[
                                        -1
                                    ] = combined

                                position += 2
                                continue

                # Otherwise preserve it as ordinary content.
                self._append_line(
                    original_line,
                    processed_line,
                    page_number,
                )

                position += 1
                continue

            # ----------------------------------------------------------
            # Standalone heading
            # ----------------------------------------------------------

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
                    raw_text=processed_line,
                )

                if self._process_marker(
                    heading_marker,
                    page_number,
                    preamble=(
                        not self.seen_real_clause
                    ),
                ):

                    position += 1
                    continue

            # ----------------------------------------------------------
            # Structural marker
            # ----------------------------------------------------------

            if marker:

                if self._process_marker(
                    marker,
                    page_number,
                    preamble=(
                        not self.seen_real_clause
                    ),
                ):

                    position += 1
                    continue

            # ----------------------------------------------------------
            # Ordinary content
            # ----------------------------------------------------------

            self._append_line(
                original_line,
                processed_line,
                page_number,
            )

            position += 1

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def finalize(
        self,
    ) -> List[Dict[str, Any]]:

        self._flush_buffer()

        cleaned: List[Clause] = []

        for clause in self.clauses:

            clause.original_text_parts = [
                part
                for part in clause.original_text_parts
                if part.strip()
            ]

            clause.processed_text_parts = [
                part
                for part in clause.processed_text_parts
                if part.strip()
            ]

            if clause.processed_text_parts:
                cleaned.append(clause)

        self.clauses = cleaned

        return [
            clause.to_dict()
            for clause in self.clauses
        ]


# ============================================================================
# Input validation
# ============================================================================

def _validate_document(
    document: Any,
) -> Tuple[bool, str]:

    if not isinstance(document, dict):
        return (
            False,
            "Input document must be a JSON object.",
        )

    if "language" not in document:
        return (
            False,
            "Missing required field: language.",
        )

    if not isinstance(
        document["language"],
        str,
    ):
        return (
            False,
            "Field 'language' must be a string.",
        )

    if "pages" not in document:
        return (
            False,
            "Missing required field: pages.",
        )

    if not isinstance(
        document["pages"],
        list,
    ):
        return (
            False,
            "Field 'pages' must be a list.",
        )

    for index, page in enumerate(
        document["pages"]
    ):

        if not isinstance(page, dict):
            return (
                False,
                f"Page {index} must be an object.",
            )

        required_fields = {
            "page",
            "original_text",
            "processed_text",
        }

        missing = (
            required_fields
            - set(page.keys())
        )

        if missing:
            return (
                False,
                f"Page {index} missing fields: "
                f"{sorted(missing)}",
            )

        if not isinstance(
            page["page"],
            int,
        ):
            return (
                False,
                f"Page {index} field 'page' "
                "must be an integer.",
            )

        if not isinstance(
            page["original_text"],
            str,
        ):
            return (
                False,
                f"Page {index} field "
                "'original_text' must be a string.",
            )

        if not isinstance(
            page["processed_text"],
            str,
        ):
            return (
                False,
                f"Page {index} field "
                "'processed_text' must be a string.",
            )

    return True, ""


# ============================================================================
# Output validation
# ============================================================================

def _validate_output(
    result: Dict[str, Any],
) -> None:

    if result.get("status") != "success":
        return

    clauses = result.get("clauses")

    if not isinstance(
        clauses,
        list,
    ):
        raise ValueError(
            "Output 'clauses' must be a list."
        )

    required = {
        "clause_id",
        "section",
        "page",
        "original_text",
        "processed_text",
        "language",
    }

    seen_ids = set()

    for clause in clauses:

        if set(clause.keys()) != required:
            raise ValueError(
                "Clause schema mismatch. "
                f"Expected {required}, "
                f"got {set(clause.keys())}."
            )

        clause_id = clause["clause_id"]

        if not isinstance(
            clause_id,
            str,
        ):
            raise ValueError(
                "clause_id must be a string."
            )

        if clause_id in seen_ids:
            raise ValueError(
                f"Duplicate clause_id detected: "
                f"{clause_id}"
            )

        seen_ids.add(clause_id)

        if not isinstance(
            clause["section"],
            str,
        ):
            raise ValueError(
                "section must be a string."
            )

        if not isinstance(
            clause["page"],
            int,
        ):
            raise ValueError(
                "page must be an integer."
            )

        if not isinstance(
            clause["original_text"],
            str,
        ):
            raise ValueError(
                "original_text must be a string."
            )

        if not isinstance(
            clause["processed_text"],
            str,
        ):
            raise ValueError(
                "processed_text must be a string."
            )

        if not isinstance(
            clause["language"],
            str,
        ):
            raise ValueError(
                "language must be a string."
            )


# ============================================================================
# Public API
# ============================================================================

def segment_contract(
    document: Dict[str, Any],
) -> Dict[str, Any]:

    try:

        valid, error_message = (
            _validate_document(document)
        )

        if not valid:
            return {
                "status": "error",
                "error_code": ERROR_CODE,
                "message": error_message,
            }

        engine = _SegmentationEngine(
            document["language"]
        )

        for page in document["pages"]:

            engine.process_page(
                page_number=page["page"],
                original_text=page["original_text"],
                processed_text=page["processed_text"],
            )

        result = {
            "status": "success",
            "clauses": engine.finalize(),
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

    if len(sys.argv) != 2:

        print(
            "Usage: python segmenter.py <input.json>",
            file=sys.stderr,
        )

        return 1

    input_path = sys.argv[1]

    try:

        with open(
            input_path,
            "r",
            encoding="utf-8",
        ) as file:

            document = json.load(file)

    except FileNotFoundError:

        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": ERROR_CODE,
                    "message": (
                        f"Input file not found: "
                        f"{input_path}"
                    ),
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
                    "message": (
                        f"Invalid JSON: {exc}"
                    ),
                },
                indent=2,
            )
        )

        return 1

    result = segment_contract(
        document
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )

    return (
        0
        if result.get("status") == "success"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(_main())