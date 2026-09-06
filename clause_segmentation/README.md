# Clause Segmentation

Standalone module for the multilingual Legal Contract Risk Analysis
pipeline. This module owns exactly one responsibility:

> Turn a page-structured contract into an ordered list of clause objects.

It does **not** do language detection, classification, risk scoring,
XLM-RoBERTa representation, or UI work — those are downstream modules.

## Purpose

`segment_contract(document)` takes a contract that has already been
extracted and (optionally) split by page, and returns a deterministic list
of clause objects following the team's agreed schema. It is a first,
rule-based + regex version, in line with the project specification. A
future sequence-labeling model can be slotted in later without changing
the public API.

## Architecture

```
Input validation
      ↓
Page normalization (CRLF/tabs, page-marker artifact stripping)
      ↓
Line stream construction (pages -> flat line sequence, page
transitions treated like paragraph breaks, never like clause breaks)
      ↓
Structural detection (regex) — ARTICLE / SECTION / numeric / recital /
list markers / heading heuristics
      ↓
Rule-based decision layer — is this detected marker a real clause
boundary, a section/heading update, a list item that stays with its
parent, or plain continuation text?
      ↓
Hierarchy / container resolution — a bare top-level numeric heading
("1. ESTABLISHMENT OF DISTRIBUTORSHIP") that is immediately followed by
its own numbered children ("1.1", "1.2", ...) is treated as a section
label, not as a clause of its own
      ↓
Clause construction & output validation
```

Regex is used **only** for detection (recognizing that a line looks like
`1.1`, `ARTICLE IV`, `(A)`, etc.). Whether a detected pattern actually
becomes a clause boundary is a separate rule-based decision, so the rule
layer can also recognize obvious boundaries (e.g. isolated ALL-CAPS
headings) that no single regex is responsible for.

Internally, detections carry a `Priority` of `STRONG` / `MEDIUM` / `WEAK`
purely to document how much the engine trusts a given kind of match
(explicit numbering vs. a parenthetical list vs. a heuristic heading).
This is **never** exposed as a numeric confidence score in the output.

## Supported structures

- Numbered clauses: `1.`, `2.`, `10.`
- Nested numbered clauses: `1.1`, `1.2`, `2.1.3`
- `ARTICLE I` / `ARTICLE IV` / `ARTICLE 4`
- `SECTION 1` / `SECTION 3.1`
- Recital-style lettered paragraphs before the first numbered clause
  (`A.`, `B.`, `C.`)
- Parenthetical lists: `(A)(B)(C)`, `(a)(b)(c)`, `(i)(ii)(iii)` — always
  kept attached to their parent clause, never promoted to top-level
  clauses
- Standalone heading lines (`CONFIDENTIALITY`, `Intellectual Property`)
- Continuation paragraphs (unmarked body text that belongs to the
  preceding clause)
- Page-spanning clauses, with page-marker artifacts (`Page -2-`, `-2-`)
  stripped before they can pollute clause text
- Section/heading context that persists until the next heading or
  container is encountered

## Input format

```json
{
  "document_name": "contract.pdf",
  "language": "en",
  "pages": [
    {"page": 1, "text": "..."},
    {"page": 2, "text": "..."}
  ]
}
```

`language` is a required, document-level field (this module does not
detect language; it only preserves whatever value it is given).

## Output format

```json
{
  "status": "success",
  "clauses": [
    {
      "clause_id": "1.1",
      "section": "ESTABLISHMENT OF DISTRIBUTORSHIP",
      "page": 1,
      "text": "Grant and Acceptance. The Company hereby grants ...",
      "language": "en"
    }
  ]
}
```

## Error format

```json
{
  "status": "error",
  "error_code": "CLAUSE_SEGMENTATION_FAILED",
  "message": "Human-readable description of what was wrong."
}
```

`segment_contract` never raises; every failure path (missing document,
wrong types, missing pages/text, invalid page numbers, empty document,
or an internal inability to build valid output) is reported through this
schema.

## Clause ID and fallback ID policy

Existing structural identifiers (`1`, `1.1`, `ARTICLE IV`, `SECTION 3.1`,
a recital letter like `A`) are always preserved exactly as found and are
never rewritten. If a paragraph has no structural identifier at all (for
example, body text that immediately follows a bare heading such as
`CONFIDENTIALITY`), it is assigned a deterministic fallback id of the
form `UNSPECIFIED-1`, `UNSPECIFIED-2`, ... in document order. This keeps
the module from silently discarding meaningful text.

## Usage

```bash
python segmenter.py sample_input.json
```

Tests:

```bash
python -m unittest discover -s clause_segmentation -p "test_*.py"
```

## Design limitations

This is a deterministic, rule-based first version, not a perfect
segmenter:

- Structural detection is line-anchored (a marker must open its own
  line). This is what keeps things like "The term is 5.2 years." or
  "See Section 3.1 for details." from being misread as numbering, but it
  also means a genuine clause number that is not placed at the very
  start of its own line will be missed.
- A single-character parenthetical marker that happens to be a roman
  numeral letter (e.g. `(i)`, `(v)`) is classified as an ordinary
  lower-case list item rather than a roman numeral, since one letter
  alone is ambiguous without broader list context.
- Heading detection is a conservative heuristic (short, standalone,
  upper/title-case, no terminal sentence punctuation). It intentionally
  favors under-splitting: when the evidence for a boundary is weak, the
  module keeps text attached to the previous clause rather than
  fragmenting it.
- The "container heading" collapse (e.g. `1. ESTABLISHMENT OF
  DISTRIBUTORSHIP` folding into a section label instead of its own
  clause) is deliberately narrow: it only applies to a bare top-level
  numeric heading that is immediately followed by its own numbered
  children with no intervening body text.
- No transformer, embedding, or sequence-labeling model is used. A
  future model-based fallback could be added behind the same
  `segment_contract` API without breaking downstream consumers.