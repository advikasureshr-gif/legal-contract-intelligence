# Clause Segmentation

This is a standalone module in the multilingual Legal Contract Risk Analysis project.

Its job is simple:

> Take the contract text and split it into separate clauses in the correct order.

This module only handles **clause segmentation**. It does not do language detection, clause classification, risk scoring, XLM-RoBERTa processing, or UI work.

---

## Purpose

`segment_contract(document)` takes a contract that has already been extracted and cleaned, and splits it into a list of clause objects.

For example, if the contract contains:

```text
1. General Terms

1.1 The company agrees to provide the services.

1.2 The customer agrees to make the payment.

2. Confidentiality

The parties agree to keep all information confidential.
```

The module identifies `1.1`, `1.2`, and `2` as clause/section markers and keeps the text belonging to each one together.

The current version uses **rules and regular expressions (regex)**.&#x20;

The main goal is to make the output consistent and predictable.

---

## Architecture

The module follows these steps:

```text
Input validation
      ↓
Clean and normalize the page text
      ↓
Combine the lines from all pages
      ↓
Find possible clause markers
      ↓
Decide whether each marker is actually a new clause
      ↓
Handle headings and parent/child clause relationships
      ↓
Build the final clause objects
      ↓
Check the output
```

### 1. Input validation

First, the module checks whether the input has the required fields and whether the values are of the correct type.

For example, it checks things like:

- Is there a document?
- Are pages present?
- Does every page have text?
- Is the page number valid?

If something is wrong, the module returns an error instead of crashing.

### 2. Page normalization

The extracted text may contain things that are not actually part of the contract, such as:

```text
Page -2-
```

or

```text
-2-
```

These page markers are removed.

Tabs and different line-ending formats are also cleaned so that the text is handled consistently.

### 3. Line stream construction

The pages are then treated as one continuous sequence of lines.

For example:

```text
Page 1:
1.1 The company agrees to provide the service.

Page 2:
The service will continue for one year.

Page 3:
1.2 The customer will make the payment.
```

The text from page 2 is treated as a continuation of `1.1`.

**A page change does not automatically mean that a new clause has started.**

### 4. Structural detection

The module looks for patterns that could indicate a new clause or heading.

For example:

```text
1.
1.1
2.1.3
ARTICLE IV
SECTION 3.1
A.
(A)
CONFIDENTIALITY
```

Regex is mainly used here to recognize what a line **looks like**.

For example, regex can recognize that:

```text
1.2 Payment Terms
```

looks like a numbered clause.

But just because something matches a regex does **not** mean that it automatically becomes a new clause.

### 5. Rule-based decision

After finding a possible marker, the rules decide what it actually means.

For example:

```text
1.2 Payment Terms
```

can be treated as a clause.

But:

```text
The payment is due within 1.2 days.
```

should not become a new clause just because `1.2` appears in the text.

The rules also decide whether something is:

- a new clause
- a section or heading
- a list item belonging to the current clause
- normal continuation text

### 6. Hierarchy and headings

The module also handles cases where one heading contains smaller clauses.

For example:

```text
1. ESTABLISHMENT OF DISTRIBUTORSHIP

1.1 Appointment of Distributor
1.2 Territory
1.3 Term
```

Here, `1. ESTABLISHMENT OF DISTRIBUTORSHIP` is treated as the **section heading**, while `1.1`, `1.2`, and `1.3` are the actual child clauses.

This prevents the section heading from being incorrectly treated as a separate clause.

### 7. Clause construction and output validation

Finally, the module creates the clause objects using the agreed output format and checks that the output is valid.

---

## How regex and rules work together

Regex is only used to **find possible patterns**.

For example, regex can recognize:

```text
1.1 Payment Terms
```

as something that looks like a numbered clause.

The rule system then decides whether it should actually start a new clause.

So the process is:

```text
Regex → "This looks like a clause marker."

Rules → "Yes, this is a new clause."
        OR
        "No, this is just normal text/list text."
```

The module also has three internal priority levels:

- `STRONG` — patterns that are usually very reliable, such as clear clause numbering
- `MEDIUM` — patterns that are useful but need more checking
- `WEAK` — patterns based more on headings or other clues

These priorities are only used internally. They are **not** returned as a confidence score.

---

## Supported structures

The module currently supports the following types of contract structures.

### Numbered clauses

```text
1. Definitions
2. Payment
10. Termination
```

### Nested numbered clauses

```text
1.1
1.2
2.1.3
```

These can be used when a clause has smaller clauses inside it.

### Articles

```text
ARTICLE I
ARTICLE IV
ARTICLE 4
```

### Sections

```text
SECTION 1
SECTION 3.1
```

### Lettered paragraphs before numbered clauses

Some contracts start with lettered paragraphs:

```text
A. The parties agree...
B. The company shall...
C. The customer shall...
```

These are supported when they appear before the first numbered clause.

### Parenthetical lists

Lists inside a clause are supported:

```text
(A) First condition
(B) Second condition
(C) Third condition
```

and:

```text
(a) First item
(b) Second item
(c) Third item
```

and:

```text
(i) First item
(ii) Second item
(iii) Third item
```

These stay attached to their parent clause.

For example:

```text
1. Responsibilities

(A) The company must provide the service.
(B) The company must maintain the system.
```

`(A)` and `(B)` do **not** become separate top-level clauses.

### Standalone headings

The module can also recognize headings such as:

```text
CONFIDENTIALITY
```

or:

```text
Intellectual Property
```

as section/heading information.

### Continuation paragraphs

Not every paragraph starts with a number.

For example:

```text
1. Payment

The customer must make the payment within 30 days.
The payment must be made through the approved method.
```

The second sentence/paragraph is treated as part of the same clause when there is no new clause marker.

### Clauses that continue across pages

A clause can start on one page and continue on another page.

For example:

```text
Page 1:

1.1 The company agrees to provide the following services...
```

and then:

```text
Page 2:

The services will be provided throughout the contract period.
```

The page change does not automatically create a new clause.

Page-number text such as:

```text
Page -2-
```

is removed so that it does not become part of the clause text.

### Section and heading context

If a clause belongs to a section, that section information is kept until another heading or section changes it.

---

## Input format

The module expects a document in this format:

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

### Important

`language` is required, but this module does **not** detect the language.

For example, if the input says:

```json
"language": "en"
```

the module simply keeps `"en"` in the output.

The language detection, if needed, is handled somewhere else in the overall project.

---

## Output format

If the segmentation is successful, the module returns:

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

Each clause contains:

- `clause_id` — the clause number or fallback ID
- `section` — the section/heading it belongs to
- `page` — the page where the clause starts
- `text` — the actual clause text
- `language` — the language given in the input

---

## Error format

If something goes wrong, the module returns:

```json
{
  "status": "error",
  "error_code": "CLAUSE_SEGMENTATION_FAILED",
  "message": "Human-readable description of what was wrong."
}
```

`segment_contract` does not raise an exception to the caller.

Instead, errors such as these are returned using the error format:

- missing document
- wrong input type
- missing pages
- missing page text
- invalid page numbers
- empty document
- failure while building valid clause output

This makes it easier for the rest of the project to handle errors.

---

## Clause ID and fallback ID policy

If the contract already has a clause ID, the module keeps it exactly as it appears.

For example:

```text
1
1.1
ARTICLE IV
SECTION 3.1
A
```

These IDs are not changed or rewritten.

Sometimes a paragraph does not have any ID.

For example:

```text
CONFIDENTIALITY

The parties agree to keep all information confidential.
```

If the paragraph has no structural identifier, the module gives it a fallback ID:

```text
UNSPECIFIED-1
UNSPECIFIED-2
UNSPECIFIED-3
```

The numbers are assigned in document order.

This makes sure that useful text is not lost just because it did not have a clause number.

---

## Usage

To run the segmenter:

```bash
python segmenter.py sample_input.json
```

### Tests

To run all clause segmentation tests:

```bash
python -m unittest discover -s clause_segmentation -p "test_*.py"
```

---

## Design limitations

This is the **first rule-based version** of the module. It is designed to work well with the contract structures we currently expect, but it is not a perfect segmenter.

### 1. Clause numbers need to start their own line

The module mainly looks for markers at the beginning of a line.

For example:

```text
1. Payment Terms
```

can be detected.

But something like:

```text
The term is 5.2 years.
```

should not be treated as a new clause.

Similarly:

```text
See Section 3.1 for details.
```

should not be treated as a new `3.1` clause.

The trade-off is that if a real clause number is placed in the middle of a line instead of starting a new line, the module may miss it.

### 2. Single-letter roman numeral lists can be ambiguous

For example:

```text
(i)
(v)
```

can look like either a roman numeral list or a normal lower-case list item.

Because there is not enough information from a single marker alone, the module treats these as ordinary lower-case list items.

When there are several items together, such as:

```text
(i) First
(ii) Second
(iii) Third
```

the surrounding structure provides more information.

### 3. Heading detection is based on simple rules

The module looks for clues such as:

- short text
- standalone line
- upper-case or title-case text
- no sentence-ending punctuation

For example:

```text
CONFIDENTIALITY
```

is likely to be a heading.

If the module is not sure that something is a heading, it prefers to keep it with the previous clause instead of splitting the text incorrectly.

### 4. Section heading handling is limited

The special handling for something like:

```text
1. ESTABLISHMENT OF DISTRIBUTORSHIP
1.1 Appointment
1.2 Territory
```

only works when the top-level heading is immediately followed by its numbered children.

If there is body text between the heading and the child clauses, the special handling may not apply.

### 5. No AI model is used

The current module does not use:

- transformers
- embeddings
- sequence-labeling models
- XLM-RoBERTa

It is completely based on rules and regex.

In the future, a machine learning model could be added behind the same `segment_contract` function without changing the way other parts of the project use the module.
