import json
import re
import pandas as pd
from pathlib import Path

CUAD_FILE = Path("Datasets/CUADv1.json")
POSITIVE_FILE = Path("Datasets/processed/cuad_clause_dataset_final.csv")
OUTPUT_FILE = Path("Datasets/processed/cuad_other_clauses.csv")

print("Loading files...")

with open(CUAD_FILE, "r", encoding="utf-8") as f:
    cuad = json.load(f)

positives = pd.read_csv(POSITIVE_FILE)

# ---------------------------------------------------------
# 1. Collect locations of all CUAD answer spans
# ---------------------------------------------------------
# We will NEVER use text overlapping a CUAD answer span
# as an OTHER example.

answer_locations = {}

for contract in cuad["data"]:
    title = contract.get("title", "")

    for paragraph in contract.get("paragraphs", []):
        context = paragraph.get("context", "")

        for qa in paragraph.get("qas", []):
            for answer in qa.get("answers", []):
                start = answer.get("answer_start", -1)
                text = answer.get("text", "")

                if start >= 0 and text:
                    end = start + len(text)

                    if title not in answer_locations:
                        answer_locations[title] = []

                    answer_locations[title].append((start, end))


# ---------------------------------------------------------
# 2. Clause/sentence segmentation
# ---------------------------------------------------------
def split_into_clauses(text):
    """
    Split contract text into reasonably sized clauses/sentences.
    We use punctuation and line breaks as boundaries.
    """

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    # Sentence/clause boundary
    pieces = re.split(
        r"(?<=[.!?;])\s+|(?<=:)\s+",
        text
    )

    clauses = []

    for piece in pieces:
        piece = piece.strip()

        if not piece:
            continue

        # Remove extremely short fragments
        if len(piece) < 40:
            continue

        # Avoid enormous blocks
        if len(piece) > 1000:
            # Split long blocks by commas where possible
            subpieces = re.split(r"(?<=,)\s+", piece)

            for subpiece in subpieces:
                subpiece = subpiece.strip()

                if 40 <= len(subpiece) <= 1000:
                    clauses.append(subpiece)
        else:
            clauses.append(piece)

    return clauses


# ---------------------------------------------------------
# 3. Determine whether a candidate overlaps a CUAD answer
# ---------------------------------------------------------
def overlaps_answer(start, end, locations):
    for answer_start, answer_end in locations:
        if start < answer_end and end > answer_start:
            return True

    return False


# ---------------------------------------------------------
# 4. Extract OTHER clauses
# ---------------------------------------------------------
other_rows = []

contracts_processed = 0
candidate_count = 0
overlap_count = 0

for contract in cuad["data"]:
    title = contract.get("title", "")
    contracts_processed += 1

    locations = answer_locations.get(title, [])

    for paragraph in contract.get("paragraphs", []):
        context = paragraph.get("context", "")

        if not context.strip():
            continue

        # Preserve approximate positions by splitting manually
        # rather than using a normalized copy.
        boundaries = []

        pattern = re.compile(
            r"(?<=[.!?;])\s+|(?<=:)\s+|\n+"
        )

        last = 0

        for match in pattern.finditer(context):
            end = match.start()

            if end > last:
                boundaries.append((last, end))

            last = match.end()

        if last < len(context):
            boundaries.append((last, len(context)))

        for start, end in boundaries:
            raw_clause = context[start:end]
            clause = re.sub(r"\s+", " ", raw_clause).strip()

            if not clause:
                continue

            candidate_count += 1

            # Basic quality filtering
            if len(clause) < 40:
                continue

            if len(clause) > 1000:
                continue

            # Do not use anything overlapping a CUAD answer
            if overlaps_answer(start, end, locations):
                overlap_count += 1
                continue

            other_rows.append({
                "contract_title": title,
                "clause_text": clause,
                "final_label": "OTHER"
            })


# ---------------------------------------------------------
# 5. Remove duplicates
# ---------------------------------------------------------
other_df = pd.DataFrame(other_rows)

if len(other_df) > 0:

    other_df["normalized_text"] = (
        other_df["clause_text"]
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Remove duplicates within OTHER
    other_df = other_df.drop_duplicates(
        subset=["contract_title", "normalized_text"]
    )

    # Also remove exact matches against positive clauses
    positive_text = set(
        positives["clause_text"]
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    other_df = other_df[
        ~other_df["normalized_text"].isin(positive_text)
    ]

    other_df = other_df.drop(columns=["normalized_text"])

    other_df.insert(0, "id", range(len(other_df)))


# ---------------------------------------------------------
# 6. Save
# ---------------------------------------------------------
other_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


# ---------------------------------------------------------
# 7. Report
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("OTHER CLAUSE GENERATION COMPLETE")
print("=" * 60)

print(f"Contracts processed       : {contracts_processed}")
print(f"Candidate clauses         : {candidate_count}")
print(f"Overlapping candidates    : {overlap_count}")
print(f"Final OTHER examples      : {len(other_df)}")

print("\nSaved to:")
print(OUTPUT_FILE)

print("\nSample OTHER clauses:")

for _, row in other_df.head(10).iterrows():
    print("\n---")
    print(row["clause_text"])