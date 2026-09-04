import json
import csv
import re
from pathlib import Path

CUAD_FILE = Path("Datasets/CUADv1.json")
MAPPING_FILE = Path("Datasets/processed/cuad_label_mapping.csv")
OUTPUT_FILE = Path("Datasets/processed/cuad_clause_dataset_v2.csv")

print("Loading CUAD...")

with open(CUAD_FILE, "r", encoding="utf-8") as f:
    cuad = json.load(f)

print("CUAD loaded.")


# =========================================================
# LOAD LABEL MAPPING
# =========================================================

mapping = {}

with open(MAPPING_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        mapping[row["cuad_label"]] = {
            "final_label": row["final_label"],
            "action": row["action"]
        }

print(f"Loaded {len(mapping)} CUAD label mappings.")


# =========================================================
# EXTRACT CUAD LABEL FROM QUESTION
# =========================================================

def extract_label(question):
    match = re.search(r'related to "([^"]+)"', question)

    if match:
        return match.group(1)

    return question


# =========================================================
# NORMALIZE WHITESPACE
# =========================================================

def normalize_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========================================================
# FIND SENTENCE BOUNDARIES
# =========================================================

def find_left_boundary(context, start):
    """
    Find a reasonable beginning for the clause containing
    the answer.
    """

    # Look for strong structural boundaries.
    candidates = []

    for char in [".", "?", "!", ";", "\n"]:
        position = context.rfind(char, 0, start)

        if position != -1:
            candidates.append(position)

    if not candidates:
        return 0

    return max(candidates) + 1


def find_right_boundary(context, end):
    """
    Find a reasonable end for the clause containing
    the answer.
    """

    candidates = []

    for char in [".", "?", "!", ";", "\n"]:
        position = context.find(char, end)

        if position != -1:
            candidates.append(position)

    if not candidates:
        return len(context)

    return min(candidates) + 1


# =========================================================
# EXTRACT SURROUNDING CLAUSE
# =========================================================

def extract_clause(context, answer_start, answer_text):

    if answer_start < 0:
        return None

    answer_end = answer_start + len(answer_text)

    # Verify the original answer span.
    if context[answer_start:answer_end] != answer_text:
        return None

    # -----------------------------------------------------
    # If the answer is already substantial, preserve it.
    # -----------------------------------------------------

    normalized_answer = normalize_text(answer_text)

    # Long answers are generally already meaningful
    # contractual provisions.
    if len(normalized_answer) >= 120:

        return normalized_answer

    # -----------------------------------------------------
    # Otherwise recover surrounding sentence/clause.
    # -----------------------------------------------------

    left = find_left_boundary(context, answer_start)
    right = find_right_boundary(context, answer_end)

    clause = context[left:right]

    clause = normalize_text(clause)

    # -----------------------------------------------------
    # If the recovered clause is still too short,
    # expand carefully to neighboring text.
    # -----------------------------------------------------

    if len(clause) < 80:

        # Try extending backwards to the previous boundary.
        previous_left_candidates = []

        for char in [".", "?", "!", ";", "\n"]:
            position = context.rfind(char, 0, max(0, left - 1))

            if position != -1:
                previous_left_candidates.append(position)

        if previous_left_candidates:
            expanded_left = max(previous_left_candidates) + 1
        else:
            expanded_left = 0

        expanded = context[expanded_left:right]
        expanded = normalize_text(expanded)

        if len(expanded) <= 600:
            clause = expanded

    # -----------------------------------------------------
    # Don't create enormous examples.
    # -----------------------------------------------------

    if len(clause) > 1000:

        # Prefer the original answer plus nearby context.
        center = answer_start

        start = max(0, center - 300)
        end = min(len(context), answer_end + 500)

        clause = normalize_text(context[start:end])

    return clause


# =========================================================
# PROCESS DATA
# =========================================================

rows = []

total_answers = 0
valid_examples = 0
skipped_examples = 0
dropped_examples = 0

for contract in cuad["data"]:

    contract_title = contract.get("title", "")

    for paragraph in contract.get("paragraphs", []):

        context = paragraph.get("context", "")

        for qa in paragraph.get("qas", []):

            question = qa.get("question", "").strip()

            cuad_label = extract_label(question)

            if cuad_label not in mapping:
                continue

            final_label = mapping[cuad_label]["final_label"]
            action = mapping[cuad_label]["action"]

            answers = qa.get("answers", [])

            # -------------------------------------------------
            # Remove metadata labels from clause classification
            # -------------------------------------------------

            if action == "DROP":

                dropped_examples += len(answers)
                continue

            for answer in answers:

                total_answers += 1

                answer_text = answer.get("text", "")
                answer_start = answer.get("answer_start", -1)

                clause_text = extract_clause(
                    context,
                    answer_start,
                    answer_text
                )

                if not clause_text:

                    skipped_examples += 1
                    continue

                rows.append({
                    "contract_title": contract_title,
                    "cuad_label": cuad_label,
                    "final_label": final_label,
                    "answer_text": normalize_text(answer_text),
                    "clause_text": clause_text
                })

                valid_examples += 1


# =========================================================
# REMOVE EXACT DUPLICATES
# =========================================================

unique_rows = []
seen = set()

for row in rows:

    key = (
        row["contract_title"],
        row["final_label"],
        row["clause_text"]
    )

    if key not in seen:

        seen.add(key)
        unique_rows.append(row)

duplicate_count = len(rows) - len(unique_rows)


# =========================================================
# SAVE
# =========================================================

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "contract_title",
            "cuad_label",
            "final_label",
            "answer_text",
            "clause_text"
        ]
    )

    writer.writeheader()
    writer.writerows(unique_rows)


# =========================================================
# STATISTICS
# =========================================================

counts = {}

for row in unique_rows:

    label = row["final_label"]

    counts[label] = counts.get(label, 0) + 1


print("\n" + "=" * 75)
print("CUAD CLAUSE DATASET V2 CREATED")
print("=" * 75)

print(f"CUAD answer spans examined : {total_answers}")
print(f"Valid examples             : {valid_examples}")
print(f"Duplicates removed         : {duplicate_count}")
print(f"Final unique examples      : {len(unique_rows)}")
print(f"Skipped examples           : {skipped_examples}")
print(f"Dropped metadata examples  : {dropped_examples}")

print(f"\nSaved to:")
print(OUTPUT_FILE)

print("\nExamples per final label:")
print("-" * 50)

for label, count in sorted(
    counts.items(),
    key=lambda x: x[1],
    reverse=True
):

    print(f"{count:5} | {label}")

print("\n" + "=" * 75)