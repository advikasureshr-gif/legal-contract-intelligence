import json
import csv
import re
from pathlib import Path

CUAD_FILE = Path("Datasets/CUADv1.json")
MAPPING_FILE = Path("Datasets/processed/cuad_label_mapping.csv")
OUTPUT_FILE = Path("Datasets/processed/cuad_clause_dataset.csv")

print("Loading CUAD...")

with open(CUAD_FILE, "r", encoding="utf-8") as f:
    cuad = json.load(f)

print("CUAD loaded.")

# ---------------------------------------------------------
# Load label mapping
# ---------------------------------------------------------

mapping = {}

with open(MAPPING_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        mapping[row["cuad_label"]] = {
            "final_label": row["final_label"],
            "action": row["action"]
        }

print(f"Loaded {len(mapping)} CUAD label mappings.")

# ---------------------------------------------------------
# Extract CUAD label from question
# ---------------------------------------------------------

def extract_label(question):
    match = re.search(r'related to "([^"]+)"', question)

    if match:
        return match.group(1)

    return question


# ---------------------------------------------------------
# Find containing sentence
# ---------------------------------------------------------

def extract_sentence(context, start, answer_text):

    if start < 0:
        return None

    end = start + len(answer_text)

    # Make sure the span actually matches
    if context[start:end] != answer_text:
        return None

    # Search backwards for sentence/clause boundary
    left_candidates = [
        context.rfind(".", 0, start),
        context.rfind("?", 0, start),
        context.rfind("!", 0, start),
        context.rfind("\n", 0, start),
        context.rfind(";", 0, start),
    ]

    left = max(left_candidates)

    # Search forwards
    right_candidates = []

    for char in [".", "?", "!", "\n", ";"]:
        pos = context.find(char, end)

        if pos != -1:
            right_candidates.append(pos)

    if right_candidates:
        right = min(right_candidates)
    else:
        right = len(context)

    text = context[left + 1:right + 1].strip()

    return text


# ---------------------------------------------------------
# Process
# ---------------------------------------------------------

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

            # Ignore labels marked DROP
            if action == "DROP":
                dropped_examples += len(qa.get("answers", []))
                continue

            for answer in qa.get("answers", []):

                total_answers += 1

                answer_text = answer.get("text", "")
                answer_start = answer.get("answer_start", -1)

                clause_text = extract_sentence(
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
                    "answer_text": answer_text,
                    "clause_text": clause_text
                })

                valid_examples += 1


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

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
    writer.writerows(rows)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CLAUSE DATASET CREATED")
print("=" * 70)

print(f"CUAD answer spans examined : {total_answers}")
print(f"Valid clause examples      : {valid_examples}")
print(f"Skipped examples           : {skipped_examples}")
print(f"Dropped metadata examples  : {dropped_examples}")

print(f"\nSaved to:")
print(OUTPUT_FILE)

print("\nExamples per final label:")

counts = {}

for row in rows:
    label = row["final_label"]
    counts[label] = counts.get(label, 0) + 1

for label, count in sorted(
    counts.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{count:5} | {label}")

print("\n" + "=" * 70)