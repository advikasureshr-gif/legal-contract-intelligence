import json
import csv
import re
from pathlib import Path


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

INPUT_FILE = Path("Datasets/CUADv1.json")
OUTPUT_FILE = Path("Datasets/processed/cuad_label_examples.csv")


# ---------------------------------------------------------
# Load CUAD
# ---------------------------------------------------------

print("Loading CUAD...")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    cuad = json.load(f)

print("CUAD loaded.")


# ---------------------------------------------------------
# Extract label name from CUAD question
# ---------------------------------------------------------

def extract_label(question):
    match = re.search(r'related to "([^"]+)"', question)

    if match:
        return match.group(1)

    return question


# ---------------------------------------------------------
# Collect labels
# ---------------------------------------------------------

labels = {}

for contract in cuad["data"]:

    for paragraph in contract.get("paragraphs", []):

        context = paragraph.get("context", "")

        for qa in paragraph.get("qas", []):

            question = qa.get("question", "").strip()
            label = extract_label(question)

            answers = qa.get("answers", [])

            if label not in labels:

                labels[label] = {
                    "count": 0,
                    "example": "",
                    "contract": ""
                }

            labels[label]["count"] += len(answers)

            # Save first valid answer as example
            if not labels[label]["example"] and answers:

                answer = answers[0]

                answer_text = answer.get("text", "")
                answer_start = answer.get("answer_start", -1)

                if answer_start >= 0:

                    extracted = context[
                        answer_start:
                        answer_start + len(answer_text)
                    ]

                    if extracted == answer_text:

                        labels[label]["example"] = answer_text
                        labels[label]["contract"] = contract.get(
                            "title", ""
                        )


# ---------------------------------------------------------
# Sort by frequency
# ---------------------------------------------------------

sorted_labels = sorted(
    labels.items(),
    key=lambda x: x[1]["count"],
    reverse=True
)


# ---------------------------------------------------------
# Save compact CSV
# ---------------------------------------------------------

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "id",
        "label",
        "answer_count",
        "example_answer"
    ])

    for i, (label, info) in enumerate(
        sorted_labels,
        start=1
    ):

        writer.writerow([
            i,
            label,
            info["count"],
            info["example"]
        ])


# ---------------------------------------------------------
# Print compact output
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("CUAD LABEL SUMMARY")
print("=" * 80)

print(
    f"{'ID':<4}"
    f"{'LABEL':<40}"
    f"{'COUNT':>7}"
)

print("-" * 80)

for i, (label, info) in enumerate(
    sorted_labels,
    start=1
):

    print(
        f"{i:<4}"
        f"{label:<40}"
        f"{info['count']:>7}"
    )


print("\n" + "=" * 80)
print(f"Total labels: {len(sorted_labels)}")
print(f"Saved to: {OUTPUT_FILE}")
print("=" * 80)