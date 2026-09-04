import json
import csv
from pathlib import Path


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

INPUT_FILE = Path("Datasets/CUADv1.json")
OUTPUT_DIR = Path("Datasets/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Load CUAD
# ---------------------------------------------------------

print("Loading CUAD...")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    cuad = json.load(f)

print("CUAD loaded.")


# ---------------------------------------------------------
# Extract answer spans
# ---------------------------------------------------------

rows = []
question_types = {}

contracts_processed = 0
qas_processed = 0
answers_extracted = 0
invalid_spans = 0


for contract in cuad["data"]:

    contract_title = contract.get("title", "")

    contracts_processed += 1

    for paragraph in contract.get("paragraphs", []):

        context = paragraph.get("context", "")

        for qa in paragraph.get("qas", []):

            qas_processed += 1

            question = qa.get("question", "").strip()
            is_impossible = qa.get("is_impossible", False)

            # Count this question type
            if question not in question_types:
                question_types[question] = 0

            # Extract answer spans
            answers = qa.get("answers", [])

            for answer in answers:

                answer_text = answer.get("text", "")
                answer_start = answer.get("answer_start", -1)

                # -------------------------------------------------
                # Verify answer span against original context
                # -------------------------------------------------

                if answer_start >= 0:

                    extracted_text = context[
                        answer_start:
                        answer_start + len(answer_text)
                    ]

                    if extracted_text != answer_text:

                        invalid_spans += 1

                        print("\nWARNING: Span mismatch")
                        print("Question:", question)
                        print("Expected:", repr(answer_text))
                        print("Found:", repr(extracted_text))

                        continue

                # -------------------------------------------------
                # Store ONLY the information we need
                # -------------------------------------------------

                rows.append({
                    "contract_title": contract_title,
                    "question_type": question,
                    "answer_text": answer_text,
                    "answer_start": answer_start
                })

                question_types[question] += 1
                answers_extracted += 1


# ---------------------------------------------------------
# Save answer-span dataset
# ---------------------------------------------------------

answer_output = OUTPUT_DIR / "cuad_answer_spans.csv"

with open(answer_output, "w", encoding="utf-8", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "contract_title",
            "question_type",
            "answer_text",
            "answer_start"
        ]
    )

    writer.writeheader()
    writer.writerows(rows)


# ---------------------------------------------------------
# Save question-type statistics
# ---------------------------------------------------------

question_output = OUTPUT_DIR / "cuad_question_types.csv"

with open(question_output, "w", encoding="utf-8", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "question_type",
        "answer_count"
    ])

    for question, count in sorted(
        question_types.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        writer.writerow([
            question,
            count
        ])


# ---------------------------------------------------------
# Print summary
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("CUAD PREPROCESSING COMPLETE")
print("=" * 60)

print(f"Contracts processed : {contracts_processed}")
print(f"QAs processed       : {qas_processed}")
print(f"Answer spans        : {answers_extracted}")
print(f"Invalid spans       : {invalid_spans}")
print(f"Question types      : {len(question_types)}")

print("\nQuestion types:")

for question, count in sorted(
    question_types.items(),
    key=lambda x: x[1],
    reverse=True
):

    print(f"{count:5d} | {question}")


print("\nSaved:")
print(f"  {answer_output}")
print(f"  {question_output}")