import csv
from pathlib import Path

INPUT_FILE = Path("Datasets/processed/cuad_clause_dataset.csv")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Total rows: {len(rows)}")

print("\n" + "=" * 100)
print("SAMPLE CLAUSES")
print("=" * 100)

# Show examples from different parts of the dataset
indices = [
    0,
    100,
    500,
    1000,
    2000,
    3000,
    4000,
    5000,
    6000,
    7000,
    8000,
    9000,
    10000,
    11000,
    12000
]

for i in indices:
    if i >= len(rows):
        continue

    row = rows[i]

    print(f"\n[{i}]")
    print(f"CUAD label  : {row['cuad_label']}")
    print(f"Final label : {row['final_label']}")
    print(f"Answer     : {row['answer_text']}")
    print(f"Clause     : {row['clause_text']}")
    print("-" * 100)