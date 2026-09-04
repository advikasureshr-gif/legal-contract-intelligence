import pandas as pd
from pathlib import Path

INPUT_FILE = Path("Datasets/processed/cuad_clause_dataset_v2.csv")
OUTPUT_FILE = Path("Datasets/processed/cuad_clause_dataset_final.csv")

print("Loading clause dataset...")
df = pd.read_csv(INPUT_FILE)

print(f"Original examples: {len(df)}")
print(f"Original labels: {df['final_label'].nunique()}")

# Remove PARTIES because it is document metadata/entity information
df = df[df["final_label"] != "PARTIES"].copy()

# Remove empty clauses just in case
df = df[
    df["clause_text"].notna()
    & (df["clause_text"].str.strip() != "")
].copy()

# Reset IDs
df.insert(0, "id", range(len(df)))

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

print("\n" + "=" * 60)
print("FINAL POSITIVE CLAUSE DATASET")
print("=" * 60)

print(f"Examples: {len(df)}")
print(f"Labels:   {df['final_label'].nunique()}")

print("\nExamples per label:")
counts = df["final_label"].value_counts()

for label, count in counts.items():
    print(f"{count:5d} | {label}")

print("\nSaved to:")
print(OUTPUT_FILE)