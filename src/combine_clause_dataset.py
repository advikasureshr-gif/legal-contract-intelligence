import pandas as pd
from pathlib import Path

POSITIVE_FILE = Path(
    "Datasets/processed/cuad_clause_dataset_final.csv"
)

OTHER_FILE = Path(
    "Datasets/processed/cuad_other_clauses.csv"
)

OUTPUT_FILE = Path(
    "Datasets/processed/clause_dataset_combined.csv"
)

TARGET_OTHER = 9500
RANDOM_SEED = 42

print("Loading datasets...")

positive_df = pd.read_csv(POSITIVE_FILE)
other_df = pd.read_csv(OTHER_FILE)

print(f"Positive examples : {len(positive_df)}")
print(f"OTHER examples    : {len(other_df)}")


# ---------------------------------------------------------
# 1. Clean columns
# ---------------------------------------------------------

positive_df = positive_df[
    ["contract_title", "clause_text", "final_label"]
].copy()

other_df = other_df[
    ["contract_title", "clause_text", "final_label"]
].copy()


# ---------------------------------------------------------
# 2. Remove duplicate text
# ---------------------------------------------------------

positive_df["normalized_text"] = (
    positive_df["clause_text"]
    .astype(str)
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

other_df["normalized_text"] = (
    other_df["clause_text"]
    .astype(str)
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

positive_df = positive_df.drop_duplicates(
    subset=["normalized_text"]
)

other_df = other_df.drop_duplicates(
    subset=["normalized_text"]
)


# ---------------------------------------------------------
# 3. Remove OTHER clauses that duplicate positives
# ---------------------------------------------------------

positive_texts = set(
    positive_df["normalized_text"]
)

other_df = other_df[
    ~other_df["normalized_text"].isin(positive_texts)
].copy()


# ---------------------------------------------------------
# 4. Sample OTHER proportionally across contracts
# ---------------------------------------------------------

print(
    f"\nUsable OTHER examples before sampling: "
    f"{len(other_df)}"
)

if len(other_df) > TARGET_OTHER:

    # Number of contracts represented
    contracts = other_df["contract_title"].unique()

    print(
        f"Contracts containing OTHER clauses: "
        f"{len(contracts)}"
    )

    # First sample a roughly equal amount per contract.
    per_contract = max(
        1,
        TARGET_OTHER // len(contracts)
    )

    sampled_parts = []

    for contract, group in other_df.groupby(
        "contract_title"
    ):
        sample_size = min(
            len(group),
            per_contract
        )

        sampled_parts.append(
            group.sample(
                n=sample_size,
                random_state=RANDOM_SEED
            )
        )

    sampled_other = pd.concat(
        sampled_parts,
        ignore_index=True
    )

    # If we are below target, fill the remaining slots
    # randomly from unused OTHER examples.
    if len(sampled_other) < TARGET_OTHER:

        sampled_indices = set(
            sampled_other.index
        )

        # Use a separate random sample from the
        # remaining pool.
        remaining = other_df[
            ~other_df["clause_text"].isin(
                sampled_other["clause_text"]
            )
        ]

        remaining_needed = (
            TARGET_OTHER - len(sampled_other)
        )

        if len(remaining) > remaining_needed:

            extra = remaining.sample(
                n=remaining_needed,
                random_state=RANDOM_SEED
            )

        else:
            extra = remaining

        sampled_other = pd.concat(
            [sampled_other, extra],
            ignore_index=True
        )

    other_df = sampled_other


# ---------------------------------------------------------
# 5. If fewer than target, use everything
# ---------------------------------------------------------

else:

    print(
        "Available OTHER examples are below "
        "the target, so all will be used."
    )


# ---------------------------------------------------------
# 6. Rename label column
# ---------------------------------------------------------

positive_df = positive_df.rename(
    columns={"final_label": "label"}
)

other_df = other_df.rename(
    columns={"final_label": "label"}
)


# ---------------------------------------------------------
# 7. Combine
# ---------------------------------------------------------

combined = pd.concat(
    [
        positive_df[
            ["contract_title", "clause_text", "label"]
        ],
        other_df[
            ["contract_title", "clause_text", "label"]
        ]
    ],
    ignore_index=True
)


# ---------------------------------------------------------
# 8. Final duplicate removal
# ---------------------------------------------------------

combined["normalized_text"] = (
    combined["clause_text"]
    .astype(str)
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

combined = combined.drop_duplicates(
    subset=["normalized_text"]
)

combined = combined.drop(
    columns=["normalized_text"]
)


# ---------------------------------------------------------
# 9. Shuffle
# ---------------------------------------------------------

combined = combined.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)


# ---------------------------------------------------------
# 10. Add IDs
# ---------------------------------------------------------

combined.insert(
    0,
    "id",
    range(len(combined))
)


# ---------------------------------------------------------
# 11. Save
# ---------------------------------------------------------

combined.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


# ---------------------------------------------------------
# 12. Report
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("COMBINED CLAUSE DATASET")
print("=" * 60)

print(f"Total examples : {len(combined)}")
print(f"Total labels   : {combined['label'].nunique()}")

print("\nExamples per label:")

counts = combined["label"].value_counts()

for label, count in counts.items():
    print(f"{count:5d} | {label}")

print("\nContracts represented:")
print(combined["contract_title"].nunique())

print("\nSaved to:")
print(OUTPUT_FILE)