import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

INPUT_FILE = Path(
    "Datasets/processed/clause_dataset_combined.csv"
)

OUTPUT_DIR = Path(
    "Datasets/processed/clause_splits"
)

RANDOM_SEED = 42

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("Loading combined dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Total examples  : {len(df)}")
print(
    f"Total contracts : "
    f"{df['contract_title'].nunique()}"
)
print(
    f"Total labels    : "
    f"{df['label'].nunique()}"
)


# ---------------------------------------------------------
# 1. Get unique contracts
# ---------------------------------------------------------

contracts = df["contract_title"].drop_duplicates().tolist()

print("\nSplitting contracts...")


# ---------------------------------------------------------
# 2. Train = 70%
#    Temporary = 30%
# ---------------------------------------------------------

train_contracts, temp_contracts = train_test_split(
    contracts,
    test_size=0.30,
    random_state=RANDOM_SEED
)


# ---------------------------------------------------------
# 3. Validation = 15%
#    Test = 15%
# ---------------------------------------------------------

validation_contracts, test_contracts = train_test_split(
    temp_contracts,
    test_size=0.50,
    random_state=RANDOM_SEED
)


# ---------------------------------------------------------
# 4. Create datasets
# ---------------------------------------------------------

train_df = df[
    df["contract_title"].isin(train_contracts)
].copy()

validation_df = df[
    df["contract_title"].isin(validation_contracts)
].copy()

test_df = df[
    df["contract_title"].isin(test_contracts)
].copy()


# ---------------------------------------------------------
# 5. Shuffle each split
# ---------------------------------------------------------

train_df = train_df.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)

validation_df = validation_df.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)

test_df = test_df.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)


# ---------------------------------------------------------
# 6. Reset IDs
# ---------------------------------------------------------
# The combined dataset already contains an "id" column,
# so we reset its values instead of inserting a new column.

train_df["id"] = range(len(train_df))
validation_df["id"] = range(len(validation_df))
test_df["id"] = range(len(test_df))


# ---------------------------------------------------------
# 7. Save
# ---------------------------------------------------------

train_file = OUTPUT_DIR / "train.csv"
validation_file = OUTPUT_DIR / "validation.csv"
test_file = OUTPUT_DIR / "test.csv"

train_df.to_csv(
    train_file,
    index=False,
    encoding="utf-8"
)

validation_df.to_csv(
    validation_file,
    index=False,
    encoding="utf-8"
)

test_df.to_csv(
    test_file,
    index=False,
    encoding="utf-8"
)


# ---------------------------------------------------------
# 8. Verify no contract leakage
# ---------------------------------------------------------

train_set = set(train_contracts)
validation_set = set(validation_contracts)
test_set = set(test_contracts)

assert train_set.isdisjoint(validation_set)
assert train_set.isdisjoint(test_set)
assert validation_set.isdisjoint(test_set)


# ---------------------------------------------------------
# 9. Report
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("CONTRACT-LEVEL DATASET SPLIT")
print("=" * 60)

print(
    f"Train      : {len(train_contracts):3d} contracts | "
    f"{len(train_df):5d} examples"
)

print(
    f"Validation : {len(validation_contracts):3d} contracts | "
    f"{len(validation_df):5d} examples"
)

print(
    f"Test       : {len(test_contracts):3d} contracts | "
    f"{len(test_df):5d} examples"
)

print("\nLabel distribution:")

print("\nTRAIN:")
print(train_df["label"].value_counts())

print("\nVALIDATION:")
print(validation_df["label"].value_counts())

print("\nTEST:")
print(test_df["label"].value_counts())

print("\nFiles saved:")
print(train_file)
print(validation_file)
print(test_file)

print("\nLeakage check: PASSED")
