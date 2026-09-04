import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer

MODEL_NAME = "joelniklaus/legal-xlm-roberta-base"

DATA_DIR = Path("Datasets/processed/clause_splits")

print("=" * 60)
print("LEGAL XLM-ROBERTA CHECK")
print("=" * 60)

print(f"\nModel: {MODEL_NAME}")
print("\nDownloading/loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Tokenizer loaded successfully.")


# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------

train = pd.read_csv(DATA_DIR / "train.csv")
validation = pd.read_csv(DATA_DIR / "validation.csv")
test = pd.read_csv(DATA_DIR / "test.csv")


# ---------------------------------------------------------
# Token length analysis
# ---------------------------------------------------------

def analyze_dataset(df, name):
    print(f"\n{'-' * 60}")
    print(f"{name}")
    print(f"{'-' * 60}")

    texts = df["clause_text"].astype(str).tolist()

    lengths = []

    for text in texts:
        tokens = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False
        )

        lengths.append(len(tokens["input_ids"]))

    series = pd.Series(lengths)

    print(f"Examples       : {len(series)}")
    print(f"Minimum tokens : {series.min()}")
    print(f"Average tokens : {series.mean():.1f}")
    print(f"Median tokens  : {series.median():.1f}")
    print(f"95th percentile: {series.quantile(0.95):.1f}")
    print(f"99th percentile: {series.quantile(0.99):.1f}")
    print(f"Maximum tokens : {series.max()}")

    print("\nWould exceed 512 tokens:")

    over_512 = (series > 512).sum()

    print(
        f"{over_512} / {len(series)} "
        f"({over_512 / len(series) * 100:.2f}%)"
    )

    return series


train_lengths = analyze_dataset(train, "TRAIN")
validation_lengths = analyze_dataset(validation, "VALIDATION")
test_lengths = analyze_dataset(test, "TEST")


# ---------------------------------------------------------
# Overall result
# ---------------------------------------------------------

all_lengths = pd.concat(
    [train_lengths, validation_lengths, test_lengths],
    ignore_index=True
)

print("\n" + "=" * 60)
print("OVERALL TOKEN LENGTH")
print("=" * 60)

print(f"Total examples : {len(all_lengths)}")
print(f"Average tokens : {all_lengths.mean():.1f}")
print(f"Median tokens  : {all_lengths.median():.1f}")
print(f"95th percentile: {all_lengths.quantile(0.95):.1f}")
print(f"99th percentile: {all_lengths.quantile(0.99):.1f}")
print(f"Maximum tokens : {all_lengths.max()}")

over_512 = (all_lengths > 512).sum()

print(
    f"\nOver 512 tokens: "
    f"{over_512} / {len(all_lengths)} "
    f"({over_512 / len(all_lengths) * 100:.2f}%)"
)

print("\nTokenizer check complete.")