from datasets import load_from_disk, DatasetDict
from pathlib import Path


INPUT_PATH = "Datasets/unfair_tos"
OUTPUT_PATH = "Datasets/processed/risk_dataset"


def prepare_split(dataset):
    records = []

    for row in dataset:
        labels = row["labels"]

        records.append({
            "text": row["text"],
            "is_unfair": 1 if len(labels) > 0 else 0,
            "unfair_labels": labels,
        })

    return dataset.from_list(records)


def main():
    print("Loading UNFAIR-ToS...")

    dataset = load_from_disk(INPUT_PATH)

    prepared = DatasetDict()

    for split in ["train", "validation", "test"]:
        print(f"\nPreparing {split}...")

        prepared[split] = prepare_split(dataset[split])

        total = len(prepared[split])
        unfair = sum(prepared[split]["is_unfair"])
        fair = total - unfair

        print(f"  Total:  {total}")
        print(f"  Fair:   {fair}")
        print(f"  Unfair: {unfair}")

    output_dir = Path(OUTPUT_PATH)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    prepared.save_to_disk(OUTPUT_PATH)

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()