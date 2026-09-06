from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from pathlib import Path
import joblib


# ============================================================
# Configuration
# ============================================================

# Best-performing fine-tuned XLM-R model
MODEL_PATH = "models/clause_classifier_weighted"

DATASET_PATH = "Datasets/processed/risk_dataset"

OUTPUT_DIR = "Datasets/processed/risk_features"

MAX_LENGTH = 256

# Adjust depending on available RAM.
BATCH_SIZE = 8


# ============================================================
# Load model
# ============================================================

def load_model():

    print("Loading XLM-R model...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    model.eval()

    return tokenizer, model


# ============================================================
# Extract features
# ============================================================

def extract_features(
    texts,
    tokenizer,
    model,
    split_name
):

    all_probabilities = []
    all_embeddings = []
    all_predictions = []

    total = len(texts)

    print()
    print(f"Extracting features for {split_name}...")
    print(f"Total examples: {total}")

    for start in range(0, total, BATCH_SIZE):

        end = min(
            start + BATCH_SIZE,
            total
        )

        batch_texts = texts[start:end]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
        )

        with torch.no_grad():

            outputs = model(
                **inputs,
                output_hidden_states=True
            )

        # ----------------------------------------------------
        # 34-class probabilities
        # ----------------------------------------------------

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )

        # ----------------------------------------------------
        # Predicted class
        # ----------------------------------------------------

        predictions = torch.argmax(
            probabilities,
            dim=-1
        )

        # ----------------------------------------------------
        # XLM-R representation
        #
        # Use token-0 representation from the final
        # hidden layer.
        # ----------------------------------------------------

        last_hidden_state = outputs.hidden_states[-1]

        embeddings = last_hidden_state[:, 0, :]

        all_probabilities.append(
            probabilities.cpu().numpy()
        )

        all_embeddings.append(
            embeddings.cpu().numpy()
        )

        all_predictions.append(
            predictions.cpu().numpy()
        )

        processed = end

        print(
            f"  Processed {processed}/{total}"
        )

    return (
        np.concatenate(all_probabilities, axis=0),
        np.concatenate(all_embeddings, axis=0),
        np.concatenate(all_predictions, axis=0),
    )


# ============================================================
# Process one split
# ============================================================

def process_split(
    dataset_split,
    split_name,
    tokenizer,
    model,
    output_dir
):

    texts = dataset_split["text"]

    probabilities, embeddings, predictions = extract_features(
        texts,
        tokenizer,
        model,
        split_name
    )

    results = {
        "texts": texts,
        "ground_truth_unfair": np.array(
            dataset_split["is_unfair"]
        ),
        "unfair_labels": dataset_split["unfair_labels"],
        "predictions": predictions,
        "probabilities": probabilities,
        "embeddings": embeddings,
    }

    output_file = (
        output_dir /
        f"{split_name}_xlmr_features.joblib"
    )

    joblib.dump(
        results,
        output_file
    )

    print(
        f"Saved {split_name} features to: "
        f"{output_file}"
    )

    print(
        f"  Embeddings shape:    {embeddings.shape}"
    )

    print(
        f"  Probabilities shape: {probabilities.shape}"
    )

    print(
        f"  Predictions shape:   {predictions.shape}"
    )

    return results


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("Loading risk dataset...")

    dataset = load_from_disk(
        DATASET_PATH
    )

    print()
    print("Dataset sizes:")

    for split in ["train", "validation", "test"]:

        print(
            f"  {split.capitalize():10s}: "
            f"{len(dataset[split])}"
        )

    # --------------------------------------------------------
    # Load XLM-R
    # --------------------------------------------------------

    tokenizer, model = load_model()

    print()
    print(
        f"Number of model labels: "
        f"{model.config.num_labels}"
    )

    # --------------------------------------------------------
    # Prepare output directory
    # --------------------------------------------------------

    output_dir = Path(
        OUTPUT_DIR
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Extract all splits
    # --------------------------------------------------------

    for split_name in [
        "train",
        "validation",
        "test"
    ]:

        process_split(
            dataset[split_name],
            split_name,
            tokenizer,
            model,
            output_dir
        )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("XLM-R FEATURE EXTRACTION COMPLETE")
    print("=" * 70)

    print(
        f"Features saved to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()