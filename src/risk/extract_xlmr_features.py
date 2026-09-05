from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from pathlib import Path
import joblib


# ============================================================
# Configuration
# ============================================================

# This will point to the trained XLM-R model once it is
# transferred from Computer 1.
MODEL_PATH = "models/clause_classifier"

DATASET_PATH = "Datasets/processed/risk_dataset"

OUTPUT_DIR = "Datasets/processed/risk_features"

MAX_LENGTH = 256

# Start small while testing.
# Later we will process the complete splits.
TEST_EXAMPLES = 20


# ============================================================
# Load model
# ============================================================

def load_model():

    print("Loading XLM-R model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    model.eval()

    return tokenizer, model


# ============================================================
# Extract features
# ============================================================

def extract_features(texts, tokenizer, model):

    all_probabilities = []
    all_embeddings = []
    all_predictions = []

    for i, text in enumerate(texts):

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
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
        )[0]

        # ----------------------------------------------------
        # Predicted class
        # ----------------------------------------------------

        prediction = torch.argmax(
            probabilities
        ).item()

        # ----------------------------------------------------
        # XLM-R representation
        #
        # Use the CLS/token-0 representation from the final
        # hidden layer.
        # ----------------------------------------------------

        last_hidden_state = outputs.hidden_states[-1]

        embedding = last_hidden_state[:, 0, :][0]

        all_probabilities.append(
            probabilities.cpu().numpy()
        )

        all_embeddings.append(
            embedding.cpu().numpy()
        )

        all_predictions.append(prediction)

        if (i + 1) % 5 == 0:
            print(f"  Processed {i + 1}/{len(texts)}")

    return (
        np.array(all_probabilities),
        np.array(all_embeddings),
        np.array(all_predictions),
    )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("Loading risk dataset...")

    dataset = load_from_disk(DATASET_PATH)

    test_data = dataset["test"].select(
        range(min(TEST_EXAMPLES, len(dataset["test"])))
    )

    texts = test_data["text"]

    print(f"Using {len(texts)} examples for inspection.")

    # --------------------------------------------------------
    # Load XLM-R
    # --------------------------------------------------------

    tokenizer, model = load_model()

    print(
        f"Number of model labels: "
        f"{model.config.num_labels}"
    )

    print()

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    probabilities, embeddings, predictions = extract_features(
        texts,
        tokenizer,
        model
    )

    # --------------------------------------------------------
    # Inspect
    # --------------------------------------------------------

    id2label = model.config.id2label

    print()
    print("=" * 70)
    print("XLM-R FEATURE INSPECTION")
    print("=" * 70)

    for i in range(len(texts)):

        predicted_index = predictions[i]
        predicted_label = id2label[predicted_index]

        top_indices = np.argsort(
            probabilities[i]
        )[::-1][:5]

        print()
        print(f"EXAMPLE {i + 1}")
        print("-" * 70)

        print("Text:")
        print(texts[i][:500])

        print()

        print(
            "UNFAIR-ToS labels:",
            test_data[i]["unfair_labels"]
        )

        print()

        print(
            "XLM-R predicted label:",
            predicted_label
        )

        print()

        print("Top 5 probabilities:")

        for index in top_indices:

            print(
                f"  {id2label[index]:45s}"
                f" {probabilities[i][index]:.4f}"
            )

        print()

        print(
            "Embedding dimension:",
            embeddings[i].shape
        )

    # --------------------------------------------------------
    # Save inspection results
    # --------------------------------------------------------

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    results = {
        "texts": texts,
        "ground_truth_labels": test_data["unfair_labels"],
        "predictions": predictions,
        "probabilities": probabilities,
        "embeddings": embeddings,
    }

    output_file = (
        output_dir /
        "xlmr_feature_inspection.joblib"
    )

    joblib.dump(
        results,
        output_file
    )

    print()
    print("=" * 70)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 70)

    print(
        f"Saved to: {output_file}"
    )


if __name__ == "__main__":
    main()