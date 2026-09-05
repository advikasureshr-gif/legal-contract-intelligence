import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    DataCollatorWithPadding,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = Path("models/clause_classifier_weighted")
DATA_DIR = Path("Datasets/processed/clause_splits")

MAX_LENGTH = 256


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("LEGAL XLM-ROBERTA CLAUSE CLASSIFIER - DETAILED EVALUATION")
print("=" * 70)


# ============================================================
# LOAD LABEL MAPPING
# ============================================================

print("\nLoading label mapping...")

with open(
    MODEL_DIR / "label_mapping.json",
    "r",
    encoding="utf-8"
) as f:
    mapping = json.load(f)

label2id = mapping["label2id"]
id2label = {
    int(k): v
    for k, v in mapping["id2label"].items()
}

labels = [
    id2label[i]
    for i in range(len(id2label))
]

print(f"Number of labels: {len(labels)}")


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

test_df = pd.read_csv(
    DATA_DIR / "test.csv"
)

print(f"Test samples: {len(test_df)}")


# ============================================================
# CONVERT LABELS
# ============================================================

test_df["label_id"] = test_df["label"].map(label2id)

if test_df["label_id"].isna().any():
    unknown = test_df.loc[
        test_df["label_id"].isna(),
        "label"
    ].unique()

    raise ValueError(
        f"Unknown labels found in test set: {unknown}"
    )


# ============================================================
# CREATE HUGGING FACE DATASET
# ============================================================

test_dataset = Dataset.from_pandas(
    test_df[["clause_text", "label_id"]].rename(
        columns={"label_id": "label"}
    ),
    preserve_index=False
)


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_DIR)
)

print("Tokenizer loaded.")


# ============================================================
# TOKENIZE
# ============================================================

def tokenize_function(examples):
    return tokenizer(
        examples["clause_text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )


print("\nTokenizing test dataset...")

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True,
    desc="Tokenizing test"
)

print("Tokenization complete.")


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained model...")

model = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL_DIR)
)

print("Trained model loaded.")


# ============================================================
# DATA COLLATOR
# ============================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(
    model=model,
    processing_class=tokenizer,
    data_collator=data_collator,
)


# ============================================================
# PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING TEST PREDICTIONS")
print("=" * 70)

prediction_output = trainer.predict(
    test_dataset
)

logits = prediction_output.predictions

true_labels = prediction_output.label_ids

predicted_labels = np.argmax(
    logits,
    axis=-1
)


# ============================================================
# BASIC ACCURACY
# ============================================================

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

print(f"\nTest Accuracy: {accuracy:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    true_labels,
    predicted_labels,
    labels=list(range(len(labels))),
    target_names=labels,
    digits=4,
    zero_division=0,
)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=list(range(len(labels)))
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print(cm_df.to_string())


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report_dict = classification_report(
    true_labels,
    predicted_labels,
    labels=list(range(len(labels))),
    target_names=labels,
    output_dict=True,
    zero_division=0,
)

report_df = pd.DataFrame(
    report_dict
).transpose()

report_path = (
    MODEL_DIR / "classification_report.csv"
)

report_df.to_csv(
    report_path
)

print(
    f"\nClassification report saved to: "
    f"{report_path}"
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_path = (
    MODEL_DIR / "confusion_matrix.csv"
)

cm_df.to_csv(
    cm_path
)

print(
    f"Confusion matrix saved to: "
    f"{cm_path}"
)


# ============================================================
# SAVE INDIVIDUAL PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({
    "clause_text": test_df["clause_text"],
    "true_label": [
        id2label[int(i)]
        for i in true_labels
    ],
    "predicted_label": [
        id2label[int(i)]
        for i in predicted_labels
    ],
})

prediction_df["correct"] = (
    prediction_df["true_label"]
    == prediction_df["predicted_label"]
)

prediction_path = (
    MODEL_DIR / "test_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False
)

print(
    f"Predictions saved to: "
    f"{prediction_path}"
)


# ============================================================
# SUMMARY
# ============================================================

correct = prediction_df["correct"].sum()
total = len(prediction_df)

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print(f"Correct predictions : {correct}")
print(f"Incorrect predictions: {total - correct}")
print(f"Total test samples  : {total}")
print(f"Accuracy            : {correct / total:.4f}")

print("\nFiles created:")
print("  classification_report.csv")
print("  confusion_matrix.csv")
print("  test_predictions.csv")