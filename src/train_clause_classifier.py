import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "joelniklaus/legal-xlm-roberta-base"

DATA_DIR = Path("Datasets/processed/clause_splits")
OUTPUT_DIR = Path("models/clause_classifier")

MAX_LENGTH = 256


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LEGAL XLM-ROBERTA CLAUSE CLASSIFIER")
print("=" * 70)

print("\nLoading datasets...")

train_df = pd.read_csv(DATA_DIR / "train.csv")
validation_df = pd.read_csv(DATA_DIR / "validation.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

print(f"Train      : {len(train_df)}")
print(f"Validation : {len(validation_df)}")
print(f"Test       : {len(test_df)}")


# ============================================================
# CREATE LABEL MAPPING
# ============================================================

labels = sorted(train_df["label"].unique())

label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

print(f"\nNumber of labels: {len(labels)}")

print("\nLabels:")
for i, label in enumerate(labels):
    print(f"{i:2d} | {label}")


# Make sure validation/test don't contain unseen labels
for df, name in [
    (validation_df, "validation"),
    (test_df, "test"),
]:
    unseen = set(df["label"]) - set(labels)

    if unseen:
        raise ValueError(
            f"{name} contains labels not present in training data: {unseen}"
        )


# ============================================================
# CONVERT LABELS TO INTEGER IDs
# ============================================================

train_df["label"] = train_df["label"].map(label2id)
validation_df["label"] = validation_df["label"].map(label2id)
test_df["label"] = test_df["label"].map(label2id)


# ============================================================
# CONVERT TO HUGGING FACE DATASETS
# ============================================================

train_dataset = Dataset.from_pandas(
    train_df[["clause_text", "label"]],
    preserve_index=False
)

validation_dataset = Dataset.from_pandas(
    validation_df[["clause_text", "label"]],
    preserve_index=False
)

test_dataset = Dataset.from_pandas(
    test_df[["clause_text", "label"]],
    preserve_index=False
)


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Tokenizer loaded.")


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_function(examples):
    return tokenizer(
        examples["clause_text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )


print("\nTokenizing datasets...")

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    desc="Tokenizing train"
)

validation_dataset = validation_dataset.map(
    tokenize_function,
    batched=True,
    desc="Tokenizing validation"
)

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True,
    desc="Tokenizing test"
)

print("Tokenization complete.")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Legal XLM-RoBERTa model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)

print("Model loaded.")


# ============================================================
# DATA COLLATOR
# ============================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_pred):
    predictions, labels_true = eval_pred

    predictions = np.argmax(predictions, axis=-1)

    accuracy = accuracy_score(labels_true, predictions)

    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            labels_true,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            labels_true,
            predictions,
            average="weighted",
            zero_division=0,
        )
    )

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
    }


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),

    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,

    num_train_epochs=1,

    weight_decay=0.01,

    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,

    logging_steps=100,

    save_total_limit=2,

    report_to="none",
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=train_dataset,
    eval_dataset=validation_dataset,

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics,
)


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING FINE-TUNING")
print("=" * 70)

trainer.train()


# ============================================================
# VALIDATION RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION RESULTS")
print("=" * 70)

validation_results = trainer.evaluate(
    validation_dataset
)

for key, value in validation_results.items():
    if isinstance(value, float):
        print(f"{key:25s}: {value:.4f}")
    else:
        print(f"{key:25s}: {value}")


# ============================================================
# TEST RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

test_results = trainer.evaluate(
    test_dataset,
    metric_key_prefix="test"
)

for key, value in test_results.items():
    if isinstance(value, float):
        print(f"{key:25s}: {value:.4f}")
    else:
        print(f"{key:25s}: {value}")


# ============================================================
# SAVE MODEL + TOKENIZER
# ============================================================

print("\nSaving final model...")

trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

with open(
    OUTPUT_DIR / "label_mapping.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        {
            "label2id": label2id,
            "id2label": {str(k): v for k, v in id2label.items()},
        },
        f,
        indent=2,
        ensure_ascii=False,
    )

print(f"Model saved to: {OUTPUT_DIR}")

print("\n" + "=" * 70)
print("FINE-TUNING COMPLETE")
print("=" * 70)