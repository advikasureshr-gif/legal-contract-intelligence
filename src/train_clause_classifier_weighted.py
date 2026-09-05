import os
import json
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# =========================
# CONFIG
# =========================

BASE_MODEL = "joelniklaus/legal-xlm-roberta-base"

TRAIN_FILE = "Datasets/processed/clause_splits/train.csv"
VAL_FILE = "Datasets/processed/clause_splits/validation.csv"
TEST_FILE = "Datasets/processed/clause_splits/test.csv"

OUTPUT_DIR = "models/clause_classifier_weighted"

MAX_LENGTH = 256

# =========================
# LOAD DATA
# =========================

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)
test_df = pd.read_csv(TEST_FILE)

print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

# =========================
# LABEL MAPPING
# =========================

labels = sorted(train_df["label"].unique())

label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}

print("\nNumber of labels:", len(labels))

# Check for labels in validation/test that aren't in training
unseen_val = set(val_df["label"]) - set(labels)
unseen_test = set(test_df["label"]) - set(labels)

if unseen_val:
    raise ValueError(f"Unseen validation labels: {unseen_val}")

if unseen_test:
    raise ValueError(f"Unseen test labels: {unseen_test}")

train_df["label_id"] = train_df["label"].map(label2id)
val_df["label_id"] = val_df["label"].map(label2id)
test_df["label_id"] = test_df["label"].map(label2id)

# =========================
# DATASETS
# =========================

train_dataset = Dataset.from_pandas(
    train_df[["clause_text", "label_id"]],
    preserve_index=False
)

val_dataset = Dataset.from_pandas(
    val_df[["clause_text", "label_id"]],
    preserve_index=False
)

test_dataset = Dataset.from_pandas(
    test_df[["clause_text", "label_id"]],
    preserve_index=False
)

# =========================
# TOKENIZER
# =========================

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

def tokenize(batch):
    return tokenizer(
        batch["clause_text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.rename_column("label_id", "labels")
val_dataset = val_dataset.rename_column("label_id", "labels")
test_dataset = test_dataset.rename_column("label_id", "labels")

train_dataset.set_format(
    "torch",
    columns=["input_ids", "attention_mask", "labels"]
)

val_dataset.set_format(
    "torch",
    columns=["input_ids", "attention_mask", "labels"]
)

test_dataset.set_format(
    "torch",
    columns=["input_ids", "attention_mask", "labels"]
)

# =========================
# CLASS WEIGHTS
# =========================

class_counts = train_df["label_id"].value_counts().sort_index()

total_samples = len(train_df)
num_classes = len(labels)

# Square-root inverse-frequency weighting.
# This is intentionally moderate so rare classes
# are not given absurdly large weights.

class_weights = np.sqrt(
    total_samples / (num_classes * class_counts.values)
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float
)

print("\nClass weights:")
for i, label in enumerate(labels):
    print(
        f"{label:45s} "
        f"count={class_counts[i]:5d} "
        f"weight={class_weights[i]:.3f}"
    )

# =========================
# MODEL
# =========================

model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=num_classes,
    id2label=id2label,
    label2id=label2id
)

# =========================
# METRICS
# =========================

def compute_metrics(eval_pred):

    predictions, labels_true = eval_pred

    predictions = np.argmax(predictions, axis=-1)

    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            labels_true,
            predictions,
            average="macro",
            zero_division=0
        )
    )

    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            labels_true,
            predictions,
            average="weighted",
            zero_division=0
        )
    )

    accuracy = accuracy_score(
        labels_true,
        predictions
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

# =========================
# CUSTOM TRAINER
# =========================

class WeightedTrainer(Trainer):

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None
    ):

        labels = inputs.pop("labels")

        outputs = model(**inputs)

        logits = outputs.logits

        loss_function = torch.nn.CrossEntropyLoss(
            weight=class_weights.to(logits.device)
        )

        loss = loss_function(
            logits.view(-1, model.config.num_labels),
            labels.view(-1)
        )

        return (
            (loss, outputs)
            if return_outputs
            else loss
        )

# =========================
# TRAINING ARGUMENTS
# =========================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=2,

    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,

    learning_rate=2e-5,
    weight_decay=0.01,

    eval_strategy="epoch",
    save_strategy="epoch",

    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,

    logging_steps=100,

    report_to="none",

    save_total_limit=2,
)

# =========================
# TRAIN
# =========================

trainer = WeightedTrainer(
    model=model,
    args=training_args,

    train_dataset=train_dataset,
    eval_dataset=val_dataset,

    processing_class=tokenizer,

    compute_metrics=compute_metrics,
)

print("\n==============================")
print("STARTING WEIGHTED TRAINING")
print("==============================\n")

trainer.train()

# =========================
# VALIDATION
# =========================

print("\n==============================")
print("VALIDATION RESULTS")
print("==============================\n")

val_results = trainer.evaluate(
    val_dataset
)

for key, value in val_results.items():
    print(f"{key}: {value}")

# =========================
# TEST
# =========================

print("\n==============================")
print("TEST RESULTS")
print("==============================\n")

test_results = trainer.evaluate(
    test_dataset
)

for key, value in test_results.items():
    print(f"{key}: {value}")

# =========================
# SAVE MODEL
# =========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

with open(
    os.path.join(OUTPUT_DIR, "label_mapping.json"),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "label2id": label2id,
            "id2label": id2label
        },
        f,
        indent=2
    )

print("\nModel saved to:")
print(OUTPUT_DIR)