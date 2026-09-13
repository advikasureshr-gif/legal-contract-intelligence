import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from xlmr_attention_model import LegalXLMRAttentionRiskModel


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "joelniklaus/legal-xlm-roberta-base"

DATASET_PATH = "Datasets/processed/risk_dataset"

OUTPUT_DIR = Path("models/risk_xlmr_attention")

MAX_LENGTH = 256

BATCH_SIZE = 8

GRADIENT_ACCUMULATION_STEPS = 2

EPOCHS = 3

ENCODER_LEARNING_RATE = 2e-5

ATTENTION_LEARNING_RATE = 5e-5

WEIGHT_DECAY = 0.01

NUM_ATTENTION_HEADS = 8

DROPOUT = 0.2

RANDOM_SEED = 42

NUM_WORKERS = 2


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Device
# ============================================================

def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# Dataset
# ============================================================

class RiskDataset(torch.utils.data.Dataset):

    def __init__(self, texts, labels, tokenizer):

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):

        return len(self.texts)

    def __getitem__(self, index):

        encoding = self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

        return {
            "input_ids": torch.tensor(
                encoding["input_ids"],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                encoding["attention_mask"],
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                int(self.labels[index]),
                dtype=torch.long,
            ),
        }


# ============================================================
# Dynamic Padding
# ============================================================

def collate_batch(batch):

    max_length = max(
        len(item["input_ids"])
        for item in batch
    )

    input_ids = []
    attention_masks = []
    labels = []

    for item in batch:

        padding_length = (
            max_length
            - len(item["input_ids"])
        )

        input_ids.append(
            torch.cat(
                [
                    item["input_ids"],
                    torch.zeros(
                        padding_length,
                        dtype=torch.long,
                    ),
                ]
            )
        )

        attention_masks.append(
            torch.cat(
                [
                    item["attention_mask"],
                    torch.zeros(
                        padding_length,
                        dtype=torch.long,
                    ),
                ]
            )
        )

        labels.append(item["labels"])

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "labels": torch.stack(labels),
    }


# ============================================================
# Class Weights
# ============================================================

def calculate_class_weights(labels):

    labels = np.asarray(labels)

    class_counts = np.bincount(
        labels,
        minlength=2,
    )

    print()
    print("Class distribution:")
    print(f"  FAIR:   {class_counts[0]}")
    print(f"  UNFAIR: {class_counts[1]}")

    total = len(labels)

    weights = total / (
        2 * class_counts
    )

    weights = torch.tensor(
        weights,
        dtype=torch.float32,
    )

    print()
    print("Class weights:")
    print(f"  FAIR:   {weights[0]:.4f}")
    print(f"  UNFAIR: {weights[1]:.4f}")

    return weights


# ============================================================
# Move Batch To Device
# ============================================================

def move_batch_to_device(batch, device):

    return {
        key: value.to(device)
        for key, value in batch.items()
    }


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    model,
    dataloader,
    device,
    criterion,
    return_attention=False,
):

    model.eval()

    total_loss = 0.0

    all_probabilities = []
    all_labels = []

    attention_results = []

    with torch.no_grad():

        for batch in dataloader:

            batch = move_batch_to_device(
                batch,
                device,
            )

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_attention=return_attention,
            )

            logits = outputs["logits"]

            loss = criterion(
                logits,
                batch["labels"],
            )

            total_loss += (
                loss.item()
                * batch["labels"].size(0)
            )

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )[:, 1]

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

            all_labels.extend(
                batch["labels"].cpu().numpy()
            )

            if return_attention:

                attention_results.append(
                    {
                        "attention_weights":
                            outputs[
                                "attention_weights"
                            ].cpu(),

                        "mha_weights":
                            outputs[
                                "mha_weights"
                            ].cpu(),

                        "input_ids":
                            batch[
                                "input_ids"
                            ].cpu(),

                        "attention_mask":
                            batch[
                                "attention_mask"
                            ].cpu(),

                        "labels":
                            batch[
                                "labels"
                            ].cpu(),
                    }
                )

    average_loss = (
        total_loss
        / len(dataloader.dataset)
    )

    return (
        average_loss,
        np.asarray(all_labels),
        np.asarray(all_probabilities),
        attention_results,
    )


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    labels,
    probabilities,
    threshold,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tp = np.sum(
        (labels == 1)
        & (predictions == 1)
    )

    fp = np.sum(
        (labels == 0)
        & (predictions == 1)
    )

    fn = np.sum(
        (labels == 1)
        & (predictions == 0)
    )

    tn = np.sum(
        (labels == 0)
        & (predictions == 0)
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
    }


# ============================================================
# Threshold Tuning
# ============================================================

def find_best_threshold(
    labels,
    probabilities,
):

    best_threshold = 0.5
    best_f1 = 0.0

    thresholds = np.arange(
        0.05,
        0.96,
        0.01,
    )

    for threshold in thresholds:

        metrics = calculate_metrics(
            labels,
            probabilities,
            threshold,
        )

        if metrics["f1"] > best_f1:

            best_f1 = metrics["f1"]

            best_threshold = threshold

    return best_threshold, best_f1


# ============================================================
# Print Metrics
# ============================================================

def print_metrics(
    name,
    loss,
    metrics,
    threshold,
):

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(f"Loss:       {loss:.4f}")
    print(f"Threshold:  {threshold:.4f}")
    print(f"Precision:  {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")
    print(f"F1:         {metrics['f1']:.4f}")

    print()
    print("Confusion Matrix:")
    print(
        f"  TN: {metrics['true_negative']}  "
        f"FP: {metrics['false_positive']}"
    )
    print(
        f"  FN: {metrics['false_negative']}  "
        f"TP: {metrics['true_positive']}"
    )


# ============================================================
# Main
# ============================================================

def main():

    set_seed(RANDOM_SEED)

    print("=" * 70)
    print("LEGAL XLM-R + MULTI-HEAD ATTENTION RISK TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    print()
    print(f"Device: {device}")

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"BF16 supported: "
            f"{torch.cuda.is_bf16_supported()}"
        )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print()
    print("Loading risk dataset...")

    dataset = load_from_disk(
        DATASET_PATH
    )

    print()
    print(dataset)

    # --------------------------------------------------------
    # Extract data
    # --------------------------------------------------------

    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["is_unfair"]

    validation_texts = dataset["validation"]["text"]
    validation_labels = dataset["validation"]["is_unfair"]

    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["is_unfair"]

    print()
    print("Dataset sizes:")
    print(f"  Train:      {len(train_texts)}")
    print(f"  Validation: {len(validation_texts)}")
    print(f"  Test:       {len(test_texts)}")

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    class_weights = calculate_class_weights(
        train_labels
    ).to(device)

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    print()
    print("Loading Legal XLM-R tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = RiskDataset(
        train_texts,
        train_labels,
        tokenizer,
    )

    validation_dataset = RiskDataset(
        validation_texts,
        validation_labels,
        tokenizer,
    )

    test_dataset = RiskDataset(
        test_texts,
        test_labels,
        tokenizer,
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print("Loading Legal XLM-R + Attention model...")

    model = LegalXLMRAttentionRiskModel(
        model_name=MODEL_NAME,
        num_labels=2,
        num_attention_heads=NUM_ATTENTION_HEADS,
        dropout=DROPOUT,
    )

    model.to(device)

    print()
    print(
        f"Total parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    print(
        f"Trainable parameters: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # --------------------------------------------------------
    # Optimizer
    #
    # Separate learning rates:
    #
    # Legal XLM-R:
    #     2e-5
    #
    # New attention/classification layers:
    #     5e-5
    # --------------------------------------------------------

    encoder_parameters = list(
        model.xlmr.parameters()
    )

    attention_parameters = (
        list(model.multihead_attention.parameters())
        + list(model.attention_norm.parameters())
        + list(model.attention_pool.parameters())
        + list(model.classifier.parameters())
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder_parameters,
                "lr": ENCODER_LEARNING_RATE,
            },
            {
                "params": attention_parameters,
                "lr": ATTENTION_LEARNING_RATE,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    total_training_steps = (
        len(train_loader)
        // GRADIENT_ACCUMULATION_STEPS
        * EPOCHS
    )

    warmup_steps = int(
        0.1 * total_training_steps
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_validation_f1 = -1.0

    best_threshold = 0.5

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0

        optimizer.zero_grad()

        print()
        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        for step, batch in enumerate(
            train_loader
        ):

            batch = move_batch_to_device(
                batch,
                device,
            )

            # ------------------------------------------------
            # BF16 mixed precision
            # ------------------------------------------------

            if (
                device.type == "cuda"
                and torch.cuda.is_bf16_supported()
            ):

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):

                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )

                    loss = criterion(
                        outputs["logits"],
                        batch["labels"],
                    )

            else:

                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )

                loss = criterion(
                    outputs["logits"],
                    batch["labels"],
                )

            # ------------------------------------------------
            # Gradient accumulation
            # ------------------------------------------------

            loss = (
                loss
                / GRADIENT_ACCUMULATION_STEPS
            )

            loss.backward()

            running_loss += (
                loss.item()
                * GRADIENT_ACCUMULATION_STEPS
            )

            if (
                (step + 1)
                % GRADIENT_ACCUMULATION_STEPS
                == 0
            ):

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

                scheduler.step()

                optimizer.zero_grad()

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                (step + 1) % 50 == 0
                or step == len(train_loader) - 1
            ):

                average_loss = (
                    running_loss
                    / (step + 1)
                )

                print(
                    f"  Step "
                    f"{step + 1}/{len(train_loader)} "
                    f"| Loss: "
                    f"{average_loss:.4f}"
                )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        validation_loss, validation_y, validation_probabilities, _ = evaluate(
            model,
            validation_loader,
            device,
            criterion,
        )

        validation_threshold, validation_f1 = (
            find_best_threshold(
                validation_y,
                validation_probabilities,
            )
        )

        validation_metrics = calculate_metrics(
            validation_y,
            validation_probabilities,
            validation_threshold,
        )

        print_metrics(
            f"VALIDATION - EPOCH {epoch + 1}",
            validation_loss,
            validation_metrics,
            validation_threshold,
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if validation_f1 > best_validation_f1:

            best_validation_f1 = validation_f1

            best_threshold = validation_threshold

            torch.save(
                model.state_dict(),
                OUTPUT_DIR
                / "best_model.pt",
            )

            joblib.dump(
                best_threshold,
                OUTPUT_DIR
                / "risk_threshold.joblib",
            )

            print()
            print(
                "✓ New best model saved."
            )

    # ========================================================
    # Load Best Model
    # ========================================================

    print()
    print("=" * 70)
    print("LOADING BEST MODEL")
    print("=" * 70)

    model.load_state_dict(
        torch.load(
            OUTPUT_DIR / "best_model.pt",
            map_location=device,
            weights_only=True,
        )
    )

    # ========================================================
    # Final Validation
    # ========================================================

    validation_loss, validation_y, validation_probabilities, _ = evaluate(
        model,
        validation_loader,
        device,
        criterion,
    )

    validation_metrics = calculate_metrics(
        validation_y,
        validation_probabilities,
        best_threshold,
    )

    print_metrics(
        "FINAL VALIDATION RESULTS",
        validation_loss,
        validation_metrics,
        best_threshold,
    )

    # ========================================================
    # Final Test
    # ========================================================

    test_loss, test_y, test_probabilities, attention_results = evaluate(
        model,
        test_loader,
        device,
        criterion,
        return_attention=True,
    )

    test_metrics = calculate_metrics(
        test_y,
        test_probabilities,
        best_threshold,
    )

    print_metrics(
        "FINAL TEST RESULTS",
        test_loss,
        test_metrics,
        best_threshold,
    )

    # ========================================================
    # Save Tokenizer
    # ========================================================

    tokenizer.save_pretrained(
        OUTPUT_DIR
    )

    # ========================================================
    # Save Model Configuration
    # ========================================================

    config = {
        "model_name": MODEL_NAME,
        "num_labels": 2,
        "num_attention_heads": NUM_ATTENTION_HEADS,
        "dropout": DROPOUT,
        "max_length": MAX_LENGTH,
        "architecture": (
            "Legal XLM-R + Multi-Head Self-Attention "
            "+ Residual LayerNorm "
            "+ Learned Attention Pooling "
            "+ Linear Classifier"
        ),
    }

    with open(
        OUTPUT_DIR / "model_config.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            indent=4,
        )

    # ========================================================
    # Save Test Predictions
    # ========================================================

    test_predictions = (
        test_probabilities
        >= best_threshold
    ).astype(int)

    joblib.dump(
        {
            "y_true": test_y,
            "probabilities": test_probabilities,
            "predictions": test_predictions,
            "threshold": best_threshold,
            "metrics": test_metrics,
        },
        OUTPUT_DIR
        / "test_predictions.joblib",
    )

    # ========================================================
    # Save Attention Examples
    #
    # Store attention information for a limited number of
    # test examples so it can later be visualized/interpreted.
    # ========================================================

    attention_examples = []

    max_attention_examples = 100

    collected = 0

    for result in attention_results:

        batch_size = (
            result["input_ids"].shape[0]
        )

        for i in range(batch_size):

            if (
                collected
                >= max_attention_examples
            ):
                break

            input_ids = result[
                "input_ids"
            ][i]

            attention_mask = result[
                "attention_mask"
            ][i]

            attention_weights = result[
                "attention_weights"
            ][i]

            valid_length = int(
                attention_mask.sum().item()
            )

            input_ids = input_ids[
                :valid_length
            ]

            attention_weights = attention_weights[
                :valid_length
            ]

            tokens = tokenizer.convert_ids_to_tokens(
                input_ids.tolist()
            )

            weights = (
                attention_weights.tolist()
            )

            attention_examples.append(
                {
                    "tokens": tokens,
                    "attention_weights": weights,
                    "label": int(
                        result["labels"][i].item()
                    ),
                }
            )

            collected += 1

        if collected >= max_attention_examples:
            break

    joblib.dump(
        attention_examples,
        OUTPUT_DIR
        / "attention_examples.joblib",
    )

    # ========================================================
    # Save Training Summary
    # ========================================================

    summary = {
        "best_validation_f1": float(
            best_validation_f1
        ),
        "best_threshold": float(
            best_threshold
        ),
        "test_precision": float(
            test_metrics["precision"]
        ),
        "test_recall": float(
            test_metrics["recall"]
        ),
        "test_f1": float(
            test_metrics["f1"]
        ),
        "test_true_positive": test_metrics[
            "true_positive"
        ],
        "test_false_positive": test_metrics[
            "false_positive"
        ],
        "test_true_negative": test_metrics[
            "true_negative"
        ],
        "test_false_negative": test_metrics[
            "false_negative"
        ],
    }

    with open(
        OUTPUT_DIR / "training_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
        )

    # ========================================================
    # Complete
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print()
    print(f"Best validation F1: {best_validation_f1:.4f}")
    print(f"Best threshold:     {best_threshold:.4f}")

    print()
    print("Final TEST:")
    print(
        f"  Precision: {test_metrics['precision']:.4f}"
    )
    print(
        f"  Recall:    {test_metrics['recall']:.4f}"
    )
    print(
        f"  F1:        {test_metrics['f1']:.4f}"
    )

    print()
    print(
        f"Model saved to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()