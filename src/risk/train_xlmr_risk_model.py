import numpy as np
import torch

from pathlib import Path
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
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
EPOCHS = 3

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

NUM_ATTENTION_HEADS = 8
DROPOUT = 0.2

RANDOM_STATE = 42


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(
                int(self.labels[index]),
                dtype=torch.long,
            ),
        }

        return item


# ============================================================
# Evaluation
# ============================================================

def get_predictions(model, dataloader, device):

    model.eval()

    probabilities = []
    labels = []

    with torch.no_grad():

        for batch in dataloader:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits = outputs["logits"]

            probs = torch.softmax(
                logits,
                dim=-1,
            )[:, 1]

            probabilities.extend(
                probs.cpu().numpy()
            )

            labels.extend(
                batch["labels"].numpy()
            )

    return (
        np.asarray(labels),
        np.asarray(probabilities),
    )


# ============================================================
# Find best threshold
# ============================================================

def find_best_threshold(y_true, probabilities):

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            probabilities,
        )
    )

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        /
        (
            precision[:-1]
            + recall[:-1]
            + 1e-12
        )
    )

    best_index = np.argmax(f1_scores)

    return (
        float(thresholds[best_index]),
        float(f1_scores[best_index]),
    )


# ============================================================
# Evaluate
# ============================================================

def evaluate(
    model,
    dataloader,
    device,
    threshold,
    split_name,
):

    y_true, probabilities = get_predictions(
        model,
        dataloader,
        device,
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities,
    )

    print()
    print("=" * 70)
    print(f"{split_name.upper()} RESULTS")
    print("=" * 70)

    print(f"Threshold:  {threshold:.4f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1:         {f1:.4f}")
    print(f"ROC-AUC:    {roc_auc:.4f}")

    print()
    print("Classification Report:")

    print(
        classification_report(
            y_true,
            predictions,
            target_names=[
                "Fair",
                "Unfair",
            ],
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_true,
            predictions,
        )
    )

    return {
        "y_true": y_true,
        "probabilities": probabilities,
        "predictions": predictions,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


# ============================================================
# Main
# ============================================================

def main():

    set_seed(RANDOM_STATE)

    print("=" * 70)
    print("LEGAL XLM-R + MULTI-HEAD ATTENTION RISK MODEL")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}")

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading risk dataset...")

    dataset = load_from_disk(
        DATASET_PATH
    )

    print("\nDataset sizes:")

    for split in [
        "train",
        "validation",
        "test",
    ]:

        print(
            f"  {split.capitalize():10s}: "
            f"{len(dataset[split])}"
        )

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    print("\nLoading Legal XLM-R tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    train_dataset = RiskDataset(
        dataset["train"]["text"],
        dataset["train"]["is_unfair"],
        tokenizer,
    )

    validation_dataset = RiskDataset(
        dataset["validation"]["text"],
        dataset["validation"]["is_unfair"],
        tokenizer,
    )

    test_dataset = RiskDataset(
        dataset["test"]["text"],
        dataset["test"]["is_unfair"],
        tokenizer,
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nLoading Legal XLM-R + Attention model...")

    model = LegalXLMRAttentionRiskModel(
        model_name=MODEL_NAME,
        num_labels=2,
        num_attention_heads=NUM_ATTENTION_HEADS,
        dropout=DROPOUT,
    )

    model.to(device)

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    train_labels = np.asarray(
        dataset["train"]["is_unfair"]
    )

    fair_count = np.sum(
        train_labels == 0
    )

    unfair_count = np.sum(
        train_labels == 1
    )

    class_weights = torch.tensor(
        [
            1.0 / fair_count,
            1.0 / unfair_count,
        ],
        dtype=torch.float,
        device=device,
    )

    class_weights = (
        class_weights
        / class_weights.mean()
    )

    print("\nClass weights:")

    print(
        f"  Fair:   {class_weights[0].item():.4f}"
    )

    print(
        f"  Unfair: {class_weights[1].item():.4f}"
    )

    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = (
        len(train_loader)
        * EPOCHS
    )

    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(
                0.1 * total_steps
            ),
            num_training_steps=total_steps,
        )
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    best_validation_f1 = -1.0
    best_threshold = 0.5

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

        print()
        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        for step, batch in enumerate(
            train_loader,
            start=1,
        ):

            input_ids = batch[
                "input_ids"
            ].to(device)

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            labels = batch[
                "labels"
            ].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits = outputs["logits"]

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if step % 50 == 0:

                print(
                    f"  Step {step}/{len(train_loader)} "
                    f"Loss: {loss.item():.4f}"
                )

        average_loss = (
            total_loss
            / len(train_loader)
        )

        print(
            f"\nAverage training loss: "
            f"{average_loss:.4f}"
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        validation_labels, validation_probabilities = (
            get_predictions(
                model,
                validation_loader,
                device,
            )
        )

        threshold, validation_f1 = (
            find_best_threshold(
                validation_labels,
                validation_probabilities,
            )
        )

        print(
            f"Validation F1: "
            f"{validation_f1:.4f}"
        )

        print(
            f"Validation threshold: "
            f"{threshold:.4f}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if validation_f1 > best_validation_f1:

            best_validation_f1 = validation_f1
            best_threshold = threshold

            OUTPUT_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            torch.save(
                model.state_dict(),
                OUTPUT_DIR / "model.pt",
            )

            tokenizer.save_pretrained(
                OUTPUT_DIR
            )

            print(
                "✓ Best model saved."
            )

    # ========================================================
    # Load best model
    # ========================================================

    print()
    print("=" * 70)
    print("LOADING BEST MODEL")
    print("=" * 70)

    model.load_state_dict(
        torch.load(
            OUTPUT_DIR / "model.pt",
            map_location=device,
        )
    )

    # --------------------------------------------------------
    # Save configuration
    # --------------------------------------------------------

    joblib_data = {
        "threshold": best_threshold,
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "num_attention_heads": NUM_ATTENTION_HEADS,
    }

    import joblib

    joblib.dump(
        joblib_data,
        OUTPUT_DIR / "config.joblib",
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    evaluate(
        model,
        validation_loader,
        device,
        best_threshold,
        "validation",
    )

    # --------------------------------------------------------
    # Final test
    # --------------------------------------------------------

    test_results = evaluate(
        model,
        test_loader,
        device,
        best_threshold,
        "test",
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    joblib.dump(
        {
            "texts": dataset["test"]["text"],
            "y_true": test_results["y_true"],
            "probabilities": test_results[
                "probabilities"
            ],
            "predictions": test_results[
                "predictions"
            ],
        },
        OUTPUT_DIR / "test_predictions.joblib",
    )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Model:     {OUTPUT_DIR / 'model.pt'}"
    )

    print(
        f"Threshold: {best_threshold:.4f}"
    )


if __name__ == "__main__":
    main()