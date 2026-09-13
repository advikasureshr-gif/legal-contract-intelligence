import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from xlmr_attention_model import LegalXLMRAttentionRiskModel


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "joelniklaus/legal-xlm-roberta-base"
DATASET_PATH = "Datasets/processed/risk_dataset"
MODEL_PATH = "models/risk_xlmr_attention/best_model.pt"

MAX_LENGTH = 256
BATCH_SIZE = 16


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# Load dataset
# ============================================================

dataset = load_from_disk(DATASET_PATH)

test_dataset = dataset["test"]

print("Test samples:", len(test_dataset))


# ============================================================
# Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# Tokenize
# ============================================================

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


test_dataset = test_dataset.map(
    tokenize,
    batched=True,
)

test_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "is_unfair",
    ],
)


# ============================================================
# DataLoader
# ============================================================

loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ============================================================
# Load model
# ============================================================

print("Loading trained model...")

model = LegalXLMRAttentionRiskModel(
    model_name=MODEL_NAME,
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
)

model.load_state_dict(checkpoint)

model.to(device)
model.eval()


# ============================================================
# Collect probabilities
# ============================================================

all_probabilities = []
all_labels = []


with torch.no_grad():

    for batch in loader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["is_unfair"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        probabilities = torch.softmax(
            outputs["logits"],
            dim=-1,
        )[:, 1]

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


probabilities = np.array(all_probabilities)
labels = np.array(all_labels)


# ============================================================
# Overall distribution
# ============================================================

print()
print("=" * 70)
print("RISK PROBABILITY DISTRIBUTION")
print("=" * 70)

print(f"Minimum:   {probabilities.min():.4f}")
print(f"25th %ile: {np.percentile(probabilities, 25):.4f}")
print(f"Median:    {np.median(probabilities):.4f}")
print(f"75th %ile: {np.percentile(probabilities, 75):.4f}")
print(f"Maximum:   {probabilities.max():.4f}")


# ============================================================
# FAIR vs UNFAIR distributions
# ============================================================

fair_probs = probabilities[labels == 0]
unfair_probs = probabilities[labels == 1]


print()
print("FAIR clauses:")
print(f"  Count:    {len(fair_probs)}")
print(f"  Mean:     {fair_probs.mean():.4f}")
print(f"  Median:   {np.median(fair_probs):.4f}")
print(f"  Min:      {fair_probs.min():.4f}")
print(f"  Max:      {fair_probs.max():.4f}")


print()
print("UNFAIR clauses:")
print(f"  Count:    {len(unfair_probs)}")
print(f"  Mean:     {unfair_probs.mean():.4f}")
print(f"  Median:   {np.median(unfair_probs):.4f}")
print(f"  Min:      {unfair_probs.min():.4f}")
print(f"  Max:      {unfair_probs.max():.4f}")


# ============================================================
# Useful probability bands
# ============================================================

print()
print("=" * 70)
print("PROBABILITY BANDS")
print("=" * 70)

bands = [
    ("0.00 - 0.19", 0.00, 0.20),
    ("0.20 - 0.39", 0.20, 0.40),
    ("0.40 - 0.59", 0.40, 0.60),
    ("0.60 - 0.79", 0.60, 0.80),
    ("0.80 - 0.92", 0.80, 0.93),
    ("0.93 - 1.00", 0.93, 1.01),
]

for name, low, high in bands:

    mask = (
        (probabilities >= low)
        & (probabilities < high)
    )

    count = mask.sum()

    if count > 0:

        unfair_count = (
            labels[mask] == 1
        ).sum()

        print(
            f"{name}: "
            f"{count:4d} samples | "
            f"{unfair_count:4d} actually UNFAIR"
        )