from datasets import load_from_disk
from pathlib import Path
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.metrics import precision_recall_curve
import numpy as np


# ============================================================
# Configuration
# ============================================================

DATASET_PATH = "Datasets/processed/risk_dataset"
OUTPUT_DIR = "models/risk_baseline"

MAX_FEATURES = 50_000
NGRAM_RANGE = (1, 2)


# ============================================================
# Load data
# ============================================================

def load_data():
    print("Loading prepared risk dataset...")

    dataset = load_from_disk(DATASET_PATH)

    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["is_unfair"]

    val_texts = dataset["validation"]["text"]
    val_labels = dataset["validation"]["is_unfair"]

    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["is_unfair"]

    print()
    print("Dataset sizes:")
    print(f"  Train:      {len(train_texts)}")
    print(f"  Validation: {len(val_texts)}")
    print(f"  Test:       {len(test_texts)}")

    return (
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        test_texts,
        test_labels,
    )


# ============================================================
# Find validation threshold
# ============================================================

def find_best_threshold(y_true, probabilities):
    """
    Find the probability threshold that gives the best
    F1 score on the validation set.

    The test set is NOT used here.
    """

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities
    )

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_index = np.argmax(f1_scores)

    best_threshold = thresholds[best_index]
    best_f1 = f1_scores[best_index]

    return best_threshold, best_f1


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    (
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        test_texts,
        test_labels,
    ) = load_data()


    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    print()
    print("Creating TF-IDF representation...")

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        lowercase=True,
        sublinear_tf=True,
    )

    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)

    print(f"  Training matrix:   {X_train.shape}")
    print(f"  Validation matrix: {X_val.shape}")
    print(f"  Test matrix:       {X_test.shape}")


    # --------------------------------------------------------
    # Train Logistic Regression
    # --------------------------------------------------------

    print()
    print("Training baseline risk model...")

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )

    model.fit(X_train, train_labels)

    print("Training complete.")


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    val_probabilities = model.predict_proba(X_val)[:, 1]

    threshold, validation_f1 = find_best_threshold(
        val_labels,
        val_probabilities
    )

    print(f"Best validation threshold: {threshold:.4f}")
    print(f"Validation F1 at threshold: {validation_f1:.4f}")


    # --------------------------------------------------------
    # Final test evaluation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    test_probabilities = model.predict_proba(X_test)[:, 1]

    test_predictions = (
        test_probabilities >= threshold
    ).astype(int)


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels,
        test_predictions,
        average="binary",
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        test_labels,
        test_probabilities
    )

    print()
    print(f"Risk threshold: {threshold:.4f}")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"F1 Score:       {f1:.4f}")
    print(f"ROC-AUC:        {roc_auc:.4f}")


    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print()
    print("Classification Report:")
    print(
        classification_report(
            test_labels,
            test_predictions,
            target_names=["Fair", "Unfair"],
            digits=4,
            zero_division=0,
        )
    )


    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            test_labels,
            test_predictions
        )
    )


    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        vectorizer,
        output_dir / "tfidf_vectorizer.joblib"
    )

    joblib.dump(
        model,
        output_dir / "risk_model.joblib"
    )

    joblib.dump(
        threshold,
        output_dir / "risk_threshold.joblib"
    )


    # --------------------------------------------------------
    # Save test predictions
    # --------------------------------------------------------

    predictions = {
        "text": test_texts,
        "ground_truth": test_labels,
        "risk_probability": test_probabilities,
        "prediction": test_predictions,
    }

    joblib.dump(
        predictions,
        output_dir / "test_predictions.joblib"
    )


    print()
    print("=" * 70)
    print("MODEL SAVED")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()