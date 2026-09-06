import joblib
import numpy as np

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ============================================================
# Configuration
# ============================================================

FEATURE_DIR = Path("Datasets/processed/risk_features")

OUTPUT_DIR = Path("models/risk_xlmr")

RANDOM_STATE = 42


# ============================================================
# Load features
# ============================================================

def load_features(split_name):

    file_path = (
        FEATURE_DIR /
        f"{split_name}_xlmr_features.joblib"
    )

    print(f"Loading {split_name} features...")

    data = joblib.load(file_path)

    embeddings = np.asarray(
        data["embeddings"]
    )

    probabilities = np.asarray(
        data["probabilities"]
    )

    labels = np.asarray(
        data["ground_truth_unfair"]
    )

    # Combine:
    # 768-dimensional XLM-R embedding
    # +
    # 34 clause-class probabilities
    X = np.concatenate(
        [embeddings, probabilities],
        axis=1
    )

    y = labels.astype(int)

    print(f"  Features shape: {X.shape}")
    print(f"  Labels shape:   {y.shape}")
    print(
        f"  Fair:           {np.sum(y == 0)}"
    )
    print(
        f"  Unfair:         {np.sum(y == 1)}"
    )

    return X, y


# ============================================================
# Find best threshold using validation set
# ============================================================

def find_best_threshold(y_true, probabilities):

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities
    )

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (
            precision[:-1]
            + recall[:-1]
            + 1e-8
        )
    )

    best_index = np.argmax(f1_scores)

    best_threshold = thresholds[best_index]

    best_f1 = f1_scores[best_index]

    return best_threshold, best_f1


# ============================================================
# Evaluate model
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    threshold,
    split_name
):

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y,
        probabilities
    )

    print()
    print("=" * 70)
    print(f"{split_name.upper()} RESULTS")
    print("=" * 70)

    print(
        f"Threshold:  {threshold:.4f}"
    )

    print(
        f"Precision:  {precision:.4f}"
    )

    print(
        f"Recall:     {recall:.4f}"
    )

    print(
        f"F1:         {f1:.4f}"
    )

    print(
        f"ROC-AUC:    {roc_auc:.4f}"
    )

    print()
    print("Classification Report:")
    print(
        classification_report(
            y,
            predictions,
            target_names=["Fair", "Unfair"],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y,
            predictions
        )
    )

    return {
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

    print("=" * 70)
    print("XLM-R RISK CLASSIFIER")
    print("=" * 70)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    X_train, y_train = load_features(
        "train"
    )

    X_validation, y_validation = load_features(
        "validation"
    )

    X_test, y_test = load_features(
        "test"
    )

    # --------------------------------------------------------
    # Train classifier
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING RISK CLASSIFIER")
    print("=" * 70)

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE
    )

    model.fit(
        X_train,
        y_train
    )

    print("Training complete.")

    # --------------------------------------------------------
    # Validation threshold tuning
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TUNING DECISION THRESHOLD")
    print("=" * 70)

    validation_probabilities = (
        model.predict_proba(X_validation)[:, 1]
    )

    best_threshold, validation_f1 = (
        find_best_threshold(
            y_validation,
            validation_probabilities
        )
    )

    print(
        f"Best validation threshold: "
        f"{best_threshold:.4f}"
    )

    print(
        f"Validation F1 at best threshold: "
        f"{validation_f1:.4f}"
    )

    # --------------------------------------------------------
    # Evaluate validation
    # --------------------------------------------------------

    evaluate_model(
        model,
        X_validation,
        y_validation,
        best_threshold,
        "validation"
    )

    # --------------------------------------------------------
    # Evaluate TEST
    # --------------------------------------------------------

    test_results = evaluate_model(
        model,
        X_test,
        y_test,
        best_threshold,
        "test"
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = (
        OUTPUT_DIR /
        "risk_model.joblib"
    )

    threshold_path = (
        OUTPUT_DIR /
        "risk_threshold.joblib"
    )

    predictions_path = (
        OUTPUT_DIR /
        "test_predictions.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    joblib.dump(
        best_threshold,
        threshold_path
    )

    joblib.dump(
        {
            "y_true": y_test,
            "probabilities": test_results["probabilities"],
            "predictions": test_results["predictions"],
        },
        predictions_path
    )

    print()
    print("=" * 70)
    print("MODEL SAVED")
    print("=" * 70)

    print(
        f"Risk model:   {model_path}"
    )

    print(
        f"Threshold:    {threshold_path}"
    )

    print(
        f"Predictions:  {predictions_path}"
    )

    print()
    print("XLM-R RISK CLASSIFIER COMPLETE")


if __name__ == "__main__":
    main()