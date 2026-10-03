import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# ============================================================
# Prediction files
# ============================================================

MODEL_FILES = {
    "TF-IDF + Logistic Regression":
        "models/risk_baseline/test_predictions.joblib",

    "Legal XLM-R + Logistic Regression":
        "models/risk_xlmr/test_predictions.joblib",

    "Legal XLM-R + MHA + Attention Pooling":
        "models/risk_xlmr_attention/test_predictions.joblib",
}


# ============================================================
# Load predictions
# ============================================================

def load_predictions(model_name, path):

    data = joblib.load(path)

    # --------------------------------------------------------
    # TF-IDF + Logistic Regression
    # --------------------------------------------------------

    if model_name == "TF-IDF + Logistic Regression":

        y_true = np.asarray(
            data["ground_truth"]
        )

        probabilities = np.asarray(
            data["risk_probability"]
        )

        predictions = np.asarray(
            data["prediction"]
        )

    # --------------------------------------------------------
    # Legal XLM-R + Logistic Regression
    # --------------------------------------------------------

    else:

        y_true = np.asarray(
            data["y_true"]
        )

        probabilities = np.asarray(
            data["probabilities"]
        )

        predictions = np.asarray(
            data["predictions"]
        )

    return y_true, probabilities, predictions


# ============================================================
# Calculate metrics
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    predictions,
):

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    error_rate = 1.0 - accuracy

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

    cm = confusion_matrix(
        y_true,
        predictions,
    )

    return {
        "accuracy": accuracy,
        "error_rate": error_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
    }


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 110)
    print("RISK CLASSIFICATION MODEL COMPARISON")
    print("=" * 110)

    results = {}

    for model_name, path in MODEL_FILES.items():

        print(f"\nLoading: {model_name}")

        y_true, probabilities, predictions = (
            load_predictions(
                model_name,
                path,
            )
        )

        results[model_name] = calculate_metrics(
            y_true,
            probabilities,
            predictions,
        )

        print(
            f"✓ Loaded {len(y_true)} test samples"
        )

    # ========================================================
    # Results table
    # ========================================================

    print()
    print("=" * 110)
    print("TEST SET RESULTS")
    print("=" * 110)

    print(
        f"{'Model':<43}"
        f"{'Accuracy':>11}"
        f"{'Error':>10}"
        f"{'Precision':>12}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
        f"{'ROC-AUC':>11}"
    )

    print("-" * 110)

    for model_name, metrics in results.items():

        print(
            f"{model_name:<43}"
            f"{metrics['accuracy'] * 100:>10.2f}%"
            f"{metrics['error_rate'] * 100:>9.2f}%"
            f"{metrics['precision'] * 100:>11.2f}%"
            f"{metrics['recall'] * 100:>9.2f}%"
            f"{metrics['f1'] * 100:>9.2f}%"
            f"{metrics['roc_auc'] * 100:>10.2f}%"
        )

    # ========================================================
    # Confusion matrices
    # ========================================================

    print()
    print("=" * 110)
    print("CONFUSION MATRICES")
    print("=" * 110)

    for model_name, metrics in results.items():

        print(f"\n{model_name}")
        print("-" * len(model_name))
        print(metrics["confusion_matrix"])

    print()
    print("=" * 110)
    print("COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()