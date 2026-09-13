import joblib
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

data = joblib.load("models/risk_baseline/test_predictions.joblib")

y_true = data["ground_truth"]
probabilities = data["risk_probability"]
predictions = data["prediction"]

print("=" * 70)
print("TF-IDF + LOGISTIC REGRESSION BASELINE")
print("=" * 70)

print(f"Precision: {precision_score(y_true, predictions):.4f}")
print(f"Recall:    {recall_score(y_true, predictions):.4f}")
print(f"F1:        {f1_score(y_true, predictions):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_true, probabilities):.4f}")

print()
print("Confusion Matrix:")
print(confusion_matrix(y_true, predictions))