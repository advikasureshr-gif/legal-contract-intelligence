from pathlib import Path

import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# Configuration
# ============================================================

CLAUSE_MODEL_PATH = "models/clause_classifier_weighted"

RISK_MODEL_PATH = "models/risk_baseline/risk_model.joblib"
TFIDF_PATH = "models/risk_baseline/tfidf_vectorizer.joblib"
THRESHOLD_PATH = "models/risk_baseline/risk_threshold.joblib"


# ============================================================
# Risk Engine
# ============================================================

class RiskEngine:

    def __init__(self):
        print("Loading Risk Engine...")

        # ----------------------------------------------------
        # Load XLM-R clause classifier
        # ----------------------------------------------------

        self.tokenizer = AutoTokenizer.from_pretrained(
            CLAUSE_MODEL_PATH
        )

        self.clause_model = AutoModelForSequenceClassification.from_pretrained(
            CLAUSE_MODEL_PATH
        )

        self.clause_model.eval()

        # ----------------------------------------------------
        # Load TF-IDF risk model
        # ----------------------------------------------------

        self.vectorizer = joblib.load(TFIDF_PATH)

        self.risk_model = joblib.load(RISK_MODEL_PATH)

        self.risk_threshold = joblib.load(THRESHOLD_PATH)

        # ----------------------------------------------------
        # Load clause label mapping
        # ----------------------------------------------------

        label_mapping_path = Path(CLAUSE_MODEL_PATH) / "label_mapping.json"

        if label_mapping_path.exists():
            import json

            with open(label_mapping_path, "r", encoding="utf-8") as f:
                self.label_mapping = json.load(f)

            self.id_to_label = {
                int(k): v
                for k, v in self.label_mapping["id2label"].items()
            }

        else:
            self.id_to_label = {
                i: str(i)
                for i in range(self.clause_model.config.num_labels)
            }

        print("Risk Engine loaded successfully.")

    # ========================================================
    # Clause Classification
    # ========================================================

    def classify_clause(self, text):

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )

        with torch.no_grad():
            outputs = self.clause_model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

        predicted_id = int(
            torch.argmax(probabilities).item()
        )

        clause_label = self.id_to_label.get(
            predicted_id,
            str(predicted_id)
        )

        clause_confidence = float(
            probabilities[predicted_id].item()
        )

        return clause_label, clause_confidence

    # ========================================================
    # Risk Classification
    # ========================================================

    def classify_risk(self, text):

        X = self.vectorizer.transform([text])

        probability = float(
            self.risk_model.predict_proba(X)[0, 1]
        )

        is_unfair = probability >= self.risk_threshold

        if is_unfair:
            risk_label = "UNFAIR"
        else:
            risk_label = "FAIR"

        return risk_label, probability

    # ========================================================
    # Complete Analysis
    # ========================================================

    def analyze_clause(self, text):

        clause_type, clause_confidence = self.classify_clause(
            text
        )

        risk_label, risk_probability = self.classify_risk(
            text
        )

        return {
            "text": text,
            "clause_type": clause_type,
            "clause_confidence": clause_confidence,
            "risk_label": risk_label,
            "risk_probability": risk_probability,
            "risk_threshold": float(self.risk_threshold),
        }


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    engine = RiskEngine()

    test_clauses = [
        (
            "The company may terminate this agreement at any time "
            "without providing notice to the other party."
        ),
        (
            "This agreement shall remain effective for a period "
            "of three years from the effective date."
        ),
        (
            "The parties shall maintain commercially reasonable "
            "insurance coverage throughout the term of this agreement."
        ),
    ]

    print()

    print("=" * 70)
    print("RISK ENGINE - MULTI-CLAUSE TEST")
    print("=" * 70)

    for i, clause in enumerate(test_clauses, start=1):

        result = engine.analyze_clause(clause)

        print()
        print(f"CLAUSE {i}")
        print("-" * 70)

        print(f"Text:")
        print(result["text"])

        print()
        print(f"Clause Type:       {result['clause_type']}")
        print(f"Clause Confidence: {result['clause_confidence']:.4f}")

        print()
        print(f"Risk:              {result['risk_label']}")
        print(f"Risk Probability:  {result['risk_probability']:.4f}")
        print(f"Risk Threshold:    {result['risk_threshold']:.4f}")

    print()
    print("=" * 70)
    print("MULTI-CLAUSE TEST COMPLETE")
    print("=" * 70)