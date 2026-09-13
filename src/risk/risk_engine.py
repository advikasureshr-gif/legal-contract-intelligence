from pathlib import Path
import json

import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from risk.xlmr_attention_model import LegalXLMRAttentionRiskModel


# ============================================================
# Configuration
# ============================================================

CLAUSE_MODEL_PATH = "models/clause_classifier_weighted"

# ------------------------------------------------------------
# Legal XLM-R + Attention Risk Model
# ------------------------------------------------------------

RISK_MODEL_DIR = Path("models/risk_xlmr_attention")

RISK_MODEL_PATH = RISK_MODEL_DIR / "best_model.pt"
RISK_TOKENIZER_PATH = RISK_MODEL_DIR
RISK_THRESHOLD_PATH = RISK_MODEL_DIR / "risk_threshold.joblib"

# Base pretrained model used to construct the architecture
LEGAL_XLMR_NAME = "joelniklaus/legal-xlm-roberta-base"

MAX_LENGTH = 256

# Class mapping
FAIR_LABEL = 0
UNFAIR_LABEL = 1


# ============================================================
# Risk Engine
# ============================================================

class RiskEngine:

    def __init__(self):

        print("Loading Risk Engine...")

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Device: {self.device}")

        # ====================================================
        # Load XLM-R Clause Classifier
        # ====================================================

        print("Loading clause classifier...")

        self.clause_tokenizer = AutoTokenizer.from_pretrained(
            CLAUSE_MODEL_PATH
        )

        self.clause_model = AutoModelForSequenceClassification.from_pretrained(
            CLAUSE_MODEL_PATH
        )

        self.clause_model.to(self.device)
        self.clause_model.eval()

        # ----------------------------------------------------
        # Load clause label mapping
        # ----------------------------------------------------

        label_mapping_path = (
            Path(CLAUSE_MODEL_PATH) / "label_mapping.json"
        )

        if label_mapping_path.exists():

            with open(
                label_mapping_path,
                "r",
                encoding="utf-8"
            ) as f:

                self.label_mapping = json.load(f)

            self.id_to_label = {
                int(k): v
                for k, v in self.label_mapping["id2label"].items()
            }

        else:

            self.id_to_label = {
                i: str(i)
                for i in range(
                    self.clause_model.config.num_labels
                )
            }

        # ====================================================
        # Load Legal XLM-R + Multi-Head Attention Risk Model
        # ====================================================

        print("Loading Legal XLM-R + Attention risk model...")

        # ----------------------------------------------------
        # Risk tokenizer
        #
        # The tokenizer files were saved with the trained model.
        # ----------------------------------------------------

        self.risk_tokenizer = AutoTokenizer.from_pretrained(
            RISK_TOKENIZER_PATH
        )

        # ----------------------------------------------------
        # Reconstruct model architecture
        # ----------------------------------------------------

        self.risk_model = LegalXLMRAttentionRiskModel(
            model_name=LEGAL_XLMR_NAME,
            num_labels=2,
            num_attention_heads=8,
            dropout=0.2,
        )

        # ----------------------------------------------------
        # Load trained weights
        # ----------------------------------------------------

        print(
            f"Loading trained weights from: {RISK_MODEL_PATH}"
        )

        checkpoint = torch.load(
            RISK_MODEL_PATH,
            map_location=self.device,
            weights_only=False,
        )

        # ----------------------------------------------------
        # Support either:
        #
        # 1. torch.save(model.state_dict(), ...)
        #
        # OR
        #
        # 2. torch.save({
        #        "model_state_dict": model.state_dict()
        #    }, ...)
        # ----------------------------------------------------

        if isinstance(checkpoint, dict):

            if "model_state_dict" in checkpoint:

                state_dict = checkpoint["model_state_dict"]

            elif "state_dict" in checkpoint:

                state_dict = checkpoint["state_dict"]

            else:

                # Assume the checkpoint itself is a state dict
                state_dict = checkpoint

        else:

            raise ValueError(
                "Unsupported risk model checkpoint format."
            )

        # ----------------------------------------------------
        # Remove DataParallel prefix if present
        # ----------------------------------------------------

        cleaned_state_dict = {}

        for key, value in state_dict.items():

            if key.startswith("module."):

                key = key[len("module."):]

            cleaned_state_dict[key] = value

        # ----------------------------------------------------
        # Load weights
        # ----------------------------------------------------

        missing_keys, unexpected_keys = (
            self.risk_model.load_state_dict(
                cleaned_state_dict,
                strict=False,
            )
        )

        if missing_keys:

            print(
                "Warning: Missing risk model keys:"
            )

            for key in missing_keys:
                print(f"  {key}")

        if unexpected_keys:

            print(
                "Warning: Unexpected risk model keys:"
            )

            for key in unexpected_keys:
                print(f"  {key}")

        # ----------------------------------------------------
        # Move to device
        # ----------------------------------------------------

        self.risk_model.to(self.device)
        self.risk_model.eval()

        # ====================================================
        # Load trained risk threshold
        # ====================================================

        self.risk_threshold = float(
            joblib.load(RISK_THRESHOLD_PATH)
        )

        print(
            f"Risk threshold: {self.risk_threshold:.4f}"
        )

        print("Risk Engine loaded successfully.")

    # ========================================================
    # Clause Classification
    # ========================================================

    def classify_clause(self, text):

        inputs = self.clause_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )

        # Move inputs to device
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = self.clause_model(
                **inputs
            )

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

        return (
            clause_label,
            clause_confidence
        )

    # ========================================================
    # Risk Classification
    # ========================================================

    def classify_risk(
        self,
        text,
        return_attention=False,
    ):

        # ----------------------------------------------------
        # Tokenize clause
        # ----------------------------------------------------

        inputs = self.risk_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        # ----------------------------------------------------
        # Legal XLM-R + Multi-Head Attention
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = self.risk_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                return_attention=return_attention,
            )

        logits = outputs["logits"]

        probabilities = torch.softmax(
            logits,
            dim=-1
        )[0]

        # Probability of UNFAIR
        unfair_probability = float(
            probabilities[UNFAIR_LABEL].item()
        )

        # ----------------------------------------------------
        # Apply trained threshold
        # ----------------------------------------------------

        is_unfair = (
            unfair_probability >= self.risk_threshold
        )

        if is_unfair:

            risk_label = "UNFAIR"

        else:

            risk_label = "FAIR"

        # ----------------------------------------------------
        # Basic result
        # ----------------------------------------------------

        result = {
            "risk_label": risk_label,
            "risk_probability": unfair_probability,
            "risk_threshold": self.risk_threshold,
        }

        # ----------------------------------------------------
        # Optional attention information
        # ----------------------------------------------------

        if return_attention:

            attention_weights = (
                outputs["attention_weights"][0]
                .detach()
                .cpu()
                .tolist()
            )

            tokens = self.risk_tokenizer.convert_ids_to_tokens(
                inputs["input_ids"][0].detach().cpu().tolist()
            )

            result["attention_weights"] = attention_weights
            result["tokens"] = tokens

            # MHA weights can be large, so keep them optional
            if outputs["mha_weights"] is not None:

                result["mha_weights"] = (
                    outputs["mha_weights"][0]
                    .detach()
                    .cpu()
                    .tolist()
                )

        return result

    # ========================================================
    # Complete Analysis
    # ========================================================

    def analyze_clause(
        self,
        text,
        return_attention=False,
    ):

        # ----------------------------------------------------
        # Clause classification
        # ----------------------------------------------------

        clause_type, clause_confidence = (
            self.classify_clause(text)
        )

        # ----------------------------------------------------
        # Risk classification
        # ----------------------------------------------------

        risk_result = self.classify_risk(
            text,
            return_attention=return_attention,
        )

        # ----------------------------------------------------
        # Combine results
        # ----------------------------------------------------

        result = {
            "text": text,

            "clause_type": clause_type,
            "clause_confidence": clause_confidence,

            "risk_label": risk_result["risk_label"],
            "risk_probability": risk_result["risk_probability"],
            "risk_threshold": risk_result["risk_threshold"],
        }

        # Optional attention information
        if return_attention:

            result["attention_weights"] = (
                risk_result["attention_weights"]
            )

            result["tokens"] = (
                risk_result["tokens"]
            )

            if "mha_weights" in risk_result:

                result["mha_weights"] = (
                    risk_result["mha_weights"]
                )

        return result


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
    print("RISK ENGINE - LEGAL XLM-R + ATTENTION TEST")
    print("=" * 70)

    for i, clause in enumerate(
        test_clauses,
        start=1
    ):

        result = engine.analyze_clause(
            clause
        )

        print()
        print(f"CLAUSE {i}")
        print("-" * 70)

        print("Text:")
        print(result["text"])

        print()
        print(
            f"Clause Type:       "
            f"{result['clause_type']}"
        )

        print(
            f"Clause Confidence: "
            f"{result['clause_confidence']:.4f}"
        )

        print()
        print(
            f"Risk:              "
            f"{result['risk_label']}"
        )

        print(
            f"Risk Probability:  "
            f"{result['risk_probability']:.4f}"
        )

        print(
            f"Risk Threshold:    "
            f"{result['risk_threshold']:.4f}"
        )

    print()
    print("=" * 70)
    print("RISK ENGINE TEST COMPLETE")
    print("=" * 70)