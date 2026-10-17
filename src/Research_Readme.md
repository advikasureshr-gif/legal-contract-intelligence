# Multilingual AI-Based Legal Contract Risk Analyzer

A multilingual legal contract analysis system that combines legal-domain transformer models, clause classification, and unfair-term risk detection to automatically analyze contractual clauses.

The system extracts and segments clauses from PDF/DOCX contracts, identifies their clause types, predicts whether they are **FAIR or UNFAIR**, and provides risk probabilities and optional explanations in English, Hindi, and Tamil.

## Overview

Legal contracts contain complex clauses that can be difficult to review manually. This project develops an NLP-based pipeline for automated contract analysis with two major tasks:

1. **Clause Classification**: identify the type/category of a contractual clause.
2. **Risk Classification**: determine whether a clause is **FAIR or UNFAIR**.

The system uses **Legal XLM-RoBERTa** for legal-domain contextual representations and compares three approaches for unfair-term detection:

- **TF-IDF + Logistic Regression**
- **Legal XLM-R + Logistic Regression**
- **Legal XLM-R + Multi-Head Self-Attention + Attention Pooling**

The third architecture is the proposed model.

## System Workflow

```text
Contract (PDF / DOCX)
        │
        ▼
Document Extraction
        │
        ▼
Preprocessing
        │
        ▼
Clause Segmentation
        │
   ┌────┴─────────────┐
   ▼                  ▼
Clause              Risk
Classification      Classification
   │                  │
   ▼                  ▼
Legal XLM-R         Risk Models
(34 classes)          │
                ┌─────┼──────────────┐
                ▼     ▼              ▼
             TF-IDF  Legal        Legal XLM-R
              + LR   XLM-R + LR   + MHA
                                     │
                                     ▼
                             Attention Pooling
                                     │
                                     ▼
                               FAIR / UNFAIR
```

## Key Features

- PDF and DOCX contract processing
- Conservative legal-text preprocessing
- Automatic clause segmentation
- CUAD-based clause classification
- Legal-domain multilingual transformer
- Binary FAIR/UNFAIR risk classification
- Three-model risk comparison
- Task-specific multi-head self-attention
- Learned attention pooling
- Validation-based classification threshold
- Risk probability estimation
- Optional attention information
- Multilingual clause explanations
- English, Hindi, and Tamil explanation support
- Web-based contract analysis interface

## Document Processing

Contracts can be uploaded as:

- `.pdf`
- `.docx`

### PDF Processing

PDF text is extracted page-by-page using **PyMuPDF**.

### DOCX Processing

DOCX text is extracted using **python-docx**.

The system also detects the document language using **langdetect**.

## Text Preprocessing

The preprocessing pipeline is intentionally conservative because legal text contains meaningful punctuation, numbering, dates, percentages, currency values, and negations.

The preprocessing stage performs:

- Unicode normalization
- OCR artifact removal
- HTML/XML tag removal
- Control-character removal
- Whitespace normalization
- Page-number artifact removal

Legally meaningful information such as the following is preserved:

- `Section 4.2`
- `30 days`
- `$10,000`
- `50%`
- `shall not`

## Clause Segmentation

The processed document is divided into individual contractual clauses.

The segmentation system recognizes structural patterns including:

- Articles
- Sections
- Numeric clause identifiers
- Nested numbering
- Recitals
- Alphabetic lists
- Roman-numbered lists
- Headings

Heuristics are used to avoid incorrectly interpreting addresses, measurements, and ordinary numbers as clause boundaries.

Each clause retains information such as:

- `clause_id`
- `section`
- `page`
- `language`
- `original_text`
- `processed_text`

## Clause Classification

The clause classification component is trained using **CUAD (Contract Understanding Atticus Dataset)**.

The original CUAD annotations are used to construct the clause classification dataset.

The `PARTIES` category is removed because it represents document/entity information rather than a useful contractual clause type for this application.

### OTHER Clause Type

CUAD does not provide an explicit `OTHER` class.

Therefore, the project constructs `OTHER` examples from the CUAD contracts themselves.

The process is:

```text
CUAD Contracts
      │
      ▼
CUAD Annotated Answer Spans
      │
      ├── Known Clause Categories
      │
      └── Remaining Contract Text
              │
              ▼
       Candidate Text Segments
              │
              ▼
     Remove Overlapping Spans
              │
              ▼
        Filter / Deduplicate
              │
              ▼
            OTHER
```

The `OTHER` class is therefore CUAD-derived, rather than an arbitrary collection of external negative examples.

> **Note:** `OTHER` is a constructed class, not an original human-annotated CUAD category.

### Clause Classification Model

The clause classifier uses:

```text
Legal XLM-RoBERTa
        │
        ▼
34-Class Classification
```

The pretrained model is:

```text
joelniklaus/legal-xlm-roberta-base
```

The classifier is fine-tuned using class-weighted Cross Entropy Loss to address class imbalance.

The clause dataset is split at the contract level, ensuring that clauses from the same contract do not appear across training, validation, and test sets.

## Unfair-Term Risk Classification

For risk detection, the project uses the **UNFAIR-ToS** dataset.

The original annotations are converted into a binary classification task:

```text
Annotations present      →  UNFAIR
No unfairness annotations →  FAIR
```

The resulting dataset contains separate training, validation, and test sets. The original dataset splits are preserved.

## Risk Models

Three models are implemented and evaluated on the UNFAIR-ToS test set.

### Model 1: TF-IDF + Logistic Regression

The classical baseline uses:

```text
Contract Clause
      │
      ▼
TF-IDF
      │
      ▼
Unigrams + Bigrams
      │
      ▼
Logistic Regression
      │
      ▼
FAIR / UNFAIR
```

Configuration includes:

- Maximum 50,000 TF-IDF features
- Unigrams and bigrams
- Sublinear TF
- Balanced class weights
- Validation-based threshold selection

### Model 2: Legal XLM-R + Logistic Regression

The second approach uses contextual representations from Legal XLM-R.

```text
Contract Clause
      │
      ▼
Legal XLM-R
      │
      ▼
Contextual Features
      │
      ▼
Logistic Regression
      │
      ▼
FAIR / UNFAIR
```

This provides a comparison between a classical lexical representation and a legal-domain transformer representation.

### Model 3: Legal XLM-R + Multi-Head Attention + Attention Pooling

The proposed model extends Legal XLM-R with an additional task-specific attention mechanism.

```text
Input Clause
     │
     ▼
Legal XLM-R
     │
     ▼
Token Representations
     │
     ▼
8-Head Multi-Head
Self-Attention
     │
     ▼
Residual Connection
     │
     ▼
Layer Normalization
     │
     ▼
Learned Attention Pooling
     │
     ▼
Dropout
     │
     ▼
Linear Classifier
     │
     ▼
FAIR / UNFAIR
```

#### Multi-Head Self-Attention

An additional 8-head self-attention layer is applied to the contextual representations produced by Legal XLM-R.

This provides a task-specific attention mechanism for learning relationships between tokens relevant to unfairness classification.

#### Residual Connection and LayerNorm

The attention output is combined with the original token representations using a residual connection and then normalized using LayerNorm.

#### Learned Attention Pooling

The model learns an attention score for each token representation. The resulting weights are normalized and used to compute a weighted representation of the entire clause.

```text
Token Representations
        │
        ▼
Attention Scores
        │
        ▼
Softmax Weights
        │
        ▼
Weighted Sum
        │
        ▼
Clause Representation
```

The resulting representation is passed to the final classifier.

## Training

The proposed model uses:

| Parameter | Value |
|-----------|-------|
| Base model | Legal XLM-R |
| Maximum sequence length | 256 |
| Attention heads | 8 |
| Dropout | 0.2 |
| Epochs | 3 |
| Batch size | 8 |
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Random seed | 42 |
| Loss | Weighted Cross Entropy |

Class weighting is used to account for the imbalance between FAIR and UNFAIR examples.

The classification threshold is selected using the validation set by maximizing validation F1.

## Evaluation

All three risk models are evaluated on the held-out UNFAIR-ToS test set.

The following metrics are reported:

- Accuracy
- Error Rate
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

The error rate is calculated as:

```text
Error Rate = 1 - Accuracy
```

## Risk Engine

The trained models are integrated into a `RiskEngine`.

### Clause Classification

```text
Clause
  ↓
Legal XLM-R Clause Classifier
  ↓
Clause Type + Confidence
```

### Risk Classification

```text
Clause
  ↓
Legal XLM-R
  ↓
Multi-Head Attention
  ↓
Attention Pooling
  ↓
UNFAIR Probability
  ↓
Validation-Selected Threshold
  ↓
FAIR / UNFAIR
```

The final output contains:

```json
{
  "text": "...",
  "clause_type": "...",
  "clause_confidence": 0.0,
  "risk_label": "UNFAIR",
  "risk_probability": 0.0,
  "risk_threshold": 0.0
}
```

The engine can optionally return token-level attention weights and multi-head attention weights.

## Multilingual Explanation

The application includes an explanation feature that uses a generative language model to explain an analyzed clause in the requested language.

Supported explanation languages include:

- English
- Hindi
- Tamil

The explanation is constrained to the supplied clause and system analysis to avoid introducing unsupported contractual facts or obligations.

## Web Application

The complete pipeline is integrated into a web application.

```text
Upload Contract
      ↓
Process Document
      ↓
Segment Clauses
      ↓
Classify Clause Type
      ↓
Calculate Risk
      ↓
Display Results
      ↓
Optional Explanation
```

The interface presents:

- Clause text
- Clause category
- Classification confidence
- Risk label
- Risk probability
- Risk threshold
- Optional explanation

## Model Comparison

| Model | Representation | Classifier |
|-------|----------------|------------|
| TF-IDF + LR | TF-IDF lexical features | Logistic Regression |
| Legal XLM-R + LR | Legal XLM-R contextual features | Logistic Regression |
| Legal XLM-R + MHA + Attention Pooling | Legal XLM-R + task-specific attention | Neural classifier |

All three models are evaluated on the same FAIR/UNFAIR task using the held-out UNFAIR-ToS test set.

## Technologies Used

**Programming**

- Python
- JavaScript

**NLP and Machine Learning**

- PyTorch
- Hugging Face Transformers
- scikit-learn
- Hugging Face Datasets
- XLM-RoBERTa

**Document Processing**

- PyMuPDF
- python-docx
- langdetect

**Data Processing**

- NumPy
- Pandas
- Joblib

**Application**

- HTML
- CSS
- JavaScript
- Python

## Project Structure

```text
project/
│
├── preprocessing.py
├── segmenter.py
├── server.py
├── pipeline.py
├── app.js
│
├── prepare_risk_dataset.py
├── train_baseline_risk_model.py
├── extract_xlmr_features.py
├── train_xlmr_risk_model.py
├── xlmr_attention_model.py
├── train_xlmr_attention.py
├── test_xlmr_attention.py
├── risk_engine.py
├── compare_models.py
│
├── Datasets/
│   ├── processed/
│   │   ├── clause_splits/
│   │   └── risk_dataset/
│   └── unfair_tos/
│
├── models/
│   ├── clause_classifier_weighted/
│   ├── risk_baseline/
│   ├── risk_xlmr/
│   └── risk_xlmr_attention/
│
└── README.md
```

## Results

### Proposed Model

Current test-set results for Legal XLM-R + Multi-Head Attention + Attention Pooling:

| Metric | Result |
|--------|--------|
| Accuracy | 96.76% |
| Error Rate | 3.24% |
| Precision | 87.50% |
| Recall | 81.40% |
| F1 Score | 84.34% |
| ROC-AUC | 97.68% |

## References

1. Hendrycks et al., "CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review."
2. Niklaus et al., "MultiLegalPile: A 689GB Multilingual Legal Corpus."
3. Vaswani et al., "Attention Is All You Need."
4. Yang et al., "Hierarchical Attention Networks for Document Classification."
5. Drawzeski et al., "A Corpus for Multilingual Analysis of Online Terms of Service."
6. Lagioia et al., "AI in Search of Unfairness in Consumer Contracts: The Terms of Service Landscape."
