# Multilingual AI-Based Legal Contract Risk Analysis


## Project Overview


This project proposes a multilingual NLP pipeline for analysing legal contracts. The system extracts contractual text, detects its language, preprocesses the text, segments it into clauses, classifies each clause, identifies potential risk indicators, assigns a risk level, and provides evidence-based explanations.


The system is intended as a legal decision-support tool rather than a replacement for a qualified legal professional.


---


# Project Pipeline


```text
Contract Upload
      ↓
PDF / DOCX / Text Extraction
      ↓
Language Detection
      ↓
Multilingual Text Preprocessing
      ↓
Clause Segmentation
      ↓
XLM-RoBERTa Representation
      ↓
Clause Classification
      ↓
Risk Indicator Detection
      ↓
Risk Scoring
      ↓
Evidence Extraction & Explanation
      ↓
Risk Report / Web Interface
Models, Methods and Datasets
Step	Recommended Method / Model	Dataset / Data Source	Purpose
1. Document Extraction	PyMuPDF + python-docx	User-provided contracts	Convert PDF/DOCX documents into machine-readable text
2. Language Detection	FastText Language Identification	User contracts + MILPaC for testing	Detect the language of the contract or individual clauses
3. Text Preprocessing	Unicode normalization + whitespace and paragraph normalization	All datasets	Clean text while preserving legally meaningful information
4. Clause Segmentation	Structural rules + sentence segmentation	CUAD contracts for evaluation	Divide contracts into meaningful clauses and sub-clauses
5. Multilingual Representation	XLM-RoBERTa (XLM-R)	MILPaC + labelled clause data	Represent multilingual legal text
6. Clause Classification	Fine-tuned XLM-RoBERTa classifier	CUAD	Classify clauses into predefined legal categories
7. Legal Contextual Inference	XLM-R-based NLI model	ContractNLI / train.json	Identify entailment, contradiction, or lack of mention
8. Risk Indicator Detection	Rules + NLP-based detection	CUAD + project-specific annotations	Detect indicators such as uncapped liability and broad indemnification
9. Risk Scoring	Hybrid risk-scoring model	Project-specific risk-labelled validation data	Assign Low, Medium, or High risk
10. Evidence Extraction	Relevant clause/span extraction	CUAD annotations + model output	Identify evidence supporting a risk decision
11. Explanation	Template / controlled generation	Model outputs	Generate concise human-readable explanations
12. Cross-Language Consistency (Optional)	XLM-R embeddings + similarity/comparison rules	MILPaC + paired contract versions	Detect differences between equivalent multilingual provisions
Dataset Roles
1. MILPaC — Multilingual Legal Text
Purpose

MILPaC is used primarily for the multilingual component of the project.

It can support:

Multilingual preprocessing
Testing language handling
Multilingual representation with XLM-R
Evaluation of Indian-language legal text
Optional cross-language comparison

Conceptually:

MILPaC
   ↓
Hindi / Malayalam / Marathi / Telugu / etc.
   ↓
Language-aware preprocessing
   ↓
XLM-RoBERTa

MILPaC is not used as the primary risk-classification dataset.

Its role is to support the project's multilingual requirement.

2. CUAD — Contract Clause Understanding

CUAD is the primary dataset for legal contract clause understanding and classification.

It contains contracts with annotations corresponding to specific contractual provisions.

The data can be transformed into examples such as:

Clause Text	Category
"Either party may terminate..."	Termination
"Supplier shall indemnify..."	Indemnification
"Neither party shall disclose..."	Confidentiality
"Liability shall not exceed..."	Liability

The resulting training pipeline is:

Clause Text
     ↓
XLM-R Tokenizer
     ↓
XLM-RoBERTa
     ↓
Contextual Representation
     ↓
Classification Layer
     ↓
Clause Category + Confidence

CUAD is the primary dataset for clause classification.

3. ContractNLI / train.json — Legal Contextual Inference

The ContractNLI-style dataset contains annotations such as:

Entailment
Contradiction
NotMentioned

This supports contextual legal inference.

Conceptually:

Contract
   +
Legal Hypothesis
   ↓
NLI Model
   ↓
Entailment / Contradiction / NotMentioned

Example:

Contract:
"The agreement may be terminated with 30 days' notice."


Hypothesis:
"The agreement can be terminated with 30 days' notice."


Result:
Entailment

This dataset is useful for contextual legal reasoning, but it is not essential to the first prototype.

Risk Assessment

Risk assessment is performed after clause classification.

The system should not assign risk solely based on the clause category. Instead, it should identify specific risk indicators and combine them with contextual information and model confidence.

Example Risk Indicators
Unlimited or uncapped liability
Broad indemnification obligations
Automatic renewal with a short cancellation window
One-sided termination rights
Unusually long contractual obligations
Broad restrictions or obligations without clear limits
Ambiguous wording or undefined obligations
Conflicting terms across language versions
Absence of important protections such as a liability cap or termination condition
Risk Assessment Pipeline
Clause
   ↓
Clause Classification
   ↓
Risk Indicator Detection
   ↓
Risk Scoring
   ↓
Low / Medium / High
Example
Clause:
"Supplier shall be liable for all losses arising from any breach."


Category:
Liability


Risk Indicators:
✓ Broad "all losses" wording
✓ No visible liability cap


Risk:
HIGH
Risk Scoring Approach

A hybrid risk-scoring approach is proposed:

Risk Score =
    w1 × Model Risk Probability
  + w2 × Risk Indicator Score
  + w3 × Contextual / Contract Factors

The weights and thresholds should be validated or tuned using project-specific validation data, rather than being chosen arbitrarily.

The resulting score is mapped to:

Low
Medium
High

The risk layer does not require a separate machine-learning risk classifier for the initial prototype. A hybrid rule-based and model-assisted approach is consistent with the proposed methodology.

Final Recommended Stack
Language Detection

FastText Language Identification

Contract
   ↓
Language Detection
   ↓
Language + Confidence
Preprocessing

Python + Unicode/Text Processing

Raw Text
   ↓
Unicode Normalization
   ↓
Whitespace / Paragraph Normalization
   ↓
Header / Footer Removal
   ↓
Clean Multilingual Text

Legal punctuation, negation, and legally meaningful words should be preserved.

Clause Segmentation

Structural Rules + Multilingual Sentence Segmentation

Contract
   ↓
Heading / Number Detection
   ↓
Paragraph Boundaries
   ↓
Sentence Segmentation
   ↓
Sub-clause / List Detection
   ↓
Final Clause Units
Main NLP Model

XLM-RoBERTa (XLM-R)

XLM-R is used as the primary multilingual transformer for representing and classifying contractual clauses across supported languages.

Clause
   ↓
XLM-R Tokenizer
   ↓
XLM-RoBERTa
   ↓
Contextual Representation
   ↓
Classification Layer
   ↓
Clause Category
Optional Contextual Reasoning

XLM-R-based NLI Model

Contract
   +
Legal Statement
   ↓
NLI Model
   ↓
Entailment / Contradiction / NotMentioned

This can be added after the core clause-classification prototype.

Risk Analysis

Hybrid Rule-Based + Model-Assisted Risk Scoring

Clause Category
      +
Model Confidence
      +
Risk Indicators
      +
Contextual Factors
      ↓
Risk Score
      ↓
LOW / MEDIUM / HIGH
Explanation

Initially, explanations can use evidence extraction + controlled templates rather than relying on unrestricted generative output.

Example:

Clause: Liability


Risk: HIGH


Risk Factors:
- No explicit liability cap
- Broad "all losses" wording


Explanation:
The clause may expose the party to broad financial
liability because it does not specify a maximum
liability amount.
Dataset Priority

The datasets have different levels of importance for the project:

Core

CUAD

Used for contractual clause understanding and classification.

Multilingual Support

MILPaC

Used for multilingual legal-language processing and evaluation, particularly for Indian-language legal text.

Optional Enhancement

ContractNLI / train.json

Used for contextual legal inference through entailment, contradiction, and not-mentioned relationships.

Project-Specific

Risk-labelled validation data

Used to tune and validate the Low / Medium / High risk scoring mechanism.

Recommended Development Order

To keep the project feasible, implementation can proceed in stages:

Phase 1 — Core Pipeline
Document Extraction
       ↓
Language Detection
       ↓
Preprocessing
       ↓
Clause Segmentation
       ↓
XLM-R Clause Classification
Phase 2 — Risk Analysis
Clause Classification
       ↓
Risk Indicator Detection
       ↓
Risk Scoring
       ↓
Evidence + Explanation
Phase 3 — Multilingual Evaluation
MILPaC
   ↓
Multilingual Processing
   ↓
XLM-R Evaluation
Phase 4 — Optional Enhancement
ContractNLI
   ↓
Contextual Legal Inference
Phase 5 — Optional Research Enhancement
Equivalent Multilingual Clauses
          ↓
Cross-Language Comparison
          ↓
Potential Inconsistency Detection
Important Scope Note

The three datasets should not be treated as one combined training dataset.

Each dataset supports a different component:

MILPaC
   → Multilingual Legal Text


CUAD
   → Contract Clause Understanding


ContractNLI / train.json
   → Legal Contextual Inference


Project-Specific Data
   → Risk Scoring Validation

The overall system combines these components into a multilingual, explainable legal contract analysis pipeline.