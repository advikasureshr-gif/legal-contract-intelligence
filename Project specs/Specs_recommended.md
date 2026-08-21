# Multilingual AI-Based Legal Contract Risk Analysis Architecture

This repository outlines the end-to-end technical architecture and pipeline stages for an AI-driven legal contract analysis system capable of processing multilingual legal documents (English, Hindi, and Tamil), segmenting clauses, predicting risk levels, generating explanations, and presenting structured outputs to end-users.

---

## 🛠️ Pipeline Overview

| Pipeline Stage | Feasible Methods / Techniques | Recommendation |
| :--- | :--- | :--- |
| **1. Document Extraction** | Apache Tika, PyMuPDF, pdfplumber, pdfminer.six, python-docx, Tesseract OCR | **Apache Tika** or **PyMuPDF** + **python-docx** |
| **2. Language Detection** | fastText, Lingua, CLD3, langdetect, langid.py | **fastText** (high accuracy & fast multilingual support) |
| **3. Text Preprocessing** | Unicode normalization, whitespace/header/footer removal, sentence segmentation, tokenization | **IndicNLP Library** (Hindi/Tamil) + **spaCy/NLTK** (English) |
| **4. Clause Segmentation** | Regex numbering patterns, rule-based heading detection, Transformer sequence labeling (NER), Layout-aware parsing | Start with **Rule-based + Regex**, fallback to sequence labeling |
| **5. Transformer NLP Model** | LegalBERT, InLegalBERT, XLM-RoBERTa, mBERT, IndicBERT v2 | **InLegalBERT** (English/Indian Legal) + **XLM-RoBERTa / IndicBERT v2** (Multilingual) |
| **6. Clause Classification** | Multi-class / Multi-label classification, Span-based QA (CUAD style) | Fine-tuned **LegalBERT / InLegalBERT** on CUAD dataset |
| **7. Risk Assessment** | Rule-based scoring, Transformer classifier, Ordinal classification, Hybrid | **Hybrid Approach** (Rules + Transformer Classification) |
| **8. Explanation Generation**| Prompt engineering, RAG, Fine-tuned Llama 3 / Gemma 2, T5/BART summarization | **Llama 3 / Gemma 2 + RAG** |
| **9. Risk Report / UI** | Flask, FastAPI, Streamlit, React + FastAPI, Django | **FastAPI** backend + **React** frontend (or **Streamlit** for rapid prototyping) |

---

## 📋 Detailed Stage Breakdowns

### Stage 1: Document Extraction
* **Inputs:** `PDF`, `DOCX`, `TXT`
* **Techniques:** Apache Tika, PyMuPDF, pdfplumber, pdfminer.six, Tesseract OCR (for scanned PDFs).
* **Output:** Cleaned raw text.

### Stage 2: Language Detection
* **Techniques:** fastText, Lingua, CLD3, langdetect, langid.py.
* **Outputs:** `English`, `Hindi`, `Tamil`.
* **Design Decisions:** Evaluate language at the **Document-level**, **Paragraph-level**, or **Sentence-level** depending on mixed-language contract structures.

### Stage 3: Text Preprocessing
* **Document Cleaning:** Header/footer stripping, page number removal, noise reduction.
* **Normalization:** Unicode normalization, lowercasing (English), whitespace trimming.
* **Tokenization & Sentence Splitting:**
  * **English:** spaCy / NLTK
  * **Hindi & Tamil:** IndicNLP Library

### Stage 4: Clause Segmentation
* **Techniques:**
  * **Rule-based / Regex:** Identification of structured patterns (`1.`, `1.1`, `ARTICLE IV`, `SECTION 3`).
  * **Heading Detection:** Semantic keyword identification (e.g., *Termination*, *Confidentiality*, *Payment*).
  * **Transformer sequence-labeling:** Treat clause boundary detection as a token classification task.
* **Recommendation:** Begin with rule-based regex patterns and structural heading detection, as legal contracts typically maintain standardized formatting.

### Stage 5: Transformer NLP Models
* **InLegalBERT:** Tailored for Indian and English legal text processing.
* **LegalBERT:** Domain-adapted BERT for standard English legal documents.
* **IndicBERT v2:** Optimized for Indic languages (Hindi, Tamil).
* **XLM-RoBERTa / mBERT:** Multilingual foundation backbones.

### Stage 6: Clause Classification
* **Input:** Individual clause text segment.
* **Target Categories:** *Termination*, *Payment*, *Liability*, *Confidentiality*, *Warranty*, *Indemnification*, etc.
* **Methods:**
  * Multi-class or Multi-label Transformer classification.
  * Fine-tuned span extraction using the **CUAD** (Contract Understanding Atticus Dataset).

### Stage 7: Risk Assessment
* **Option 1 (Rule-based):** Hardcoded logic mapping (e.g., *Unlimited Liability* -> **High Risk**).
* **Option 2 (Transformer Classifier):** Dedicated classifier predicting risk level (**Low / Medium / High**).
* **Option 3 (Hybrid - Recommended):** Combines business rules with Transformer-based semantic classification for robustness and domain precision.

### Stage 8: Explanation Generation
* **Methods:**
  * **Prompt Engineering:** Direct clause framing sent to an LLM.
  * **RAG (Retrieval-Augmented Generation):** Retrieve relevant legal knowledge and statutes to ground the LLM's explanation.
  * **Fine-Tuned LLMs:** Llama 3, Gemma 2, or T5/BART models trained on legal reasoning.
* **Sample Output:**
  ```text
  Risk: High
  Reason: The clause allows unilateral termination without prior notice or cure period.
### Stage 9: User Interface & API
* **Backend:** FastAPI / Flask for serving model inferences.
* **Frontend:** React for production deployment or Streamlit for rapid MVP prototyping.

---

## 💡 Architecture Refinement: Multi-Task Architecture

A standard implementation runs tasks sequentially:

```text
[Segmented Clause] --> [Clause Classification] --> [Risk Assessment]
Proposed Multi-Task Learning Design
Instead of running two sequential models, leverage a Multi-Task Transformer Head. A single shared encoder produces the clause embedding, which feeds into two simultaneous prediction heads:

Plaintext
                     +--> Clause Type Head (Classification)
                     |
[Segmented Clause] --> [Transformer Backbone]
                     |
                     +--> Risk Level Head (Low / Medium / High)
```
### Key Advantages:

- Efficiency: Reduces inference latency and GPU memory requirements by half.
- Shared Representation: Sentence representation learning benefits from optimizing for both taxonomy placement and risk detection simultaneously.
