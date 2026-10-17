# Multilingual AI-Based Legal Contract Risk Analyzer

An AI-powered web application that analyzes legal contracts, identifies contractual clause types, and flags potentially unfair terms. It helps users understand complex contract language by presenting clause-level risk assessments and optional explanations in English, Hindi, and Tamil.

## Overview

Legal contracts often contain complex terms and conditions that can be difficult to interpret. Reviewing every clause manually can be time-consuming, especially when users need to identify potentially unfair or unfavorable terms.

This project uses Natural Language Processing (NLP) and machine learning to automate parts of the contract review process. Users can upload a contract, view its extracted clauses, review clause classifications, and identify clauses flagged as potentially unfair.

The application is intended to support preliminary contract review, not replace professional legal advice.

## Features

- **Contract Upload:** Process PDF and DOCX documents.
- **Automatic Clause Extraction:** Extract text and divide contracts into individual clauses.
- **Clause Classification:** Categorize clauses into relevant contractual categories.
- **Risk Detection:** Classify clauses as FAIR or UNFAIR.
- **Risk Assessment:** Display risk probabilities and classification results.
- **Multilingual Explanations:** Generate optional explanations in English, Hindi, and Tamil.
- **Interactive Web Interface:** Upload contracts and review analysis results through a web-based interface.

## How It Works

1. Upload a legal contract in PDF or DOCX format.
2. Extract and preprocess the document text.
3. Segment the contract into individual clauses.
4. Identify the type of each clause using a legal-domain language model.
5. Analyze clauses using the risk classification model.
6. Display the clause categories, risk labels, and risk probabilities.
7. Request an optional explanation in the preferred supported language.

## Technology Stack

| Component | Technologies |
|---|---|
| Programming Languages | Python, JavaScript |
| NLP and Deep Learning | PyTorch, Hugging Face Transformers |
| Machine Learning | scikit-learn |
| Language Model | Legal XLM-RoBERTa |
| Document Processing | PyMuPDF, python-docx |
| Data Processing | NumPy, Pandas, Joblib |
| Language Detection | langdetect |
| Frontend | HTML, CSS, JavaScript |
| Backend | Python |

## Installation

### Prerequisites

- Python 3.10 or another Python version compatible with the project's dependencies
- Git
- pip
- A compatible environment for running PyTorch and Hugging Face Transformers

A GPU is recommended for faster model inference, although CPU execution may be possible depending on the available resources.

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_PROJECT_DIRECTORY>
```

Replace the placeholders with your Git repository URL and project directory.

### 2. Create a Virtual Environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

If the repository includes a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Otherwise, install the dependencies specified by the project and its model-loading scripts.

## Configuration

The application may require an API key for its optional generative-AI explanation feature.

If the explanation feature uses Google Gemini, configure the `GEMINI_API_KEY` environment variable with a valid API key.

**Windows PowerShell**

```powershell
$env:GEMINI_API_KEY="your_api_key"
```

**Linux / macOS**

```bash
export GEMINI_API_KEY="your_api_key"
```

Keep API keys private. Do not commit them to the repository or include them in screenshots or shared configuration files.

The core contract analysis and the optional explanation feature may have different configuration requirements.

## Running the Application

From the project root, activate the virtual environment and run the application using its backend entry point:

```bash
python server.py
```

If the server starts successfully, open the local URL printed in the terminal in your web browser.

The exact startup command and local URL depend on the server configuration in `server.py`.

## Project Structure

```text
project/
├── server.py
├── pipeline.py
├── preprocessing.py
├── segmenter.py
├── risk_engine.py
├── app.js
├── models/
│   ├── clause_classifier_weighted/
│   ├── risk_baseline/
│   ├── risk_xlmr/
│   └── risk_xlmr_attention/
├── Datasets/
│   └── processed/
└── README.md
```

*The structure above highlights the main application components. Additional training scripts, datasets, configuration files, and frontend assets may be present in the repository.*

### Main Components

- `server.py` — Backend entry point for the web application.
- `pipeline.py` — Coordinates document processing and contract analysis.
- `preprocessing.py` — Cleans extracted contract text.
- `segmenter.py` — Splits contract text into clauses.
- `risk_engine.py` — Loads the trained models and performs clause classification and risk assessment.
- `app.js` — Handles frontend interactions.
- `models/` — Contains the trained models and associated artifacts.
- `Datasets/` — Contains processed datasets used by the project.

## Limitations

- Model predictions are automated assessments and may contain errors.
- Incorrect text extraction or clause segmentation can affect the results.
- Performance may vary across contract types, writing styles, and languages.
- Explanations are generated automatically and should be checked against the original contract.
- The application does not determine the legal enforceability of a clause or provide professional legal advice.

## Disclaimer

This project is intended for educational and informational purposes. Its outputs should not be treated as definitive legal judgments. Consult a qualified legal professional before making decisions based on a contract analysis.

## License

A license has not yet been specified. Add an appropriate `LICENSE` file before distributing the project under an open-source license.
