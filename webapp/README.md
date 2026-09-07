# Web UI — Legal Contract Intelligence

A local, single-page web interface for the contract analysis pipeline.
Upload a contract, get a clause-by-clause breakdown with clause types,
type confidence, and a FAIR / potentially-unfair risk label per clause.

This folder is **additive** — it imports the existing project modules
(`document_processing`, `clause_segmentation`, `risk`) and does not modify
any existing file.

## Running

From the **project root**:

```bash
python webapp/server.py
```

Then open <http://127.0.0.1:8000>.

Options:

```bash
python webapp/server.py --port 8000 --open      # open a browser tab
python webapp/server.py --preload               # load the model at startup
```

Only the Python standard library is needed to serve the UI. No Flask /
FastAPI / Node / build step.

## Modes

The banner in the top-right shows which mode is active.

| Mode | When | Behaviour |
| --- | --- | --- |
| **Models ready** | `models/clause_classifier_weighted/` and `models/risk_baseline/risk_model.joblib` exist and load | Real text extraction, real clause segmentation, real XLM-R clause classifier + TF-IDF risk model |
| **Demo mode** | trained models or ML libraries not found | Real text extraction + real clause segmentation, but clause types and risk are produced by a lightweight keyword heuristic so the interface is still fully demonstrable |

For the professor demo on the machine that has the trained `models/`
directory, it will run in **Models ready** mode automatically.

## Supported files

PDF, DOCX, TXT (up to 30 MB). PDF/DOCX extraction uses the project's
existing `document_processing` module (PyMuPDF / python-docx). Plain-text
upload works even without those installed.

## What each screen shows

- **Risk overview** — split of standard vs. flagged clauses
- **Clause / confidence / language** summary tiles
- **Most common clause types** in the document
- **Clause list** — searchable, filterable by clause type and by risk;
  each card shows the clause text, its predicted type with confidence,
  and the risk badge with probability (and, in demo mode, the phrases
  that triggered the flag)
- **Export JSON** — full machine-readable report
- **Print report** — print / save-as-PDF friendly layout

## Notes

- Uploaded files are written to `webapp/.uploads/` only for the duration
  of a request and deleted immediately after. That folder is git-ignored.
- The server binds to `127.0.0.1` only.
