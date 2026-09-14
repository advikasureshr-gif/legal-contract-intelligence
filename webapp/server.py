"""
Legal Contract Intelligence -- local web UI
==========================================

A self-contained web front-end for the existing pipeline.

This file does NOT modify any existing project file. It imports the
project's own modules:

    document_processing.processor.process_document
    clause_segmentation.segmenter.segment_contract
    risk.risk_engine.RiskEngine

and exposes them through a small HTTP API plus a static single-page UI.

Run:

    python webapp/server.py                 # from the project root
    python webapp/server.py --port 8000
    python webapp/server.py --open           # also open a browser tab

Modes
-----
real  : trained models found under models/  -> full pipeline
demo  : models (or ML libs) missing         -> real text extraction +
        real clause segmentation + heuristic clause/risk labels, so the
        interface is fully demonstrable without the model artifacts.

Only the Python standard library is required to serve the app. The
"real" mode additionally uses whatever the project already depends on
(PyMuPDF, python-docx, langdetect, torch, transformers, scikit-learn).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from google import genai

# --------------------------------------------------------------------------
# Paths / import wiring
# --------------------------------------------------------------------------

WEBAPP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEBAPP_DIR.parent
STATIC_DIR = WEBAPP_DIR / "static"

# The project modules expect to be importable from the project root, and
# risk.risk_engine loads model files from paths relative to the working
# directory ("models/..."). Make both true regardless of where this
# script is launched from.
for path in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

os.chdir(PROJECT_ROOT)

MAX_UPLOAD_BYTES = 30 * 1024 * 1024  # 30 MB
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

CLAUSE_MODEL_DIR = PROJECT_ROOT / "models" / "clause_classifier_weighted"
RISK_MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "risk_xlmr_attention"
    / "best_model.pt"
)


# --------------------------------------------------------------------------
# Lazy pipeline wiring
# --------------------------------------------------------------------------

_engine = None
_engine_lock = threading.Lock()
_engine_error: str | None = None

_translator = None
_translator_lock = threading.Lock()
_translator_error: str | None = None

# Gemini is loaded lazily.
_gemini_client = None
_gemini_lock = threading.Lock()
_gemini_error: str | None = None


def get_translator():
    """Load the NLLB translator once, lazily."""
    global _translator, _translator_error

    if _translator is not None or _translator_error is not None:
        return _translator

    with _translator_lock:
        if _translator is not None or _translator_error is not None:
            return _translator

        try:
            from translation.translator import NLLBTranslator

            _translator = NLLBTranslator()
        except Exception as exc:
            _translator_error = f"{type(exc).__name__}: {exc}"
            print("[webapp] Translator unavailable ->", _translator_error)

    return _translator

def get_gemini_client():
    """Create the Gemini client once, lazily."""
    global _gemini_client, _gemini_error

    if _gemini_client is not None or _gemini_error is not None:
        return _gemini_client

    with _gemini_lock:
        if _gemini_client is not None or _gemini_error is not None:
            return _gemini_client

        try:
            api_key = os.environ.get("GEMINI_API_KEY")

            if not api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY environment variable is not set."
                )

            _gemini_client = genai.Client(
                api_key=api_key
            )

        except Exception as exc:
            _gemini_error = (
                f"{type(exc).__name__}: {exc}"
            )
            print(
                "[webapp] Gemini unavailable ->",
                _gemini_error
            )

    return _gemini_client

def models_present() -> bool:
    return CLAUSE_MODEL_DIR.is_dir() and RISK_MODEL_FILE.is_file()


def get_engine():
    """Load the project's RiskEngine once, lazily."""
    global _engine, _engine_error

    if _engine is not None or _engine_error is not None:
        return _engine

    with _engine_lock:
        if _engine is not None or _engine_error is not None:
            return _engine
        try:
            from risk.risk_engine import RiskEngine

            _engine = RiskEngine()
        except Exception as exc:  # noqa: BLE001 - surface any load failure
            _engine_error = f"{type(exc).__name__}: {exc}"
            print("[webapp] RiskEngine unavailable ->", _engine_error)
    return _engine


def current_mode() -> str:
    if not models_present():
        return "demo"
    if _engine_error is not None:
        return "demo"
    return "real"


# --------------------------------------------------------------------------
# Text extraction
# --------------------------------------------------------------------------


def clean_text_safe(text: str) -> str:
    """document_processing.preprocessing.clean_text is stdlib-only, so it
    is always available. Fall back to identity if that ever changes."""
    try:
        from document_processing.preprocessing import clean_text

        return clean_text(text)
    except Exception:
        return text


def extract_document(file_path: Path) -> dict:
    """Return {document_name, language, pages:[{page, original_text,
    processed_text}]}.

    Prefers the project's own process_document(). Falls back to a
    dependency-light extractor when the heavy libs are missing (demo
    mode on a machine without PyMuPDF etc.)."""
    ext = file_path.suffix.lower()

    # 1. Try the real project extractor.
    try:
        from document_processing.processor import process_document

        document = process_document(str(file_path))
        return {
            "document_name": document.document_name,
            "language": document.language or "unknown",
            "pages": [
                {
                    "page": page.page,
                    "original_text": page.original_text,
                    "processed_text": page.processed_text,
                }
                for page in document.pages
            ],
        }
    except ImportError:
        pass  # heavy deps missing -> fall through to light extractor
    except Exception as exc:
        # Real extractor is available but rejected the file. Surface that.
        raise RuntimeError(f"Text extraction failed: {exc}") from exc

    # 2. Dependency-light fallback.
    if ext == ".txt":
        raw = file_path.read_text(encoding="utf-8", errors="replace")
        pages_raw = [raw]
    elif ext == ".pdf":
        try:
            import fitz  # type: ignore

            with fitz.open(str(file_path)) as doc:
                pages_raw = [p.get_text() for p in doc]
        except Exception as exc:
            raise RuntimeError(
                "Cannot read PDF without PyMuPDF installed. "
                "Install project requirements or upload a .txt file."
            ) from exc
    elif ext == ".docx":
        try:
            from docx import Document  # type: ignore

            doc = Document(str(file_path))
            pages_raw = ["\n".join(p.text for p in doc.paragraphs)]
        except Exception as exc:
            raise RuntimeError(
                "Cannot read DOCX without python-docx installed. "
                "Install project requirements or upload a .txt file."
            ) from exc
    else:
        raise RuntimeError(f"Unsupported file type: {ext}")

    if not any(p.strip() for p in pages_raw):
        raise RuntimeError("No text could be extracted from the document.")

    language = "unknown"
    try:
        from langdetect import detect  # type: ignore

        language = detect("\n".join(pages_raw))
    except Exception:
        pass

    return {
        "document_name": file_path.name,
        "language": language,
        "pages": [
            {
                "page": i + 1,
                "original_text": text,
                "processed_text": clean_text_safe(text),
            }
            for i, text in enumerate(pages_raw)
        ],
    }


# --------------------------------------------------------------------------
# Heuristic (demo) clause + risk labelling
# --------------------------------------------------------------------------

_TYPE_RULES: list[tuple[tuple[str, ...], str]] = [
    (("terminate for convenience", "terminate at any time", "sole discretion to terminate"),
     "TERMINATION_FOR_CONVENIENCE"),
    (("upon termination", "after termination", "following termination"), "POST_TERMINATION"),
    (("term of this agreement", "effective date", "shall expire", "initial term"),
     "TERM_AND_EXPIRATION"),
    (("automatically renew", "renewal term", "renew for"), "RENEWAL"),
    (("limitation of liability", "shall not be liable", "in no event", "aggregate liability"),
     "LIMITATION_OF_LIABILITY"),
    (("liquidated damages",), "LIQUIDATED_DAMAGES"),
    (("governing law", "governed by the laws", "jurisdiction of the courts"), "GOVERNING_LAW"),
    (("hereby grants", "license to use", "grants a", "right to use"), "LICENSE_GRANT"),
    (("perpetual license", "perpetual, irrevocable"), "PERPETUAL_LICENSE"),
    (("audit", "right to inspect", "inspect the records"), "AUDIT_RIGHTS"),
    (("shall not assign", "assignment of this agreement", "may not be assigned"),
     "ASSIGNMENT_RESTRICTION"),
    (("insurance", "insured", "coverage of not less than"), "INSURANCE"),
    (("exclusive right", "exclusivity", "exclusive distributor"), "EXCLUSIVITY"),
    (("non-compete", "shall not compete", "will not compete"), "NON_COMPETE"),
    (("non-solicit", "shall not solicit", "will not solicit"), "NON_SOLICIT"),
    (("non-disparagement", "shall not disparage"), "NON_DISPARAGEMENT"),
    (("intellectual property", "work product", "inventions", "assigns to"),
     "IP_OWNERSHIP_ASSIGNMENT"),
    (("change of control", "merger or acquisition"), "CHANGE_OF_CONTROL"),
    (("minimum purchase", "minimum commitment", "minimum quantity"), "MINIMUM_COMMITMENT"),
    (("right of first refusal", "first negotiation", "first offer"),
     "RIGHT_OF_FIRST_REFUSAL_OFFER_NEGOTIATION"),
    (("revenue share", "profit sharing", "share of net"), "REVENUE_PROFIT_SHARING"),
    (("warrant", "warranty", "represents and warrants"), "WARRANTY"),
    (("most favored", "most favoured"), "MOST_FAVORED_NATION"),
    (("source code", "escrow"), "SOURCE_CODE_ESCROW"),
    (("covenant not to sue",), "COVENANT_NOT_TO_SUE"),
]

_RISK_PATTERNS: list[tuple[str, str]] = [
    ("sole discretion", "grants one party unilateral discretion"),
    ("without notice", "action permitted without prior notice"),
    ("without cause", "termination or action permitted without cause"),
    ("at any time", "open-ended right exercisable at any time"),
    ("for any reason", "broad 'for any reason' entitlement"),
    ("unlimited", "no stated cap or limit"),
    ("in no event", "sweeping liability exclusion"),
    ("shall not be liable", "broad disclaimer of liability"),
    ("indemnif", "indemnification obligation"),
    ("all losses", "broad 'all losses' exposure"),
    ("all claims", "broad 'all claims' exposure"),
    ("waive", "waiver of a right or protection"),
    ("irrevocab", "irrevocable commitment"),
    ("perpetual", "perpetual / indefinite obligation"),
    ("automatically renew", "auto-renewal that may extend obligations"),
    ("liquidated damages", "pre-set damages payable on breach"),
    ("non-compete", "post-term competitive restriction"),
    ("exclusive", "exclusivity restricts the other party's options"),
]


def _stable_unit(text: str) -> float:
    """Deterministic pseudo-value in [0, 1) from the clause text."""
    acc = 0
    for ch in text[:400]:
        acc = (acc * 131 + ord(ch)) % 1_000_003
    return acc / 1_000_003


def heuristic_analysis(processed_text: str, original_text: str) -> dict:
    haystack = (original_text or processed_text or "").lower()

    clause_type = "OTHER"
    for needles, label in _TYPE_RULES:
        if any(n in haystack for n in needles):
            clause_type = label
            break

    jitter = _stable_unit(processed_text)
    if clause_type == "OTHER":
        clause_confidence = round(0.45 + 0.30 * jitter, 4)
    else:
        clause_confidence = round(0.70 + 0.28 * jitter, 4)

    reasons = [reason for needle, reason in _RISK_PATTERNS if needle in haystack]
    score = min(0.92, 0.12 + 0.19 * len(reasons) + 0.06 * jitter)
    if not reasons:
        score = round(0.05 + 0.30 * jitter, 4)

    risk_label = "UNFAIR" if score >= 0.5 else "FAIR"

    return {
        "clause_type": clause_type,
        "clause_confidence": clause_confidence,
        "risk_label": risk_label,
        "risk_probability": round(float(score), 4),
        "risk_threshold": 0.5,
        "risk_reasons": reasons[:4],
    }


# --------------------------------------------------------------------------
# Full analysis
# --------------------------------------------------------------------------


def analyze_file(file_path: Path) -> dict:
    from clause_segmentation.segmenter import segment_contract  # stdlib-only

    extracted = extract_document(file_path)

    segmentation_input = {
        "document_name": extracted["document_name"],
        "language": extracted["language"] or "unknown",
        "pages": extracted["pages"],
    }
    seg = segment_contract(segmentation_input)
    if seg.get("status") != "success":
        return {
            "status": "error",
            "stage": "clause_segmentation",
            "message": seg.get("message", "Clause segmentation failed."),
        }

    clauses = seg["clauses"]
    mode = current_mode()

    engine = get_engine() if mode == "real" else None
    if mode == "real" and engine is None:
        mode = "demo"

    analyzed = []
    for clause in clauses:
        text_for_model = clause["processed_text"] or clause["original_text"]

        if engine is not None:
            result = engine.analyze_clause(text_for_model)
            result.setdefault("risk_reasons", [])
        else:
            result = heuristic_analysis(
                clause["processed_text"], clause["original_text"]
            )

        analyzed.append(
            {
                "clause_id": clause["clause_id"],
                "section": clause["section"],
                "page": clause["page"],
                "language": clause["language"],
                "original_text": clause["original_text"],
                "processed_text": clause["processed_text"],
                "clause_type": result["clause_type"],
                "clause_confidence": float(result["clause_confidence"]),
                "risk_label": result["risk_label"],
                "risk_probability": float(result["risk_probability"]),
                "risk_threshold": float(result["risk_threshold"]),
                "risk_reasons": result.get("risk_reasons", []),
            }
        )

    risk_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    for c in analyzed:
        risk_counts[c["risk_label"]] = risk_counts.get(c["risk_label"], 0) + 1
        type_counts[c["clause_type"]] = type_counts.get(c["clause_type"], 0) + 1

    avg_conf = (
        sum(c["clause_confidence"] for c in analyzed) / len(analyzed)
        if analyzed
        else 0.0
    )

    return {
        "status": "success",
        "mode": mode,
        "document_name": extracted["document_name"],
        "language": extracted["language"],
        "page_count": len(extracted["pages"]),
        "clause_count": len(analyzed),
        "summary": {
            "risk_counts": risk_counts,
            "flagged": risk_counts.get("UNFAIR", 0),
            "clause_type_counts": type_counts,
            "avg_clause_confidence": round(avg_conf, 4),
        },
        "clauses": analyzed,
    }


# --------------------------------------------------------------------------
# HTTP layer
# --------------------------------------------------------------------------

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".json": "application/json; charset=utf-8",
}


class Handler(BaseHTTPRequestHandler):
    server_version = "LCIWebApp/1.0"

    # -- helpers ---------------------------------------------------------

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.is_file():
            self._send_json({"error": "Not found"}, status=404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header(
            "Content-Type",
            _CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream"),
        )
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _static(self, rel: str) -> None:
        rel = rel.lstrip("/")
        target = (STATIC_DIR / rel).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())):
            self._send_json({"error": "Forbidden"}, status=403)
            return
        self._send_file(target)

    # -- routes --------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        route = self.path.split("?", 1)[0]

        if route == "/" or route == "/index.html":
            self._send_file(STATIC_DIR / "index.html")
            return

        if route == "/api/health":
            self._send_json(
                {
                    "status": "ok",
                    "mode": current_mode(),
                    "models_present": models_present(),
                    "engine_error": _engine_error,
                }
            )
            return

        if route.startswith("/static/"):
            self._static(route[len("/static/") :])
            return

        self._send_json({"error": "Not found"}, status=404)
    def handle_translate(self) -> None:
        """Translate a batch of English texts for presentation."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0

        if length <= 0:
            self._send_json({"error": "Empty request."}, status=400)
            return

        if length > 2 * 1024 * 1024:
            self._send_json(
                {"error": "Translation request too large."},
                status=413,
            )
            return

        try:
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8"))

            texts = payload.get("texts", [])
            target_language = payload.get("target_language", "en")

            if not isinstance(texts, list):
                self._send_json(
                    {"error": "'texts' must be a list."},
                    status=400,
                )
                return

            if target_language not in {"en", "hi", "ta"}:
                self._send_json(
                    {"error": "Unsupported target language."},
                    status=400,
                )
                return

            if not texts:
                self._send_json(
                    {
                        "status": "success",
                        "target_language": target_language,
                        "translations": [],
                    }
                )
                return

            if target_language == "en":
                translations = texts
            else:
                translator = get_translator()

                if translator is None:
                    self._send_json(
                        {
                            "status": "error",
                            "message": (
                                "Translation model unavailable: "
                                + str(_translator_error)
                            ),
                        },
                        status=500,
                    )
                    return

                translations = translator.translate_many(
                    texts,
                    target_language,
                    batch_size=4,
                )

            self._send_json(
                {
                    "status": "success",
                    "target_language": target_language,
                    "translations": translations,
                }
            )

        except json.JSONDecodeError:
            self._send_json(
                {"error": "Invalid JSON request."},
                status=400,
            )
        except Exception as exc:
            traceback.print_exc()
            self._send_json(
                {
                    "status": "error",
                    "message": str(exc),
                },
                status=500,
            )
    def handle_explain(self) -> None:
        """Explain one contractual clause using Gemini."""

        try:
            length = int(
                self.headers.get(
                    "Content-Length",
                    "0",
                )
            )
        except ValueError:
            length = 0

        if length <= 0:
            self._send_json(
                {"error": "Empty request."},
                status=400,
            )
            return

        if length > 100 * 1024:
            self._send_json(
                {
                    "error":
                        "Explanation request too large."
                },
                status=413,
            )
            return

        try:
            body = self.rfile.read(length)

            payload = json.loads(
                body.decode("utf-8")
            )

            clause = str(
                payload.get(
                    "clause",
                    "",
                )
            ).strip()

            language = payload.get(
                "language",
                "en",
            )

            risk_label = str(
                payload.get(
                    "risk_label",
                    "UNKNOWN",
                )
            )

            clause_type = str(
                payload.get(
                    "clause_type",
                    "UNKNOWN",
                )
            )

            risk_reasons = payload.get(
                "risk_reasons",
                [],
            )

            if not clause:
                self._send_json(
                    {
                        "error":
                            "Clause text is required."
                    },
                    status=400,
                )
                return

            if language not in {
                "en",
                "hi",
                "ta",
            }:
                self._send_json(
                    {
                        "error":
                            "Unsupported explanation language."
                    },
                    status=400,
                )
                return

            if not isinstance(
                risk_reasons,
                list,
            ):
                risk_reasons = []

            client = get_gemini_client()

            if client is None:
                self._send_json(
                    {
                        "status": "error",
                        "message":
                            "Gemini is unavailable: "
                            + str(_gemini_error),
                    },
                    status=503,
                )
                return

            language_names = {
                "en": "English",
                "hi": "Hindi",
                "ta": "Tamil",
            }

            target_language = (
                language_names[language]
            )

            if risk_reasons:
                reasons_text = ", ".join(
                    str(reason)
                    for reason in risk_reasons
                )
            else:
                reasons_text = (
                    "No specific risk indicators "
                    "were provided."
                )

            prompt = f"""
You are a legal-information assistant inside
a contract analysis application.

Your task is to explain ONE contractual clause
to a user in {target_language}.

IMPORTANT RULES:

- Explain only the supplied clause and the analysis provided by our system.
- Do not invent facts, obligations, deadlines, penalties, rights, or other contract terms.
- Do not independently change, assign, or justify the risk classification.
- The risk classification and probability were produced by our machine-learning model.
- Do not claim to know why the model made its classification unless a reason is explicitly provided.
- Do not state that a clause is definitely legal or illegal.
- Do not provide personalized legal advice.
- Use clear, accessible language.
- Respond entirely in {target_language}.
- Do not use Markdown formatting, bullet symbols, or special formatting characters. Use simple numbered sections.

CLAUSE:
{clause}

OUR MODEL'S ANALYSIS:

Risk classification:
{risk_label}

Clause type:
{clause_type}

Risk indicators:
{reasons_text}
If Risk indicators is empty or indicates that none were provided, do not infer or speculate about why the model assigned its classification.

Please provide:

1. What this clause means in simple language.
2. What our system classified this clause as.
3. What parts of the clause the user should pay attention to.

Keep the explanation concise but useful.
""".strip()

            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=prompt,
            )

            explanation = (
                response.text or ""
            ).strip()

            if not explanation:
                self._send_json(
                    {
                        "status": "error",
                        "message":
                            "Gemini returned an empty "
                            "explanation.",
                    },
                    status=502,
                )
                return

            self._send_json(
                {
                    "status": "success",
                    "language": language,
                    "explanation": explanation,
                }
            )

        except json.JSONDecodeError:
            self._send_json(
                {
                    "error":
                        "Invalid JSON request."
                },
                status=400,
            )

        except Exception as exc:
            traceback.print_exc()

            self._send_json(
                {
                    "status": "error",
                    "message": str(exc),
                },
                status=500,
            )
    def do_POST(self) -> None:  # noqa: N802
        route = self.path.split("?", 1)[0]

        if route == "/api/translate":
            self.handle_translate()
            return

        if route == "/api/explain":
            self.handle_explain()
            return
        
        if route != "/api/analyze":
            self._send_json({"error": "Not found"}, status=404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0

        if length <= 0:
            self._send_json({"error": "Empty upload."}, status=400)
            return
        if length > MAX_UPLOAD_BYTES:
            self._send_json(
                {"error": "File too large (limit 30 MB)."}, status=413
            )
            return

        raw_name = self.headers.get("X-Filename", "contract")
        safe_name = os.path.basename(raw_name).strip() or "contract"
        safe_name = re.sub(r"[^A-Za-z0-9._ -]", "_", safe_name)
        ext = Path(safe_name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            self._send_json(
                {
                    "error": (
                        f"Unsupported file type '{ext or '?'}'. "
                        "Upload a PDF, DOCX, or TXT file."
                    )
                },
                status=400,
            )
            return

        body = self.rfile.read(length)

        tmp_dir = WEBAPP_DIR / ".uploads"
        tmp_dir.mkdir(exist_ok=True)
        tmp_path = tmp_dir / f"upload_{threading.get_ident()}{ext}"
        try:
            tmp_path.write_bytes(body)
            report = analyze_file(tmp_path)
            report["document_name"] = safe_name
            status = 200 if report.get("status") == "success" else 422
            self._send_json(report, status=status)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._send_json(
                {"status": "error", "message": str(exc)}, status=500
            )
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass

    # quieter, single-line logs
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        sys.stderr.write(
            "[webapp] %s - %s\n" % (self.address_string(), fmt % args)
        )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Legal Contract Intelligence web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--open", action="store_true", help="open a browser tab on start"
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="load the RiskEngine at startup instead of on first request",
    )
    args = parser.parse_args()

    mode = current_mode()
    print("=" * 64)
    print(" Legal Contract Intelligence  --  local web UI")
    print("=" * 64)
    print(f" Project root : {PROJECT_ROOT}")
    print(f" Mode         : {mode.upper()}", end="")
    if mode == "demo":
        if not models_present():
            print("  (trained models not found under models/)")
        else:
            print("  (model load failed -- see log)")
    else:
        print()
    print(f" URL          : http://{args.host}:{args.port}")
    print("=" * 64)

    if args.preload and mode == "real":
        print(" Preloading RiskEngine ...")
        get_engine()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)

    if args.open:
        threading.Timer(
            0.6, lambda: webbrowser.open(f"http://{args.host}:{args.port}")
        ).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[webapp] shutting down")
        httpd.shutdown()


if __name__ == "__main__":
    main()