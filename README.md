# ClaimLens AI — Local Multimodal Insurance Claims Intake System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-28%20passed-brightgreen.svg)](./tests/)

A POC demonstrating how small/local AI models can handle insurance company-level claims intake and triage — entirely on-premise.

## What is ClaimLens AI?

ClaimLens AI is a modular, explainable, local-first AI pipeline that processes European/Dutch motor accident claim forms (PDF or image) containing printed text, handwritten notes, checkboxes, diagrams, and incomplete fields. It extracts, fuses, reasons over, and makes deterministic decisions about insurance claims — all locally, with no data leaving the machine.

**This is not a chatbot. This is not a simple FastAPI wrapper.**

## The Problem It Solves

Insurance claims intake is often:
- **Slow** — days to weeks for manual review
- **Expensive** — large call centers for simple form processing
- **Opaque** — no explanation for why a claim was flagged
- **Black-box** — over-reliance on AI confidence scores that aren't actually calibrated

ClaimLens AI demonstrates a better architecture:
- Tesseract OCR for robust printed/handwritten text extraction
- qwen2.5:7b for structured JSON extraction from OCR text
- qwen3-vl:4b for direct vision-based analysis (optional)
- Deterministic rules for business decisions
- Full auditability and explainability

## Why Small/Local Models Fit Insurance Use Cases

| Reason | Why it matters |
|--------|---------------|
| **Data Privacy** | Claims contain PII, medical info, addresses. Local processing keeps data on-prem. |
| **Explainability** | Small models + deterministic rules produce explainable decisions. |
| **Cost Efficiency** | No per-token API costs. Fixed infrastructure cost. |
| **Latency** | Local models eliminate network round-trips. |
| **Compliance** | Easier to audit what's running and how decisions are made. |
| **No hallucinations about policy** | Small models don't have encyclopedic knowledge to hallucinate from. |

---

## Architecture

### OCR Pipeline (Recommended)

```
[PDF/Image Upload]
      ↓
[Document Preprocessor]  — PDF→image conversion, resize, normalization
      ↓
[Tesseract OCR]          — ~500ms, extracts raw text from scan/image
      ↓
[qwen2.5:7b via Ollama]  — ~24s, reasons over OCR text, outputs structured JSON
      ↓
[Fusion Layer]           — merges OCR + AI output, detects conflicts/missing fields
      ↓
[Rules Engine]           — deterministic flags: injuries, drunk driving, witnesses, etc.
      ↓
[Decision Engine]        — final workflow routing: auto / review / reject
      ↓
[Audit Log]              — JSON file per claim, full pipeline trace
```

### Vision Pipeline (Experimental)

```
[Image Upload]
      ↓
[qwen3-vl:4b via Ollama]  — ~60s+ on CPU, directly extracts structured data from image
      ↓
[Fusion Layer]
      ↓
[Rules Engine]
      ↓
[Decision Engine]
      ↓
[Audit Log]
```

> **Note:** The OCR pipeline is recommended for production. Vision is slower on CPU and checkbox detection is unreliable. See [Performance Benchmarks](#performance-benchmarks) below.

### Layer Responsibilities

| Layer | Purpose | Technology |
|-------|---------|------------|
| Document Preprocessor | PDF→image, resize, normalize | Python (Pillow, pdf2image) |
| Tesseract OCR | Extract raw text from images | Tesseract (local) |
| qwen2.5:7b | Structured reasoning + JSON extraction | Ollama (local, 7B params) |
| qwen3-vl:4b | Vision-based extraction (experimental) | Ollama (local, 4B params) |
| Fusion | Merge outputs, detect conflicts | Deterministic Python logic |
| Rules Engine | Business rule flags | Deterministic Python logic |
| Decision Engine | Final workflow routing | Deterministic Python logic |
| Audit Logger | Persistent claim record | JSON files |

### Model Evaluation History

See `journey.md` for full details.

| Model | Role | Status |
|-------|------|--------|
| Florence-2 | Vision extraction | Removed — misidentified forms |
| gemma4:e4b | Text reasoning | Removed — hallucinated fields |
| phi4 | Text reasoning | Removed — replaced |
| **qwen2.5:7b** | Text reasoning | **Active** — clean JSON, reliable |
| **qwen3-vl:4b** | Vision extraction | Available — slower, needs prompt tuning |

---

## Performance Benchmarks

Tested on: macOS, Apple Silicon M4, 24GB RAM, with Ollama running locally.

### OCR Pipeline (Tesseract + qwen2.5:7b)

| Metric | Value | Notes |
|--------|-------|-------|
| **Total end-to-end** | **~33s** | File upload → final decision |
| Tesseract OCR | ~550ms | Single page image |
| qwen2.5:7b reasoning | ~24s | OCR text → structured JSON |
| Rules + Decision | <10ms | Deterministic |
| **Field extraction accuracy** | **~95%** | On synthetic test form |
| **Steps completed** | 8/8 | All pipeline stages |

**Extracted fields from test form:**
- Accident date: ✓ 14-04-2025
- Accident time: ✓ 08:35
- Location: ✓ Kruisdrift / Parkkant junction, Utrecht
- Insured A: ✓ JANSSEN, Toyota Corolla Hybrid, NL-AB 123-6
- Insured B: ✓ DE VRIES, Volkswagen Golf, NL-20-106-11
- Insurance A: ✓ ANWB Insurance, Policy POL-A-4589021
- Insurance B: ✓ Centraal Beheer, Policy CB-993812-77
- Damage A: ✓ Rear bumper dented
- Damage B: ✓ Front bumper scratched
- Witnesses: ✓ Sarah Miller (+phone)
- OCR accuracy: ~95% on synthetic test image

### Vision Pipeline (qwen3-vl:4b)

| Metric | Value | Notes |
|--------|-------|-------|
| **Total end-to-end** | **~60–90s** | CPU-only, no GPU |
| qwen3-vl:4b inference | ~60s+ | Very slow on CPU |
| Checkbox detection | Poor | Hallucinates checkbox states |
| Text extraction | Good | When prompt is detailed |

> Vision pipeline is recommended only for environments where OCR preprocessing is unavailable, or with GPU acceleration.

### Model Comparison (1 sample)

| Model | Latency | JSON Reliability | Checkbox Detection | Recommendation |
|-------|---------|-----------------|-------------------|----------------|
| qwen2.5:7b | ~24s | High | Via OCR text | **Recommended** |
| qwen3-vl:4b | ~60s+ | Medium | Poor (hallucinates) | Experimental |
| phi4 | ~45s | Low (hallucinated) | Broken | Avoid |
| gemma4:e4b | ~30s | Low | Broken | Avoid |

---

## Web UI

A frontend is included at `frontend/index.html`. It provides:
- Claim information form (optional fields — all extracted from form anyway)
- Drag-and-drop file upload (PDF, PNG, JPG)
- Pipeline mode selector: OCR or Vision
- Real-time results: latency breakdown, extracted fields, risk level, decision
- No data leaves the machine

To use:
1. Start the backend: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
2. Open `frontend/index.html` in a browser (or serve it with any static file server)
3. Upload a claim form image or PDF

---

## API Usage

### Health Check

```bash
curl http://localhost:8000/api/health
```

### OCR Pipeline — Analyze Claim (Recommended)

```bash
curl -X POST http://localhost:8000/api/claims/analyze \
  -F "file=@path/to/claim_form.pdf" \
  -F "user_text=Optional context from the claimant"
```

### Vision Pipeline — Analyze Claim (Experimental)

```bash
curl -X POST http://localhost:8000/api/claims/analyze-vision \
  -F "file=@path/to/claim_form.jpg" \
  -F "user_text=Optional additional context"
```

### Analyze Text Only (for testing)

```bash
curl -X POST http://localhost:8000/api/claims/analyze-text-only \
  -H "Content-Type: application/json" \
  -d '{"user_text": "Ik werd van achteren aangereden bij het stoplicht. Achterbumper beschadigd."}'
```

### Swagger UI

Open http://localhost:8000/docs for the interactive API explorer.

---

## Response Structure

```json
{
  "claim_id": "uuid",
  "status": "processed",
  "input_summary": {
    "file_name": "test_claim.jpg",
    "file_type": "image/jpeg",
    "user_text_provided": false,
    "language": "nl"
  },
  "extraction": {
    "ocr": {
      "source": "tesseract",
      "raw_text": "CONSTATS AMIABLES...",
      "language_used": "nl",
      "warnings": [],
      "latency_ms": 553
    },
    "vision": { ... }
  },
  "fusion": {
    "normalized_claim_input": { ... },
    "missing_fields": [],
    "conflicts": [],
    "source_map": {}
  },
  "ai_reasoning": {
    "model": "qwen2.5:7b",
    "claim_interpretation": {
      "accident_date": "14-04-2025",
      "accident_time": "08:35",
      "location": "Kruisdrift / Parkkant junction, Utrecht",
      "insured_name_a": "JANSSEN",
      "vehicle_a": "Toyota Corolla Hybrid",
      "plate_a": "NL-AB 123-6",
      "insurance_a": "ANWB Insurance",
      "damage_a": "Rear bumper dented",
      "damage_b": "Front bumper scratched",
      "witnesses": ["Sarah Miller, 18 Oak Lane, Utrecht, +31 6 1234 7788"]
    },
    "raw_output": "...",
    "latency_ms": 24259
  },
  "rules": {
    "applied_rules": ["injury_flag", "witness_present"],
    "rule_flags": {
      "injury": false,
      "property_damage": true,
      "witness_present": true,
      "alcohol_suspected": false
    }
  },
  "decision": {
    "risk_level": "medium",
    "recommended_action": "request_more_info",
    "decision_reasons": [
      "No injuries reported",
      "Witness present - corroboration recommended",
      "Damage is minor to moderate"
    ]
  },
  "audit": {
    "pipeline_version": "0.1.0",
    "steps_completed": [
      "file_upload", "image_loaded", "tesseract_ocr",
      "florence_extraction", "fusion", "qwen_reasoning",
      "rules_engine", "decision_engine"
    ],
    "warnings": [],
    "total_latency_ms": 33000
  }
}
```

---

## Sample Test Cases

See `sample_data/sample_claim_texts.json`:

| Case | Description |
|------|-------------|
| `simple_rear_end_collision` | Basic rear-end, no injuries, clear fault |
| `repeated_claim_signal` | Keyword repetition suggesting fraud pattern |
| `missing_information` | Critical fields missing |
| `conflicting_damage_location` | Conflict between text and visual evidence |

A synthetic test image is included at `sample_data/test_claim.jpg`.

---

## No Fake Confidence Scores

This POC intentionally does **not** fake confidence scores. Instead it exposes:
- **Source evidence** — what was extracted and from where
- **Missing fields** — what couldn't be extracted
- **Conflicts** — where sources disagree
- **Rule flags** — deterministic signals
- **Manual review decisions** — clear reasons for routing

This is more honest and enterprise-appropriate than pretending the system knows how confident it is.

---

## Important Note on `auto_process`

`recommended_action: "auto_process"` does **NOT** mean payment approval. It means the claim can proceed to the next automated workflow step (damage estimation, adjuster assignment). Final payment decisions always require human review.

---

## Setup Instructions

### 1. Clone and Virtual Environment

```bash
git clone https://github.com/aniketkarne/claimlens-ai.git
cd claimlens-ai
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Install Tesseract OCR

**macOS:**
```bash
brew install tesseract tesseract-lang
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-nld
```

**Verify:**
```bash
tesseract --version
```

### 3. Install Ollama

```bash
# Install (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the required reasoning model
ollama pull qwen2.5:7b

# Pull the optional vision model
ollama pull qwen3-vl:4b

# Verify
ollama list
```

### 4. Start Ollama Server

```bash
ollama serve  # keeps running on port 11434
```

### 5. Start the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

API available at:
- **API**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Known Limitations

1. **Handwritten text** — Tesseract OCR accuracy drops significantly. Use high-quality scans/photos.
2. **qwen3-vl:4b checkbox detection** — Hallucinates checkbox states. Use OCR + qwen2.5:7b pipeline.
3. **Vision pipeline speed** — ~60s+ on CPU. Not suitable for production without GPU.
4. **Dutch language** — Optimized for Dutch. Other languages untested.
5. **Single-page processing** — POC processes first page of PDFs only.
6. **No database** — Uses local JSON files for audit logs. Production needs PostgreSQL/MongoDB.

---

## Future Improvements

- [ ] GPU-accelerated inference (CUDA) for vision pipeline
- [ ] Multi-page PDF support
- [ ] English/German/French language support
- [ ] Damage severity classification
- [ ] Integration with insurance company databases (policy lookup)
- [ ] PostgreSQL for persistent claim storage
- [ ] Celery/Redis for async processing
- [ ] S3-compatible storage for uploaded files
- [ ] Real-time webhook notifications for decisions
- [ ] Dashboard for manual review queue

---

## Project Structure

```
claimlens-ai/
├── README.md
├── journey.md                    # Full model selection history
├── requirements.txt
├── .env.example
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI application
│   ├── config.py                # Configuration management
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py            # API endpoints (analyze, analyze-vision, analyze-text-only)
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py           # Pydantic request/response models
│   └── services/
│       ├── __init__.py
│       ├── document_loader.py   # PDF/image loading + conversion
│       ├── image_preprocessor.py # Image preprocessing
│       ├── tesseract_ocr.py     # Tesseract OCR extraction
│       ├── qwen_vl_extractor.py # qwen3-vl:4b vision extraction
│       ├── qwen25_reasoner.py   # qwen2.5:7b structured reasoning
│       ├── phi4_reasoner.py      # (deprecated) phi4 reasoning — no longer wired into routes
│       ├── florence_extractor.py # (deprecated) Florence-2 — empty passthrough placeholder
│       ├── fusion.py             # Fusion layer — merge + conflict detection
│       ├── rules_engine.py       # Deterministic business rules
│       ├── decision_engine.py    # Final decision routing
│       └── audit_logger.py       # JSON audit log persistence
├── frontend/
│   └── index.html               # Web UI (single-file, no build needed)
├── sample_data/
│   ├── README.md
│   ├── sample_claim_texts.json  # Test case texts
│   └── test_claim.jpg           # Synthetic test form image
├── tests/
│   ├── test_fusion.py
│   ├── test_rules_engine.py
│   └── test_decision_engine.py
├── logs/
│   └── claims/                  # Per-claim JSON audit logs
└── uploads/                     # Temporary uploaded files
```

---

## Ollama Commands Reference

```bash
# Check installed models
ollama list

# Pull models
ollama pull qwen2.5:7b   # Primary reasoning model (required)
ollama pull qwen3-vl:4b  # Vision model (optional)

# Check running models
ollama ps

# Remove legacy models
ollama rm phi4
ollama rm gemma4:e4b
ollama rm florence2
```

---

## License

[MIT](./LICENSE) — aniket Karne, 2025.
