# Changelog

All notable changes to ClaimLens AI will be documented in this file.

## Unreleased — 2026-09-16

### Fixed
- `app/services/fusion.py`: `_create_conflict` now returns a `SourceConflict`
  Pydantic model instead of a plain dict. The call site was calling
  `c.model_dump()` on dicts and raising `AttributeError` the moment any
  source disagreement was detected — which crashed the entire
  `/api/claims/analyze` endpoint with HTTP 500. Now conflict detection
  actually works end-to-end.
- `README.md`: corrected the clone URL (`aniketkarne-com/...` → `aniketkarne/...`).
  The old URL 404'd.
- `app/config.py` and `.env.example`: default `OLLAMA_MODEL` is now
  `qwen2.5:7b` (the model the README documents as Active) instead of
  `gemma4:e4b` / `phi4` (both explicitly Removed in the model table).
  New users following the README no longer hit "model not found" on
  first call.
- `app/main.py`: dropped `allow_credentials=True` from CORS. The CORS
  spec forbids credentials + wildcard origin, and this API has no auth
  anyway — browsers were rejecting credentialed responses.

### Added
- `LICENSE` (MIT, aniket Karne 2025). README "License" section now
  points at it.
- `.github/workflows/tests.yml` — runs the 28-test pytest suite on
  Python 3.10, 3.11, 3.12 against every push and PR to `main`.
- `Dockerfile` + `.dockerignore` — backend image based on
  `python:3.11-slim-bookworm` with Tesseract (Dutch + English),
  poppler-utils, and FastAPI on port 8000. Documents the assumption
  that Ollama runs on the host (not in the image) — that's a real
  install on macOS, not a docker-in-docker trap.
- `tests/test_fusion.py::test_fuse_returns_valid_conflicts_for_real_disagreement`
  — regression test for the conflict-detection crash.
- README badges (license, Python version, tests pass count).

### Documented
- README project tree: `phi4_reasoner.py` and `florence_extractor.py`
  marked `(deprecated)` — no longer imported by `routes.py`.

## Earlier history

See `journey.md` for the model-selection history (Florence-2,
gemma4:e4b, phi4, qwen3-vl:4b, qwen2.5:7b).