# Changelog

All notable changes to ClaimLens AI are documented here. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Removed
- **Florence-2 (`microsoft/Florence-2-base`)** — the model was already documented as `Removed — misidentified forms` in the README §Model Evaluation History, but the dead code was still in the repo: `app/services/florence_extractor.py` (321 lines), `app/prompts/florence_extraction_prompts.py` (84 lines), and the Florence step in `analyze_claim` / `analyze_text_only`. The Florence-2 health check, the `FLORENCE_MODEL_NAME` env var, and the `florence_model_name` setting are gone. `app/services/fusion.py::fuse()` still accepts a `florence_output` keyword for backward compatibility but force-ignores it.
- **Phi-4 (`phi4`)** — the model was already documented as `Removed — replaced` in the README §Model Evaluation History, but the dead code was still in the repo: `app/services/phi4_reasoner.py` (205 lines) and `app/prompts/phi4_claim_reasoning_prompt.py` (50 lines). `get_phi4()` accessor and the `phi4_reasoner` import are gone from `app/api/routes.py`. The `phi4_result` parameter name in `decision_engine.decide()` and `rules_engine.apply_rules()` is retained for backward compatibility with existing callers.

### Fixed
- **Default Ollama model was wrong.** `app/config.py` defaulted `ollama_model` to `gemma4:e4b` (Removed — hallucinated fields per README §Model Evaluation History). Now defaults to `qwen2.5:7b`, the only text-reasoning model documented as active. `.env.example` updated to match.
- **`/api/claims/analyze-vision` was rejecting every claim as `reject_not_supported`.** The route built `fusion_result` without a `raw_sources` key. The rules engine's `DOCUMENT_EXTRACTION_FAILED` check then fired unconditionally — the vision pipeline intentionally skips Tesseract OCR and Florence-2, so `ocr_text` / `florence_structured` / `user_text` are all empty by design. The decision engine routed that to `recommended_action="reject_not_supported"` with explanation `"Document could not be reliably processed."` So every successful `qwen3-vl:4b` extraction — the model *did* read the form and returned fields — was thrown away silently. The route now populates `raw_sources` (a JSON dump of the model's `structured_output` as a surrogate `ocr_text`, plus `florence_structured` and `user_text`), matching the contract `rules_engine._check_extraction_failed` already expects.
- **`/api/claims/analyze-vision` logged `latency_ms: 0` for every call.** The exception-path dict and the `ai_reasoning` payload both read `latency_ms`, which `qwen_vl_extractor.extract()` never sets (it returns `latency_seconds` — `qwen_vl_extractor.py:125,138,152,165`). The audit log recorded `latency_ms=0` regardless of actual wall-clock cost. Switched to `latency_seconds` in both spots.

### Tests
- New `tests/test_pipeline_cleanup.py` (10 tests) — pins the post-cleanup contract: default model is `qwen2.5:7b`, no `florence_model_name` setting, no `phi4_reasoner` / `florence_extractor` / `phi4_prompts` / `florence_prompts` modules, no `Phi4Reasoner` / `FlorenceExtractor` / `get_phi4` / `get_florence` symbols in `app.api.routes`. Confirms `fuse()` ignores the `florence_output` kwarg end-to-end.
- `tests/test_rules_engine.py` +3 tests — pins the `analyze_vision` raw_sources contract from the rules-engine side: vision payload with populated `raw_sources` does NOT trip `DOCUMENT_EXTRACTION_FAILED`; the pre-fix shape (no `raw_sources`) still trips it; the real extraction-failed case (all three sources empty) still trips it.
- `tests/test_fusion.py::test_fuse_preserves_raw_sources` updated — confirms `florence_structured` is `{}` after the cleanup (Florence-2 data is intentionally dropped).

### Docs
- README.md — drop `phi4_reasoner.py` from the project tree, drop `florence_extraction` from the sample response `steps_completed`.
- `app/models/schemas.py` — `VisionStructuredOutput` / `VisionResult` docstrings clarify the schemas are reserved for future use (vision-pipeline response payloads stay forward-compatible); the active vision path is `/api/claims/analyze-vision` with `qwen3-vl:4b`.
- `app/services/decision_engine.py`, `app/services/rules_engine.py` — module docstrings clarified: the active reasoner is `qwen2.5:7b`; the `phi4_result` parameter name is retained for backward compatibility.

### Stats
- 4 files deleted (`app/services/florence_extractor.py`, `app/services/phi4_reasoner.py`, `app/prompts/florence_extraction_prompts.py`, `app/prompts/phi4_claim_reasoning_prompt.py`) — 660 lines of dead code.
- 41 tests passing (was 27, +14).