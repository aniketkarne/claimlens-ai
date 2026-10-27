# Changelog

All notable changes to ClaimLens AI are documented here. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed
- **The OCR layer never used the Dutch language pack.** `app/services/tesseract_ocr.py` called `pytesseract.get_tesseract_lang_codes()`, which does not exist in pytesseract 0.3.10 — the pinned version. The resulting `AttributeError` was swallowed by a bare `except Exception`, so every OCR call silently fell back to the English pack *and* reported "Language 'nl' not available" even when the Dutch pack was installed. A Dutch claim form was therefore read by an English OCR engine on every request. The lookup now uses `pytesseract.get_languages()`, which is the API that actually exists, and the installed-pack set is cached per process. A genuinely missing pack still falls back to English with the warning.
- **The reasoner never produced the fields the deterministic layers read.** The active reasoner's prompt (`qwen2.5:7b`) did not ask for `claim_type` or `fraud_indicators`, but `DecisionEngine._check_claim_type_unknown()` reads `claim_type` and `RulesEngine._check_fraud_indicators()` reads `fraud_indicators`. `claim_type` therefore always defaulted to `"unknown"`, so every claim was routed to `request_more_info` with `risk_level: unknown` and the `auto_process` branch was unreachable; the fraud rule could never fire. The prompt now requests both fields, and a new `app/services/interpretation.py` owns the field names and their normalisation so the producer and the two consumers cannot drift apart again.
- **`/api/claims/analyze-vision` returned HTTP 500 on an unparseable model reply.** `qwen_vl_extractor.extract()` returns `structured_output: None` when the vision model's answer cannot be parsed as JSON. The route read it with `vision_result.get("structured_output", {})`, but `dict.get` only applies the default when the key is *absent* — the key was present and `None`, so the next line raised `AttributeError: 'NoneType' object has no attribute 'get'`. One unparseable model reply took down the whole endpoint.
- **The repeated-claim rule fired on ordinary claims.** The keyword check was a plain substring test, and `"again"` is a substring of `"against"` — so any English claim mentioning a collision "against" a barrier tripped a **high**-severity repeated-claim flag and was forced to manual review. The check now matches whole words.
- **`pytest` failed to collect from the repository root.** The package lives at the repo root rather than under `src/`, and there was no `conftest.py`, so `pytest` aborted with `ModuleNotFoundError: No module named 'app'` — only `python -m pytest` happened to work. A root `conftest.py` puts the repo root on `sys.path`, so the documented command works.

### Added
- `app/prompts/qwen25_claim_reasoning_prompt.py` — the `qwen2.5:7b` system prompt moved out of the service module so the contract it defines can be imported and tested without pulling in the HTTP client.
- `app/services/interpretation.py` — the shared contract between the reasoning layer and the deterministic layers: field names, `normalize_claim_type()`, `normalize_fraud_indicators()` and `normalize_interpretation()`.
- `tests/test_interpretation_contract.py` — pins the prompt/consumer contract (the prompt must request `claim_type` and `fraud_indicators`; the rules engine must fire on reported fraud indicators; the decision engine must reach `auto_process` for a complete claim with a known claim type) and the whole-word keyword check.
- `tests/test_tesseract_language.py` — pins the language-pack resolution: the Dutch pack is actually used when installed, a missing pack still falls back with a warning, and the installed-pack lookup is cached.
- `tests/test_vision_route.py` — regression test for the `analyze-vision` crash on an unparseable model reply.
- `conftest.py` — makes the repo root importable so `pytest` works from a clean checkout.

### Tests
- 54 tests passing (was 27, +27). 8 of the 27 new tests fail against the pre-fix production code — the other 19 assert the behaviour of the new shared-contract module itself and pass with it in place.
