"""Tests for the qwen3-vl:4b extractor's output handling.

Self-contained: no network, no Ollama, no model download. The Ollama call
is stubbed at ``_call_ollama_vision`` and the image is arbitrary bytes.
"""

import pytest

from app.services.qwen_vl_extractor import QwenVLExtractor


MINIMAL_ANSWER = (
    '{"accident_details": {"date": "2026-01-05", "time": "14:30",'
    ' "location": "Utrecht", "injuries": false,'
    ' "property_damage_other": true, "witnesses": ["M. Jansen"]},'
    ' "vehicle_a": {"insured_name": "Jan de Vries"}}'
)


def make_extractor(reply: dict) -> QwenVLExtractor:
    """Build an extractor whose Ollama call is stubbed to return `reply`."""
    extractor = QwenVLExtractor()
    extractor._call_ollama_vision = lambda img_b64: reply
    return extractor


def extract_from(reply: dict, tmp_path) -> dict:
    image = tmp_path / "form.jpg"
    image.write_bytes(b"\xff\xd8fake-jpeg-bytes")
    return make_extractor(reply).extract(image)


class TestParseJson:
    def test_plain_json_object(self):
        assert QwenVLExtractor()._parse_json(MINIMAL_ANSWER)["vehicle_a"] == {
            "insured_name": "Jan de Vries"
        }

    def test_json_in_code_block(self):
        text = "Here you go:\n```json\n" + MINIMAL_ANSWER + "\n```\nDone."
        assert QwenVLExtractor()._parse_json(text)["accident_details"]["date"] == "2026-01-05"

    def test_trailing_prose_with_brace_does_not_break_parse(self):
        # Model chatter after the answer that contains a stray '}' must not
        # poison the first-{..last-} span.
        text = MINIMAL_ANSWER + "\nExtraction complete } hope that helps"
        assert QwenVLExtractor()._parse_json(text) is not None

    def test_multiple_objects_returns_the_extraction_not_the_echo(self):
        # Thinking models often echo the schema before answering. The span
        # from the first '{' to the last '}' covers both objects and fails to
        # parse; the scanner must find each object and return the real answer.
        echo = '{"accident_details": {"date": "string", "time": "string"}}'
        text = "Schema understood: " + echo + " Answer: " + MINIMAL_ANSWER
        parsed = QwenVLExtractor()._parse_json(text)
        assert parsed is not None
        assert parsed["accident_details"]["date"] == "2026-01-05"
        assert parsed["vehicle_a"]["insured_name"] == "Jan de Vries"

    def test_braces_inside_json_strings_are_skipped(self):
        text = 'prefix {"remarks": "curly } inside string", "x": 1} suffix }'
        parsed = QwenVLExtractor()._parse_json(text)
        assert parsed == {"remarks": "curly } inside string", "x": 1}

    def test_rejects_non_dict_json(self):
        # A bare list or scalar is not a claim extraction; returning it
        # crashes the vision route's structured.get(...) chain.
        assert QwenVLExtractor()._parse_json("[1, 2, 3]") is None
        assert QwenVLExtractor()._parse_json('"just a string"') is None
        assert QwenVLExtractor()._parse_json("```json\n[1, 2]\n```") is None

    def test_unparseable_returns_none(self):
        assert QwenVLExtractor()._parse_json("no json here } at { all") is None
        assert QwenVLExtractor()._parse_json("") is None


class TestExtract:
    def test_falls_back_to_response_field_when_thinking_unparseable(self, tmp_path):
        # qwen3-vl puts visible output in 'thinking', but if the thinking
        # field holds only reasoning and the JSON landed in 'response', the
        # extraction must still succeed.
        reply = {
            "thinking": "hmm, let me look at the form... {unclosed",
            "response": MINIMAL_ANSWER,
        }
        result = extract_from(reply, tmp_path)
        assert result["structured_output"] is not None
        assert result["structured_output"]["accident_details"]["date"] == "2026-01-05"

    def test_prefers_thinking_field_when_both_parse(self, tmp_path):
        reply = {
            "thinking": MINIMAL_ANSWER,
            "response": '{"accident_details": {"date": "WRONG"}}',
        }
        result = extract_from(reply, tmp_path)
        assert result["structured_output"]["accident_details"]["date"] == "2026-01-05"

    def test_thinking_only_still_works(self, tmp_path):
        result = extract_from({"thinking": MINIMAL_ANSWER, "response": ""}, tmp_path)
        assert result["structured_output"]["vehicle_a"]["insured_name"] == "Jan de Vries"
        assert result["warnings"] == []

    def test_unparseable_reply_returns_none_with_warning(self, tmp_path):
        result = extract_from({"thinking": "garbage", "response": "also garbage"}, tmp_path)
        assert result["structured_output"] is None
        assert any("parse" in w.lower() for w in result["warnings"])

    def test_latency_reported_in_both_units(self, tmp_path):
        # latency_seconds is the extractor's documented contract (the audit
        # route reads it); latency_ms is what the app-wide convention and the
        # web UI use. Both must be present and consistent.
        result = extract_from({"thinking": MINIMAL_ANSWER, "response": ""}, tmp_path)
        assert isinstance(result["latency_seconds"], (int, float))
        assert result["latency_seconds"] >= 0
        assert result["latency_ms"] == round(result["latency_seconds"] * 1000)

    def test_missing_file_reports_warning(self, tmp_path):
        result = QwenVLExtractor().extract(tmp_path / "nope.jpg")
        assert result["structured_output"] is None
        assert any("not found" in w.lower() for w in result["warnings"])

    def test_raw_output_truncated(self, tmp_path):
        reply = {"thinking": "x" * 9000, "response": ""}
        result = extract_from(reply, tmp_path)
        assert len(result["raw_output"]) == 5000
