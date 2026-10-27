"""Tests for the reasoner -> rules engine -> decision engine contract.

The deterministic layers read ``claim_type`` and ``fraud_indicators`` out of
the reasoner's interpretation. The active reasoner (qwen2.5:7b) was never
asked for either field, so:

* the fraud rule could never fire, and
* every claim was routed to ``request_more_info`` because ``claim_type``
  defaulted to "unknown" - the ``auto_process`` path was unreachable.

These tests pin the contract so the next model swap cannot break it again.
"""

import re

import pytest

from app.prompts.qwen25_claim_reasoning_prompt import QWEN25_SYSTEM_PROMPT
from app.services.decision_engine import DecisionEngine
from app.services.interpretation import (
    CLAIM_TYPE_FIELD,
    FRAUD_INDICATORS_FIELD,
    normalize_claim_type,
    normalize_fraud_indicators,
    normalize_interpretation,
)
from app.services.rules_engine import RulesEngine


def _prompt_schema_block() -> str:
    """Return the JSON-shaped schema block the prompt asks the model to fill in."""
    start = QWEN25_SYSTEM_PROMPT.index("{")
    end = QWEN25_SYSTEM_PROMPT.rindex("}") + 1
    return QWEN25_SYSTEM_PROMPT[start:end]


def _prompt_schema_fields() -> set:
    """Return the field names declared in the prompt's schema block."""
    return set(re.findall(r'^\s*"([a-z0-9_]+)"\s*:', _prompt_schema_block(), re.MULTILINE))


class TestPromptContract:
    """The prompt must request the fields the deterministic layers read."""

    def test_prompt_requests_claim_type(self):
        assert CLAIM_TYPE_FIELD in _prompt_schema_fields()

    def test_prompt_requests_fraud_indicators(self):
        assert FRAUD_INDICATORS_FIELD in _prompt_schema_fields()

    def test_prompt_schema_block_is_balanced(self):
        """A truncated schema block makes the model emit unparseable JSON."""
        block = _prompt_schema_block()
        assert block.count("{") == block.count("}")
        assert len(_prompt_schema_fields()) > 20


class TestReasonerUsesSharedPrompt:
    """The reasoner must send the prompt the contract test pins."""

    def test_reasoner_sends_the_shared_prompt(self):
        pytest.importorskip("httpx", reason="the reasoner needs the Ollama client")

        from app.services import qwen25_reasoner

        assert qwen25_reasoner.QWEN25_SYSTEM_PROMPT is QWEN25_SYSTEM_PROMPT

    def test_reasoner_normalises_its_output(self):
        pytest.importorskip("httpx", reason="the reasoner needs the Ollama client")

        from app.services.qwen25_reasoner import Qwen25Reasoner

        reasoner = Qwen25Reasoner.__new__(Qwen25Reasoner)
        parsed = reasoner._parse_response('{"accident_date": "14-04-2025"}')

        normalised = normalize_interpretation(parsed)

        assert normalised[CLAIM_TYPE_FIELD] == "unknown"
        assert normalised[FRAUD_INDICATORS_FIELD] == []


class TestNormalisation:
    """Normalisation of the contract fields."""

    def test_claim_type_defaults_to_unknown(self):
        assert normalize_claim_type(None) == "unknown"
        assert normalize_claim_type("") == "unknown"
        assert normalize_claim_type("unknown") == "unknown"

    def test_claim_type_is_case_and_separator_insensitive(self):
        assert normalize_claim_type("Motor Accident") == "motor_accident"
        assert normalize_claim_type("MOTOR_ACCIDENT") == "motor_accident"

    def test_fraud_indicators_default_to_empty_list(self):
        assert normalize_fraud_indicators(None) == []
        assert normalize_fraud_indicators("") == []

    def test_fraud_indicators_accept_a_bare_string(self):
        assert normalize_fraud_indicators("impossible date") == ["impossible date"]

    def test_normalize_interpretation_fills_the_contract_fields(self):
        result = normalize_interpretation({"accident_date": "14-04-2025"})

        assert result[CLAIM_TYPE_FIELD] == "unknown"
        assert result[FRAUD_INDICATORS_FIELD] == []
        assert result["accident_date"] == "14-04-2025"

    def test_normalize_interpretation_preserves_other_fields(self):
        result = normalize_interpretation(
            {CLAIM_TYPE_FIELD: "motor_accident", "location": "Utrecht"}
        )

        assert result[CLAIM_TYPE_FIELD] == "motor_accident"
        assert result["location"] == "Utrecht"

    def test_normalize_interpretation_handles_none(self):
        result = normalize_interpretation(None)

        assert result[CLAIM_TYPE_FIELD] == "unknown"
        assert result[FRAUD_INDICATORS_FIELD] == []


class TestEndToEndRouting:
    """The rules and decision engines must see the reasoner's fields."""

    def _fusion_result(self):
        return {
            "normalized_claim_input": {
                "damage_description": "Front bumper damaged",
                "damage_location": "front",
                "accident_type": "rear_end",
            },
            "missing_fields": [],
            "conflicts": [],
            "raw_sources": {"ocr_text": "Locatie Amsterdam", "user_text": ""},
        }

    def test_clean_claim_reaches_auto_process(self):
        """A complete claim with a known claim type must be auto-processable."""
        interpretation = normalize_interpretation(
            {
                CLAIM_TYPE_FIELD: "motor_accident",
                "accident_date": "14-04-2025",
                "location": "Utrecht",
            }
        )
        reasoner_result = {"model": "qwen2.5:7b", "claim_interpretation": interpretation}

        rules = RulesEngine().apply_rules(self._fusion_result(), reasoner_result)
        decision = DecisionEngine().decide(rules, reasoner_result)

        assert decision["recommended_action"] == "auto_process"
        assert decision["risk_level"] == "low"

    def test_fraud_indicators_trigger_the_fraud_rule(self):
        """A reasoner that reports fraud indicators must raise the rule."""
        interpretation = normalize_interpretation(
            {
                CLAIM_TYPE_FIELD: "motor_accident",
                FRAUD_INDICATORS_FIELD: ["incident date is in the future"],
            }
        )
        reasoner_result = {"model": "qwen2.5:7b", "claim_interpretation": interpretation}

        rules = RulesEngine().apply_rules(self._fusion_result(), reasoner_result)

        assert "POSSIBLE_FRAUD_INDICATORS" in rules["rule_flags"]

    def test_fraud_indicators_route_to_manual_review(self):
        """Fraud indicators must not be auto-processed."""
        interpretation = normalize_interpretation(
            {
                CLAIM_TYPE_FIELD: "motor_accident",
                FRAUD_INDICATORS_FIELD: ["staged damage"],
            }
        )
        reasoner_result = {"model": "qwen2.5:7b", "claim_interpretation": interpretation}

        rules = RulesEngine().apply_rules(self._fusion_result(), reasoner_result)
        decision = DecisionEngine().decide(rules, reasoner_result)

        assert decision["recommended_action"] == "manual_review"
        assert decision["manual_review_required"] is True

    def test_missing_claim_type_still_requires_more_info(self):
        """An absent claim type must not silently auto-process a claim."""
        reasoner_result = {"model": "qwen2.5:7b", "claim_interpretation": None}

        rules = RulesEngine().apply_rules(self._fusion_result(), reasoner_result)
        decision = DecisionEngine().decide(rules, reasoner_result)

        assert decision["recommended_action"] == "request_more_info"


class TestRepeatedClaimKeywords:
    """The repeated-claim keywords must match whole words."""

    def _fusion_result(self, text):
        return {
            "normalized_claim_input": {
                "damage_description": "Front bumper damaged",
                "damage_location": "front",
                "accident_type": "rear_end",
            },
            "missing_fields": [],
            "conflicts": [],
            "raw_sources": {"ocr_text": text, "user_text": ""},
        }

    def test_against_is_not_a_repeated_claim(self):
        """"again" is a substring of "against" — a plain check misfires."""
        rules = RulesEngine().apply_rules(
            self._fusion_result("The other driver drove against the barrier at night."),
            None,
        )

        assert "REPEATED_CLAIM_SIGNAL" not in rules["rule_flags"]

    def test_again_still_trips_the_rule(self):
        rules = RulesEngine().apply_rules(
            self._fusion_result("This is my previous claim for the same damage again."),
            None,
        )

        assert "REPEATED_CLAIM_SIGNAL" in rules["rule_flags"]
