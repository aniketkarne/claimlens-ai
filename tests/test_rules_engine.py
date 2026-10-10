"""Tests for the rules engine."""

import pytest

from app.services.rules_engine import RulesEngine


class TestRulesEngine:
    """Test cases for RulesEngine."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rules = RulesEngine()

    def test_missing_critical_fields_rule(self):
        """Test MISSING_CRITICAL_FIELDS rule."""
        fusion_result = {
            "normalized_claim_input": {},
            "missing_fields": ["damage_description", "damage_location", "accident_type"],
            "conflicts": [],
            "raw_sources": {
                "ocr_text": "",
                "florence_structured": {},
                "user_text": ""
            }
        }

        result = self.rules.apply_rules(fusion_result, None)

        rule_ids = [r["rule_id"] for r in result["applied_rules"]]
        assert "MISSING_CRITICAL_FIELDS" in rule_ids

        missing_rule = next(r for r in result["applied_rules"] if r["rule_id"] == "MISSING_CRITICAL_FIELDS")
        assert missing_rule["severity"] == "medium"

    def test_source_conflict_rule(self):
        """Test SOURCE_CONFLICT rule."""
        fusion_result = {
            "normalized_claim_input": {},
            "missing_fields": [],
            "conflicts": [
                {"field": "damage_location", "sources": {"user_text": "rear", "florence-2": "front"}}
            ],
            "raw_sources": {
                "ocr_text": "",
                "florence_structured": {},
                "user_text": "rear"
            }
        }

        result = self.rules.apply_rules(fusion_result, None)

        rule_ids = [r["rule_id"] for r in result["applied_rules"]]
        assert "SOURCE_CONFLICT" in rule_ids

        conflict_rule = next(r for r in result["applied_rules"] if r["rule_id"] == "SOURCE_CONFLICT")
        assert conflict_rule["severity"] == "high"

    def test_unknown_damage_location_rule(self):
        """Test UNKNOWN_DAMAGE_LOCATION rule."""
        fusion_result = {
            "normalized_claim_input": {
                "damage_location": None
            },
            "missing_fields": [],
            "conflicts": [],
            "raw_sources": {
                "ocr_text": "Some text",
                "florence_structured": {},
                "user_text": ""
            }
        }

        result = self.rules.apply_rules(fusion_result, None)

        rule_ids = [r["rule_id"] for r in result["applied_rules"]]
        assert "UNKNOWN_DAMAGE_LOCATION" in rule_ids

    def test_unknown_accident_type_rule(self):
        """Test UNKNOWN_ACCIDENT_TYPE rule."""
        fusion_result = {
            "normalized_claim_input": {
                "accident_type": None
            },
            "missing_fields": [],
            "conflicts": [],
            "raw_sources": {
                "ocr_text": "Some text",
                "florence_structured": {},
                "user_text": ""
            }
        }

        result = self.rules.apply_rules(fusion_result, None)

        rule_ids = [r["rule_id"] for r in result["applied_rules"]]
        assert "UNKNOWN_ACCIDENT_TYPE" in rule_ids

    def test_repeated_claim_signal_dutch(self):
        """Test REPEATED_CLAIM_SIGNAL rule with Dutch keywords."""
        fusion_result = {
            "normalized_claim_input": {},
            "missing_fields": [],
            "conflicts": [],
            "raw_sources": {
                "ocr_text": "",
                "florence_structured": {},
                "user_text": "Ik heb opnieuw schade aan dezelfde bumper. Vorig jaar had ik ook een vergelijkbare claim."
            }
        }

        result = self.rules.apply_rules(fusion_result, None)

        rule_ids = [r["rule_id"] for r in result["applied_rules"]]
        assert "REPEATED_CLAIM_SIGNAL" in rule_ids

        repeated_rule = next(r for r in result["applied_rules"] if r["rule_id"] == "REPEATED_CLAIM_SIGNAL")
        assert repeated_rule["severity"] == "high"

    def test_repeated_claim_signal_english(self):
        """Test REPEATED_CLAIM_SIGNAL rule with English keywords."""
        fusion_result = {
            "normalized_claim_input": {},
            "missing_fields": [],
            "conflicts": [],
            "raw_sources": {
                "ocr_text": "",
                "florence_structured": {},
                "user_text": "This is my previous claim for the same damage again."
            }
        }

        result = self.rules.apply_rules(fusion_result, None)

        rule_ids = [r["rule_id"] for r in result["applied_rules"]]
        assert "REPEATED_CLAIM_SIGNAL" in rule_ids

    def test_possible_fraud_indicators_rule(self):
        """Test POSSIBLE_FRAUD_INDICATORS rule."""
        fusion_result = {
            "normalized_claim_input": {},
            "missing_fields": [],
            "conflicts": [],
            "raw_sources": {
                "ocr_text": "Some claim text",
                "florence_structured": {},
                "user_text": ""
            }
        }

        phi4_result = {
            "claim_interpretation": {
                "fraud_indicators": [
                    "Claim submitted very quickly after incident",
                    "Damage pattern unusual"
                ]
            }
        }

        result = self.rules.apply_rules(fusion_result, phi4_result)

        rule_ids = [r["rule_id"] for r in result["applied_rules"]]
        assert "POSSIBLE_FRAUD_INDICATORS" in rule_ids

        fraud_rule = next(r for r in result["applied_rules"] if r["rule_id"] == "POSSIBLE_FRAUD_INDICATORS")
        assert fraud_rule["severity"] == "high"

    def test_extraction_failed_rule(self):
        """Test DOCUMENT_EXTRACTION_FAILED rule."""
        fusion_result = {
            "normalized_claim_input": {},
            "missing_fields": [],
            "conflicts": [],
            "raw_sources": {
                "ocr_text": "",
                "florence_structured": {},
                "user_text": ""
            }
        }

        result = self.rules.apply_rules(fusion_result, None)

        rule_ids = [r["rule_id"] for r in result["applied_rules"]]
        assert "DOCUMENT_EXTRACTION_FAILED" in rule_ids

    def test_no_rules_triggered(self):
        """Test when no rules should be triggered."""
        fusion_result = {
            "normalized_claim_input": {
                "damage_description": "Rear bumper damage",
                "damage_location": "rear",
                "accident_type": "rear_end"
            },
            "missing_fields": [],
            "conflicts": [],
            "raw_sources": {
                "ocr_text": "Damage to rear bumper",
                "florence_structured": {},
                "user_text": "Rear end collision"
            }
        }

        result = self.rules.apply_rules(fusion_result, None)

        # Should have no high/medium rules triggered
        high_medium_rules = [
            r for r in result["applied_rules"]
            if r["severity"] in ["high", "medium"]
        ]
        assert len(high_medium_rules) == 0

    def test_multiple_rules_triggered(self):
        """Test when multiple rules are triggered."""
        fusion_result = {
            "normalized_claim_input": {},
            "missing_fields": ["damage_description"],
            "conflicts": [
                {"field": "damage_location", "sources": {}}
            ],
            "raw_sources": {
                "ocr_text": "",
                "florence_structured": {},
                "user_text": ""
            }
        }

        result = self.rules.apply_rules(fusion_result, None)

        # Should have multiple rules
        assert len(result["applied_rules"]) >= 2

    # ----------------------------------------------------------------
    # /api/claims/analyze-vision contract: raw_sources must be populated
    # ----------------------------------------------------------------
    #
    # analyze_vision runs qwen3-vl:4b directly (no Tesseract, no
    # Florence-2) and returns structured JSON. The route must still
    # populate raw_sources so the rules engine's "all three sources
    # empty" guard (DOCUMENT_EXTRACTION_FAILED) does not false-trip on
    # the vision pipeline.

    def test_vision_pipeline_with_raw_sources_does_not_trip_extraction_failed(self):
        """A vision-pipeline payload with populated raw_sources must NOT trip DOCUMENT_EXTRACTION_FAILED.

        This is the contract the analyze_vision route must satisfy:
        it cannot omit raw_sources just because qwen3-vl:4b produces
        structured JSON instead of raw text.
        """
        # Mirrors the shape analyze_vision() builds after the fix
        fusion_result = {
            "normalized_claim_input": {
                "accident_date": "2026-04-12",
                "insured_name_a": "J. Doe",
                "vehicle_a": "VW Golf",
                "plate_a": "AB-123-CD",
            },
            "missing_fields": [],
            "conflicts": [],
            "source_map": {"vision": "qwen3-vl:4b"},
            "raw_sources": {
                "ocr_text": '{"accident_date": "2026-04-12"}',  # JSON dump of structured_output
                "florence_structured": {},
                "user_text": "",  # optional
            },
        }
        result = self.rules.apply_rules(fusion_result, None)
        rule_ids = [r["rule_id"] for r in result["applied_rules"]]
        assert "DOCUMENT_EXTRACTION_FAILED" not in rule_ids, (
            "Vision pipeline with populated raw_sources must NOT trip "
            "DOCUMENT_EXTRACTION_FAILED — that's the whole point of the fix."
        )

    def test_vision_pipeline_without_raw_sources_still_trips_extraction_failed(self):
        """The pre-fix shape (no raw_sources) must still trip DOCUMENT_EXTRACTION_FAILED.

        Sanity check the fix does not weaken the rule: if a caller
        forwards a raw_structures call without raw_sources, the
        extraction-failed guard must still fire.
        """
        # Pre-fix shape: missing raw_sources entirely
        fusion_result = {
            "normalized_claim_input": {"accident_date": "2026-04-12"},
            "missing_fields": [],
            "conflicts": [],
            "source_map": {"vision": "qwen3-vl:4b"},
            # raw_sources absent on purpose
        }
        result = self.rules.apply_rules(fusion_result, None)
        rule_ids = [r["rule_id"] for r in result["applied_rules"]]
        assert "DOCUMENT_EXTRACTION_FAILED" in rule_ids, (
            "Without raw_sources, the extraction-failed guard must still fire."
        )

    def test_real_extraction_failure_still_trips(self):
        """When all three sources are genuinely empty, the rule must still fire.

        Sanity check the fix does not mask real failures.
        """
        fusion_result = {
            "normalized_claim_input": {},
            "missing_fields": [],
            "conflicts": [],
            "source_map": {},
            "raw_sources": {
                "ocr_text": "",
                "florence_structured": {},
                "user_text": "",
            },
        }
        result = self.rules.apply_rules(fusion_result, None)
        rule_ids = [r["rule_id"] for r in result["applied_rules"]]
        assert "DOCUMENT_EXTRACTION_FAILED" in rule_ids
