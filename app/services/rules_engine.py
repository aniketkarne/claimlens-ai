"""Rules engine - deterministic business logic applied after Phi-4 reasoning."""

import re
from typing import Any, Dict, List, Optional

from app.services.interpretation import (
    FRAUD_INDICATORS_FIELD,
    normalize_fraud_indicators,
)


class RulesEngine:
    """
    Deterministic rules engine for insurance claim processing.

    Rules are applied AFTER Phi-4 reasoning and do NOT use any AI/LLM.

    The rules check for:
    1. Missing critical fields
    2. Source conflicts
    3. Fraud indicators from AI reasoning
    4. Unknown damage location
    5. Unknown accident type
    6. Repeated claim signals
    7. Document extraction failure
    """

    # Critical fields that must be present
    CRITICAL_FIELDS = [
        "damage_description",
        "damage_location",
        "accident_type"
    ]

    # Dutch keywords indicating repeated claims
    REPEATED_CLAIM_KEYWORDS_DUTCH = [
        "vorige claim",
        "eerder",
        "opnieuw",
        "alweer",
        "vorig jaar",
        "zelfde schade",
        "vergelijkbare claim",
        "opnieuw schade"
    ]

    # English keywords indicating repeated claims
    REPEATED_CLAIM_KEYWORDS_ENGLISH = [
        "previous claim",
        "again",
        "same damage",
        "last year",
        "similar claim",
        "repeated"
    ]

    def apply_rules(
        self,
        fusion_result: dict,
        phi4_result: Optional[dict]
    ) -> dict:
        """
        Apply deterministic business rules to the claim.

        Args:
            fusion_result: Result from the fusion layer
            phi4_result: Result from Phi-4 reasoning (optional)

        Returns:
            Dictionary containing:
            - applied_rules: List of rules that were triggered
            - rule_flags: List of rule flag names
        """
        applied_rules = []
        rule_flags = []

        normalized = fusion_result.get("normalized_claim_input", {})
        missing_fields = fusion_result.get("missing_fields", [])
        conflicts = fusion_result.get("conflicts", [])
        raw_sources = fusion_result.get("raw_sources", {})

        # Rule 1: Missing critical fields
        missing_critical = self._check_missing_critical_fields(missing_fields)
        if missing_critical:
            applied_rules.append(missing_critical)
            rule_flags.append(missing_critical["rule_id"])

        # Rule 2: Source conflicts
        if conflicts:
            conflict_rule = self._check_source_conflicts(conflicts)
            applied_rules.append(conflict_rule)
            rule_flags.append(conflict_rule["rule_id"])

        # Rule 3: Possible fraud indicators from Phi-4
        if phi4_result and phi4_result.get("claim_interpretation"):
            fraud_rule = self._check_fraud_indicators(phi4_result)
            if fraud_rule:
                applied_rules.append(fraud_rule)
                rule_flags.append(fraud_rule["rule_id"])

        # Rule 4: Unknown damage location
        damage_location = normalized.get("damage_location")
        if not damage_location or damage_location == "unknown":
            unknown_damage_rule = self._check_unknown_damage_location()
            applied_rules.append(unknown_damage_rule)
            rule_flags.append(unknown_damage_rule["rule_id"])

        # Rule 5: Unknown accident type
        accident_type = normalized.get("accident_type")
        if not accident_type or accident_type == "unknown":
            unknown_type_rule = self._check_unknown_accident_type()
            applied_rules.append(unknown_type_rule)
            rule_flags.append(unknown_type_rule["rule_id"])

        # Rule 6: Repeated claim signal
        repeated_rule = self._check_repeated_claim_signal(raw_sources)
        if repeated_rule:
            applied_rules.append(repeated_rule)
            rule_flags.append(repeated_rule["rule_id"])

        # Rule 7: Document extraction failed
        extraction_failed_rule = self._check_extraction_failed(fusion_result)
        if extraction_failed_rule:
            applied_rules.append(extraction_failed_rule)
            rule_flags.append(extraction_failed_rule["rule_id"])

        return {
            "applied_rules": applied_rules,
            "rule_flags": rule_flags
        }

    def _check_missing_critical_fields(self, missing_fields: List[str]) -> Optional[dict]:
        """Check for missing critical fields."""
        critical_missing = [
            f for f in missing_fields
            if f in self.CRITICAL_FIELDS
        ]

        if critical_missing:
            return {
                "rule_id": "MISSING_CRITICAL_FIELDS",
                "severity": "medium",
                "message": f"Critical claim information is missing: {', '.join(critical_missing)}"
            }

        return None

    def _check_source_conflicts(self, conflicts: List[dict]) -> dict:
        """Check for conflicts between data sources."""
        return {
            "rule_id": "SOURCE_CONFLICT",
            "severity": "high",
            "message": f"Conflicting information detected between OCR/user text and visual extraction ({len(conflicts)} conflicts)"
        }

    def _check_fraud_indicators(self, phi4_result: dict) -> Optional[dict]:
        """Check for fraud indicators from Phi-4 reasoning."""
        interpretation = phi4_result.get("claim_interpretation") or {}
        fraud_indicators = normalize_fraud_indicators(
            interpretation.get(FRAUD_INDICATORS_FIELD)
        )

        if fraud_indicators and len(fraud_indicators) > 0:
            return {
                "rule_id": "POSSIBLE_FRAUD_INDICATORS",
                "severity": "high",
                "message": f"Potential fraud indicators detected ({len(fraud_indicators)} items). Manual review required."
            }

        return None

    def _check_unknown_damage_location(self) -> dict:
        """Check for unknown damage location."""
        return {
            "rule_id": "UNKNOWN_DAMAGE_LOCATION",
            "severity": "medium",
            "message": "Damage location could not be determined from available data."
        }

    def _check_unknown_accident_type(self) -> dict:
        """Check for unknown accident type."""
        return {
            "rule_id": "UNKNOWN_ACCIDENT_TYPE",
            "severity": "medium",
            "message": "Accident type could not be determined from available data."
        }

    def _check_repeated_claim_signal(self, raw_sources: dict) -> Optional[dict]:
        """Check for signals of repeated/prior claims."""
        # Combine all text sources
        all_text = []

        if raw_sources.get("ocr_text"):
            all_text.append(raw_sources["ocr_text"].lower())

        if raw_sources.get("user_text"):
            all_text.append(raw_sources["user_text"].lower())

        combined_text = " ".join(all_text)

        # Check Dutch keywords
        for keyword in self.REPEATED_CLAIM_KEYWORDS_DUTCH:
            if self._contains_phrase(combined_text, keyword):
                return {
                    "rule_id": "REPEATED_CLAIM_SIGNAL",
                    "severity": "high",
                    "message": f"Repeated or similar prior claim signal detected (keyword: '{keyword}')"
                }

        # Check English keywords
        for keyword in self.REPEATED_CLAIM_KEYWORDS_ENGLISH:
            if self._contains_phrase(combined_text, keyword):
                return {
                    "rule_id": "REPEATED_CLAIM_SIGNAL",
                    "severity": "high",
                    "message": f"Repeated or similar prior claim signal detected (keyword: '{keyword}')"
                }

        return None

    @staticmethod
    def _contains_phrase(text: str, phrase: str) -> bool:
        """
        Check whether the text contains the phrase as whole words.

        A plain substring check matches far too much: "again" is a substring
        of "against", so every English claim mentioning a collision
        "against" something tripped the repeated-claim rule and forced a
        manual review.

        Args:
            text: Lower-cased text to search.
            phrase: Phrase to look for.

        Returns:
            True when the phrase occurs as a whole word (or word sequence).
        """
        if not text or not phrase:
            return False

        pattern = r"(?<!\w)" + re.escape(phrase.lower()) + r"(?!\w)"
        return re.search(pattern, text) is not None

    def _check_extraction_failed(self, fusion_result: dict) -> Optional[dict]:
        """Check if document extraction failed completely."""
        raw_sources = fusion_result.get("raw_sources", {})

        ocr_text = raw_sources.get("ocr_text", "")
        florence_structured = raw_sources.get("florence_structured", {})
        user_text = raw_sources.get("user_text", "")

        # If all sources are empty/minimal, extraction likely failed
        all_empty = (
            (not ocr_text or len(ocr_text.strip()) < 10) and
            (not florence_structured or len(florence_structured) == 0) and
            (not user_text or len(user_text.strip()) < 10)
        )

        if all_empty:
            return {
                "rule_id": "DOCUMENT_EXTRACTION_FAILED",
                "severity": "high",
                "message": "Document could not be reliably processed. No meaningful data extracted."
            }

        return None
