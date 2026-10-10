"""Decision engine - converts rules and reasoning into final workflow decision."""

from typing import Any, Dict, List, Optional


class DecisionEngine:
    """
    Deterministic decision engine for insurance claim processing.

    Converts rules and AI reasoning into a final workflow decision.

    IMPORTANT:
    - "auto_process" does NOT mean payment approval
    - "auto_process" means the claim can proceed to the next automated workflow step
    - Final payment decisions always require human review
    """

    def decide(
        self,
        rules_result: dict,
        phi4_result: Optional[dict]
    ) -> dict:
        """
        Make the final decision on how to process the claim.

        Decision logic:
        1. If any HIGH severity rule exists:
           - risk_level = "high"
           - manual_review_required = true
           - recommended_action = "manual_review"

        2. Else if any MEDIUM severity rule exists:
           - risk_level = "medium"
           - manual_review_required = true
           - recommended_action = "request_more_info"

        3. Else if claim_type is unknown:
           - risk_level = "unknown"
           - manual_review_required = true
           - recommended_action = "request_more_info"

        4. Else:
           - risk_level = "low"
           - manual_review_required = false
           - recommended_action = "auto_process"

        Args:
            rules_result: Result from the rules engine
            phi4_result: Result from the active reasoning layer (named
                ``phi4_result`` for backward compatibility — the active
                reasoner is now qwen2.5:7b).

        Returns:
            Dictionary containing:
            - risk_level: "low", "medium", "high", or "unknown"
            - manual_review_required: true or false
            - recommended_action: "auto_process", "request_more_info", "manual_review", or "reject_not_supported"
            - decision_reasons: List of reasons for the decision
        """
        applied_rules = rules_result.get("applied_rules", [])
        decision_reasons = []

        # Determine highest severity
        highest_severity = self._get_highest_severity(applied_rules)

        # Make decision based on severity
        if highest_severity == "high":
            return self._make_high_risk_decision(applied_rules, decision_reasons)

        elif highest_severity == "medium":
            return self._make_medium_risk_decision(applied_rules, decision_reasons)

        else:
            # Check if claim_type is unknown
            claim_type_unknown = self._check_claim_type_unknown(phi4_result)

            if claim_type_unknown:
                return self._make_unknown_claim_type_decision(
                    applied_rules,
                    decision_reasons
                )

            # Default: low risk
            return self._make_low_risk_decision(applied_rules, decision_reasons)

    def _get_highest_severity(self, applied_rules: List[dict]) -> str:
        """Get the highest severity from applied rules."""
        severities = [rule.get("severity", "low") for rule in applied_rules]

        if "high" in severities:
            return "high"
        elif "medium" in severities:
            return "medium"
        else:
            return "low"

    def _check_claim_type_unknown(self, phi4_result: Optional[dict]) -> bool:
        """Check if the claim type is unknown."""
        if not phi4_result:
            return True

        interpretation = phi4_result.get("claim_interpretation")
        if not interpretation:
            return True

        claim_type = interpretation.get("claim_type", "unknown")
        return claim_type == "unknown" or claim_type is None

    def _make_high_risk_decision(
        self,
        applied_rules: List[dict],
        decision_reasons: List[str]
    ) -> dict:
        """Make decision when high severity rules exist."""
        decision_reasons.append("High severity rules triggered - manual review required")

        for rule in applied_rules:
            if rule.get("severity") == "high":
                decision_reasons.append(f"HIGH: {rule.get('message', rule.get('rule_id'))}")

        # Check if we should reject instead of manual review
        for rule in applied_rules:
            if rule.get("rule_id") == "DOCUMENT_EXTRACTION_FAILED":
                return {
                    "risk_level": "high",
                    "manual_review_required": True,
                    "recommended_action": "reject_not_supported",
                    "decision_reasons": decision_reasons + [
                        "Document could not be processed - rejecting claim"
                    ]
                }

        return {
            "risk_level": "high",
            "manual_review_required": True,
            "recommended_action": "manual_review",
            "decision_reasons": decision_reasons
        }

    def _make_medium_risk_decision(
        self,
        applied_rules: List[dict],
        decision_reasons: List[str]
    ) -> dict:
        """Make decision when medium severity rules exist."""
        decision_reasons.append("Medium severity rules triggered - additional information needed")

        for rule in applied_rules:
            if rule.get("severity") == "medium":
                decision_reasons.append(f"MEDIUM: {rule.get('message', rule.get('rule_id'))}")

        return {
            "risk_level": "medium",
            "manual_review_required": True,
            "recommended_action": "request_more_info",
            "decision_reasons": decision_reasons
        }

    def _make_unknown_claim_type_decision(
        self,
        applied_rules: List[dict],
        decision_reasons: List[str]
    ) -> dict:
        """Make decision when claim type is unknown."""
        decision_reasons.append("Claim type could not be determined - additional information needed")

        return {
            "risk_level": "unknown",
            "manual_review_required": True,
            "recommended_action": "request_more_info",
            "decision_reasons": decision_reasons
        }

    def _make_low_risk_decision(
        self,
        applied_rules: List[dict],
        decision_reasons: List[str]
    ) -> dict:
        """Make decision for low risk claims."""
        if applied_rules:
            decision_reasons.append(f"Claim passed all rules checks ({len(applied_rules)} low-severity notes)")
        else:
            decision_reasons.append("No rules triggered - claim suitable for auto-processing")

        decision_reasons.append("IMPORTANT: auto_process does not mean payment approval")
        decision_reasons.append("Claim can proceed to next automated workflow step")

        return {
            "risk_level": "low",
            "manual_review_required": False,
            "recommended_action": "auto_process",
            "decision_reasons": decision_reasons
        }
