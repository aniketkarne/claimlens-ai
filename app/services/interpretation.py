"""Shared contract between the reasoning layer and the deterministic layers.

The rules engine and the decision engine read a small, fixed set of fields
out of the reasoner's interpretation. Every producer of an interpretation
(the OCR reasoner and the vision extractor) has to supply them, otherwise
the deterministic layers degrade silently: a missing ``claim_type`` makes
every claim look unclassifiable, which routes it to manual review even when
not a single rule fired.

This module owns the field names and the normalisation rules so that
producers, consumers and the contract test cannot drift apart.
"""

from typing import Any, Dict, List

CLAIM_TYPE_FIELD = "claim_type"
FRAUD_INDICATORS_FIELD = "fraud_indicators"

UNKNOWN_CLAIM_TYPE = "unknown"

#: Values that all mean "not determined".
_UNKNOWN_CLAIM_TYPES = {"", "unknown", "none", "null", "n/a", "na"}


def normalize_claim_type(value: Any) -> str:
    """
    Normalise a claim type into a comparable string.

    Values meaning "not determined" collapse to ``"unknown"``. Anything else
    is lower-cased with spaces and hyphens turned into underscores, so
    "Motor Accident" and "motor_accident" compare equal.

    Args:
        value: Raw claim type from a reasoner.

    Returns:
        Normalised claim type.
    """
    if value is None:
        return UNKNOWN_CLAIM_TYPE

    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")

    if text in _UNKNOWN_CLAIM_TYPES:
        return UNKNOWN_CLAIM_TYPE

    return text


def normalize_fraud_indicators(value: Any) -> List[str]:
    """
    Normalise fraud indicators into a list of non-empty strings.

    Args:
        value: Raw fraud indicators from a reasoner.

    Returns:
        List of indicator strings, empty when there are none.
    """
    if value is None:
        return []

    if isinstance(value, str):
        value = [value]

    if not isinstance(value, (list, tuple, set)):
        return []

    return [str(item).strip() for item in value if str(item).strip()]


def normalize_interpretation(interpretation: Any) -> Dict[str, Any]:
    """
    Return the interpretation with the deterministic fields well-typed.

    Producers call this on the model's parsed output so that consumers can
    read the contract fields without re-checking their types.

    Args:
        interpretation: Raw interpretation from a reasoner.

    Returns:
        A copy of the interpretation with the contract fields normalised.
    """
    result: Dict[str, Any] = dict(interpretation) if isinstance(interpretation, dict) else {}

    result[CLAIM_TYPE_FIELD] = normalize_claim_type(result.get(CLAIM_TYPE_FIELD))
    result[FRAUD_INDICATORS_FIELD] = normalize_fraud_indicators(result.get(FRAUD_INDICATORS_FIELD))

    return result
