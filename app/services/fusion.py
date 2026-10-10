"""Fusion layer - combines OCR and user text into normalized structure."""

from typing import Any, Dict, List, Optional, Tuple


class FusionEngine:
    """
    Fusion layer that combines outputs from:
    - Tesseract OCR
    - User-provided text

    This is deterministic logic, NOT an AI model.

    The ``florence_output`` parameter is preserved for backward compatibility
    with callers that still pass a vision-source dict, but it is ignored
    (Florence-2 was removed from the pipeline — see README §"Model Evaluation
    History" and journey.md §"Removed").
    """

    # Critical fields that must be present for a valid claim
    CRITICAL_FIELDS = [
        "damage_description",
        "damage_location",
        "accident_type"
    ]

    # All fields we try to extract
    ALL_FIELDS = [
        "incident_date",
        "location",
        "damage_description",
        "damage_location",
        "accident_type",
        "vehicle_a_make",
        "vehicle_a_model",
        "vehicle_a_license_plate",
        "vehicle_b_make",
        "vehicle_b_model",
        "vehicle_b_license_plate"
    ]

    def fuse(
        self,
        ocr_output: Optional[dict],
        florence_output: Optional[dict] = None,
        user_text: Optional[str] = None
    ) -> dict:
        """
        Fuse OCR and user text into normalized claim input.

        The ``florence_output`` argument is accepted for backward
        compatibility with earlier callers but is ignored — Florence-2 was
        removed from the pipeline. Pass ``None`` (the default).

        Args:
            ocr_output: Tesseract OCR result
            florence_output: Ignored. Retained as a kwarg for API stability.
            user_text: Optional user-provided text

        Returns:
            Dictionary containing:
            - normalized_claim_input: The fused structured claim
            - missing_fields: List of fields that couldn't be extracted
            - conflicts: List of conflicts between sources
            - source_map: Mapping of fields to their sources
            - raw_sources: Preserved raw outputs from each source
        """
        # Florence-2 removed from the active pipeline. Force the param to
        # None so downstream logic (and the rules engine's
        # _check_extraction_failed) sees a deterministic, well-defined input.
        florence_output = None

        # Initialize result structures
        normalized = {
            "incident_date": None,
            "location": None,
            "vehicle_a": {},
            "vehicle_b": {},
            "damage_description": None,
            "damage_location": None,
            "accident_type": None,
            "user_description": user_text,
            "document_notes": []
        }

        missing_fields = []
        conflicts = []
        source_map: Dict[str, List[str]] = {}
        raw_sources = {
            "ocr_text": "",
            "florence_structured": {},
            "user_text": user_text or ""
        }

        # Extract from OCR
        ocr_text = ""
        if ocr_output and ocr_output.get("raw_text"):
            ocr_text = ocr_output["raw_text"]
            raw_sources["ocr_text"] = ocr_text

        # Extract information from each source. "florence" is now always
        # empty — the rules engine's _check_extraction_failed handles the
        # (ocr == "" AND user_text == "") case explicitly.
        sources_data = {
            "ocr": self._parse_ocr_text(ocr_text),
            "user_text": self._parse_user_text(user_text or "")
        }

        # Merge data with source attribution
        for field in self.ALL_FIELDS:
            sources_with_value = []

            for source_name, source_data in sources_data.items():
                if field in source_data and source_data[field]:
                    sources_with_value.append((source_name, source_data[field]))

            if len(sources_with_value) == 0:
                missing_fields.append(field)
            elif len(sources_with_value) == 1:
                # Single source - use it
                source_name, value = sources_with_value[0]
                self._set_field(normalized, field, value)
                source_map[field] = [source_name]
            else:
                # Multiple sources - check for conflicts
                values = [v for _, v in sources_with_value]
                sources = [s for s, _ in sources_with_value]

                # Check if values are consistent (allow some variation)
                if self._values_match(values):
                    # Values are consistent - use first source
                    self._set_field(normalized, field, values[0])
                    source_map[field] = sources
                else:
                    # Conflict detected
                    conflict = self._create_conflict(
                        field,
                        sources_with_value,
                        sources_data
                    )
                    conflicts.append(conflict)
                    source_map[field] = sources

        return {
            "normalized_claim_input": normalized,
            "missing_fields": missing_fields,
            "conflicts": [c.model_dump() for c in conflicts],
            "source_map": source_map,
            "raw_sources": raw_sources
        }

    def _parse_ocr_text(self, text: str) -> Dict[str, Any]:
        """
        Parse OCR text to extract structured information.

        This is a simple heuristic-based extraction.
        """
        result = {}

        if not text:
            return result

        text_lower = text.lower()

        # Date patterns (Dutch and English)
        import re

        date_patterns = [
            r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',  # DD-MM-YYYY or DD/MM/YYYY
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',    # YYYY-MM-DD
            r'\d{1,2}\s+(januari|februari|maart|april|mei|juni|juli|augustus|september|oktober|november|december)\s+\d{4}',  # Dutch
        ]

        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["incident_date"] = match.group()
                break

        # Location indicators
        location_indicators = ["locatie", "location", "adres", "address", "plaats", "place"]
        for indicator in location_indicators:
            if indicator in text_lower:
                # Try to extract the next few words after the indicator
                idx = text_lower.find(indicator)
                after = text[idx:idx+100].split('\n')[0]
                if len(after) > len(indicator) + 2:
                    result["location"] = after[len(indicator)+1:].strip()
                    break

        # Damage location patterns
        damage_location_indicators = [
            "voorzijde", "achterzijde", "zijkant", "links", "rechts",
            "front", "rear", "side", "left", "right",
            "bumper", "deur", "motorkap", "kofferbak"
        ]
        for indicator in damage_location_indicators:
            if indicator in text_lower:
                result["damage_location"] = indicator
                break

        # Damage description
        damage_desc_indicators = ["schade", "beschadigd", "damage", "beschrijving"]
        for indicator in damage_desc_indicators:
            if indicator in text_lower:
                idx = text_lower.find(indicator)
                after = text[idx:idx+200].split('\n')[0]
                if len(after) > len(indicator) + 2:
                    result["damage_description"] = after.strip()
                    break

        # Accident type
        accident_types = {
            "rear_end": ["aanrijding van achteren", "rear-end", "rear end", "kettingbotsing"],
            "frontal": ["frontal", "frontaal", "front-on"],
            "side_impact": ["zijdelingse", "side impact", "flank"],
            "parking": ["parkeer", "parking", "parkeerincident"],
            "hit_and_run": ["hit and run", "doorrijding", "onbekende dader"]
        }

        for accident_type, keywords in accident_types.items():
            for keyword in keywords:
                if keyword in text_lower:
                    result["accident_type"] = accident_type
                    break
            if "accident_type" in result:
                break

        return result

    def _parse_user_text(self, text: str) -> Dict[str, Any]:
        """
        Parse user-provided text to extract structured information.
        """
        result = {}

        if not text:
            return result

        # User text is treated similarly to OCR text
        # This can be enhanced with more sophisticated NLP
        return self._parse_ocr_text(text)

    def _set_field(self, normalized: dict, field: str, value: Any) -> None:
        """Set a field in the normalized structure."""
        if field.startswith("vehicle_a_"):
            vehicle_field = field.replace("vehicle_a_", "")
            if "vehicle_a" not in normalized:
                normalized["vehicle_a"] = {}
            normalized["vehicle_a"][vehicle_field] = value
        elif field.startswith("vehicle_b_"):
            vehicle_field = field.replace("vehicle_b_", "")
            if "vehicle_b" not in normalized:
                normalized["vehicle_b"] = {}
            normalized["vehicle_b"][vehicle_field] = value
        else:
            normalized[field] = value

    def _values_match(self, values: List[str]) -> bool:
        """
        Check if multiple values can be considered matching.

        Allow for some variation since different sources may phrase things differently.
        """
        if len(values) <= 1:
            return True

        # Normalize values for comparison
        normalized_values = [v.lower().strip() for v in values]

        # Check exact match
        if len(set(normalized_values)) == 1:
            return True

        # Check if one value contains the other
        for i, v1 in enumerate(normalized_values):
            for v2 in normalized_values[i+1:]:
                if v1 in v2 or v2 in v1:
                    return True
                # Check for common subwords
                words1 = set(v1.split())
                words2 = set(v2.split())
                if words1 & words2:  # Intersection
                    return True

        return False

    def _create_conflict(
        self,
        field: str,
        sources_with_value: List[Tuple[str, str]],
        all_source_data: Dict[str, Dict]
    ) -> dict:
        """Create a conflict record for the given field."""
        sources_dict = {}
        for source_name, value in sources_with_value:
            sources_dict[source_name] = value

        message = f"Conflicting values for '{field}' detected between sources."

        return {
            "field": field,
            "sources": sources_dict,
            "message": message
        }
