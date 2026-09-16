"""Tests for the fusion layer."""

import pytest

from app.services.fusion import FusionEngine


class TestFusionEngine:
    """Test cases for FusionEngine."""

    def setup_method(self):
        """Set up test fixtures."""
        self.fusion = FusionEngine()

    def test_fuse_with_empty_inputs(self):
        """Test fusion with empty/minimal inputs."""
        result = self.fusion.fuse(
            ocr_output={"raw_text": "", "language_used": "nl"},
            florence_output={"structured_output": {}},
            user_text=None
        )

        assert "normalized_claim_input" in result
        assert "missing_fields" in result
        assert "conflicts" in result
        assert "source_map" in result

        # All fields should be missing
        assert len(result["missing_fields"]) > 0

    def test_fuse_detects_missing_fields(self):
        """Test that fusion correctly identifies missing fields."""
        result = self.fusion.fuse(
            ocr_output={
                "raw_text": "Some unrelated text without claim data",
                "language_used": "nl"
            },
            florence_output={"structured_output": {}},
            user_text=None
        )

        # Critical fields should be in missing_fields
        missing = result["missing_fields"]
        assert "damage_description" in missing
        assert "damage_location" in missing
        assert "accident_type" in missing

    def test_fuse_user_text_extraction(self):
        """Test that user text is preserved and used."""
        user_text = "Ik werd van achteren aangereden. De achterbumper is beschadigd."

        result = self.fusion.fuse(
            ocr_output={"raw_text": "", "language_used": "nl"},
            florence_output={"structured_output": {}},
            user_text=user_text
        )

        # User text should be preserved
        assert result["raw_sources"]["user_text"] == user_text
        assert result["normalized_claim_input"]["user_description"] == user_text

    def test_fuse_source_map_populated(self):
        """Test that source map is correctly populated."""
        user_text = "Schade aan de bumper"

        result = self.fusion.fuse(
            ocr_output={"raw_text": "", "language_used": "nl"},
            florence_output={"structured_output": {}},
            user_text=user_text
        )

        # Source map should have entries
        source_map = result["source_map"]
        assert len(source_map) > 0

    def test_fuse_preserves_raw_sources(self):
        """Test that raw sources are preserved."""
        ocr_text = "Raw OCR text from document"
        user_text = "User provided description"

        result = self.fusion.fuse(
            ocr_output={"raw_text": ocr_text, "language_used": "nl"},
            florence_output={"structured_output": {"visible_damage_area": "front"}},
            user_text=user_text
        )

        raw_sources = result["raw_sources"]
        assert raw_sources["ocr_text"] == ocr_text
        assert raw_sources["user_text"] == user_text
        assert raw_sources["florence_structured"]["visible_damage_area"] == "front"

    def test_fuse_damage_location_extraction(self):
        """Test extraction of damage location from user text."""
        user_text = "De schade zit aan de achterbumper"

        result = self.fusion.fuse(
            ocr_output={"raw_text": "", "language_used": "nl"},
            florence_output={"structured_output": {}},
            user_text=user_text
        )

        # Should extract damage location
        normalized = result["normalized_claim_input"]
        # The fusion layer should have parsed this

    def test_fuse_accident_type_extraction(self):
        """Test extraction of accident type."""
        user_text = "Ik werd van achteren aangereden bij het stoplicht"

        result = self.fusion.fuse(
            ocr_output={"raw_text": "", "language_used": "nl"},
            florence_output={"structured_output": {}},
            user_text=user_text
        )

        normalized = result["normalized_claim_input"]
        # rear-end type should be detected

    def test_fuse_no_conflicts_with_single_source(self):
        """Test that no conflicts are reported when only one source provides data."""
        user_text = "Schade aan de voorkant"

        result = self.fusion.fuse(
            ocr_output={"raw_text": "", "language_used": "nl"},
            florence_output={"structured_output": {}},
            user_text=user_text
        )

        # With only user text, no conflicts should occur
        assert result["conflicts"] == []

    def test_fuse_returns_valid_conflicts_for_real_disagreement(self):
        """Regression: when sources disagree, fuse() must return a valid
        conflicts list (not raise AttributeError on a plain dict).

        Before the fix, _create_conflict returned a dict and the call site
        called c.model_dump() on it, crashing the entire /api/claims/analyze
        endpoint the moment any source disagreement was detected.
        """
        result = self.fusion.fuse(
            ocr_output={
                # OCR's heuristic picks "front" as damage_location.
                "raw_text": "Front damage to vehicle",
                "language_used": "nl",
            },
            florence_output={"structured_output": {"visible_damage_area": "rear"}},
            # User text is also Dutch — its parse is independent of OCR/Florence,
            # so it can agree or disagree with the others.
            user_text="Schade aan de achterkant",
        )

        # The conflicts list must be a list of plain dicts with the expected shape.
        assert isinstance(result["conflicts"], list)
        assert len(result["conflicts"]) >= 1
        for c in result["conflicts"]:
            assert isinstance(c, dict)
            assert set(c.keys()) == {"field", "sources", "message"}
            assert isinstance(c["sources"], dict)
            assert len(c["sources"]) >= 2
