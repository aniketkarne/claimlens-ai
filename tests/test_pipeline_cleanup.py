"""Tests that pin the contract of the post-cleanup pipeline.

These tests are deliberately small and assert exactly the things this
audit changed — they fail loudly if someone re-introduces Florence-2 /
Phi-4 or flips the default model back to a removed name.
"""

import pytest

from app.config import get_settings


class TestConfigDefaults:
    """The default Ollama model must match the active model in the README."""

    def test_default_ollama_model_is_qwen25_7b(self):
        """The default must be qwen2.5:7b — gemma4:e4b and phi4 were removed."""
        s = get_settings()
        assert s.ollama_model == "qwen2.5:7b"

    def test_florence_settings_removed(self):
        """The Florence-2 setting was removed with the model."""
        s = get_settings()
        assert not hasattr(s, "florence_model_name"), (
            "florence_model_name leaked back into Settings — the model "
            "was removed from the pipeline (see README §Model Evaluation "
            "History)."
        )


class TestFusionPipeline:
    """The fusion layer must not consume Florence-2 data."""

    def test_florence_kwarg_is_ignored(self):
        from app.services.fusion import FusionEngine

        eng = FusionEngine()
        result = eng.fuse(
            ocr_output={"raw_text": "JAAR: 2025", "language_used": "nl"},
            florence_output={
                "structured_output": {
                    "visible_damage_area": "this must NOT appear in raw_sources",
                }
            },
            user_text=None,
        )
        # Florence-2 data is intentionally dropped
        assert result["raw_sources"]["florence_structured"] == {}

    def test_florence_does_not_contribute_fields(self):
        from app.services.fusion import FusionEngine

        eng = FusionEngine()
        # Florence says damage is at the "rear"; OCR is empty; user text
        # is empty. With Florence out of the pipeline, no source is
        # providing damage_location, so it must be missing.
        result = eng.fuse(
            ocr_output={"raw_text": "", "language_used": "nl"},
            florence_output={"structured_output": {"visible_damage_area": "rear"}},
            user_text=None,
        )
        assert result["normalized_claim_input"].get("damage_location") is None


class TestDeadCodeRemoved:
    """Phi-4 and Florence-2 modules are gone — pin that."""

    def test_phi4_reasoner_module_removed(self):
        with pytest.raises(ModuleNotFoundError):
            import app.services.phi4_reasoner  # noqa: F401

    def test_florence_extractor_module_removed(self):
        with pytest.raises(ModuleNotFoundError):
            import app.services.florence_extractor  # noqa: F401

    def test_phi4_prompts_module_removed(self):
        with pytest.raises(ModuleNotFoundError):
            import app.prompts.phi4_claim_reasoning_prompt  # noqa: F401

    def test_florence_prompts_module_removed(self):
        with pytest.raises(ModuleNotFoundError):
            import app.prompts.florence_extraction_prompts  # noqa: F401


class TestRoutesNoFlorence:
    """The route module must not import the removed extractors."""

    def test_routes_does_not_import_phi4(self):
        import app.api.routes as routes

        # The Phi4Reasoner symbol is gone from the module
        assert not hasattr(routes, "Phi4Reasoner")
        assert not hasattr(routes, "get_phi4")

    def test_routes_does_not_import_florence(self):
        import app.api.routes as routes

        assert not hasattr(routes, "FlorenceExtractor")
        assert not hasattr(routes, "get_florence")
