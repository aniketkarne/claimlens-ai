"""Tests for the Tesseract language-pack resolution.

The OCR layer used to call ``pytesseract.get_tesseract_lang_codes()``, which
does not exist in pytesseract 0.3.10. The resulting ``AttributeError`` was
swallowed by a bare ``except Exception``, so every OCR call silently fell
back to English and reported "Language 'nl' not available" even when the
Dutch language pack was installed.
"""

import pytest

from app.services import tesseract_ocr as tesseract_ocr_module
from app.services.tesseract_ocr import TesseractOCR


class _FakePytesseract:
    """Stand-in for the pytesseract module."""

    def __init__(self, installed_languages):
        self.installed_languages = list(installed_languages)
        self.calls = []

    def get_languages(self, config=""):
        return list(self.installed_languages)

    def get_tesseract_version(self):
        return "5.5.2"

    def image_to_string(self, path, lang=None, config=None):
        self.calls.append(("image_to_string", lang))
        return "Locatie: Amsterdam. Schade aan de voorzijde."

    def image_to_data(self, path, lang=None, output_type=None, config=None):
        self.calls.append(("image_to_data", lang))
        return {}


@pytest.fixture
def ocr(monkeypatch):
    """Build a TesseractOCR with a fake pytesseract installed."""
    fake = _FakePytesseract(["eng", "nld"])
    monkeypatch.setattr(tesseract_ocr_module, "pytesseract", fake)
    monkeypatch.setattr(tesseract_ocr_module, "Output", type("Output", (), {"DICT": "DICT"}))

    service = TesseractOCR()
    service._available = True  # skip the binary probe
    service._installed_languages = None
    return service


class TestLanguageResolution:
    """Language pack selection."""

    def test_dutch_pack_is_used_when_installed(self, ocr):
        """Dutch OCR must actually run with the Dutch pack."""
        result = ocr.extract_text("/tmp/claim.png", language="nl")

        langs = [lang for _, lang in tesseract_ocr_module.pytesseract.calls]
        assert langs, "pytesseract was never called"
        assert set(langs) == {"nld"}, f"expected the Dutch pack, got {langs}"
        assert result["warnings"] == []
        assert result["raw_text"] != ""

    def test_no_false_language_warning_when_pack_is_present(self, ocr):
        """The old code blamed a missing language pack for its own bug."""
        result = ocr.extract_text("/tmp/claim.png", language="nl")

        assert not any("not available" in w for w in result["warnings"])

    def test_english_request_uses_english_pack(self, ocr):
        """An English request keeps using the English pack."""
        ocr.extract_text("/tmp/claim.png", language="en")

        langs = [lang for _, lang in tesseract_ocr_module.pytesseract.calls]
        assert set(langs) == {"eng"}

    def test_falls_back_to_english_when_dutch_pack_missing(self, ocr, monkeypatch):
        """A genuinely missing Dutch pack still falls back, with a warning."""
        monkeypatch.setattr(
            tesseract_ocr_module, "pytesseract", _FakePytesseract(["eng"])
        )
        ocr._installed_languages = None

        result = ocr.extract_text("/tmp/claim.png", language="nl")

        langs = [lang for _, lang in tesseract_ocr_module.pytesseract.calls]
        assert set(langs) == {"eng"}
        assert any("not available" in w for w in result["warnings"])

    def test_unknown_language_falls_back_to_english(self, ocr):
        """An unsupported language code uses the English pack."""
        ocr.extract_text("/tmp/claim.png", language="de")

        langs = [lang for _, lang in tesseract_ocr_module.pytesseract.calls]
        assert set(langs) == {"eng"}


class TestInstalledLanguages:
    """The installed-language lookup."""

    def test_reports_installed_packs(self, ocr, monkeypatch):
        fake = _FakePytesseract(["eng", "nld", "deu"])
        monkeypatch.setattr(tesseract_ocr_module, "pytesseract", fake)
        ocr._installed_languages = None

        assert ocr.get_installed_languages() == {"eng", "nld", "deu"}

    def test_lookup_is_cached(self, ocr, monkeypatch):
        fake = _FakePytesseract(["eng", "nld"])
        monkeypatch.setattr(tesseract_ocr_module, "pytesseract", fake)
        ocr._installed_languages = None

        ocr.get_installed_languages()
        fake.get_languages = lambda config="": []  # would empty the cache

        assert ocr.get_installed_languages() == {"eng", "nld"}

    def test_lookup_failure_degrades_to_empty_set(self, ocr, monkeypatch):
        class _Broken:
            def get_languages(self, config=""):
                raise RuntimeError("tesseract not found")

        monkeypatch.setattr(tesseract_ocr_module, "pytesseract", _Broken())
        ocr._installed_languages = None

        assert ocr.get_installed_languages() == set()
