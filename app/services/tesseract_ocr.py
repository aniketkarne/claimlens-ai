"""Tesseract OCR service."""

import shutil
import time
from typing import Optional, Set

try:
    import pytesseract
    from pytesseract import Output
except ImportError:  # pragma: no cover - exercised on installs without Tesseract
    pytesseract = None
    Output = None

from app.config import get_settings


class TesseractOCR:
    """Handles OCR extraction using Tesseract."""

    def __init__(self):
        self.settings = get_settings()
        self._available = None
        self._installed_languages: Optional[Set[str]] = None

    def is_available(self) -> bool:
        """Check if Tesseract is available and working."""
        if self._available is not None:
            return self._available

        if pytesseract is None:
            self._available = False
            return False

        try:
            result = shutil.which("tesseract")
            if result is None:
                self._available = False
                return False

            version = pytesseract.get_tesseract_version()
            self._available = True
            return True
        except Exception:
            self._available = False
            return False

    def get_installed_languages(self) -> Set[str]:
        """
        Return the set of language packs Tesseract has installed.

        The result is cached — the installed packs do not change while the
        process runs.

        Returns:
            Set of installed language codes, empty when Tesseract cannot
            report them.
        """
        if self._installed_languages is not None:
            return self._installed_languages

        installed: Set[str] = set()

        if pytesseract is not None:
            try:
                installed = set(pytesseract.get_languages(config=""))
            except Exception:
                installed = set()

        self._installed_languages = installed
        return installed

    def _get_lang_code(self, language: str) -> str:
        """Get the Tesseract language code for the given language."""
        if language == "nl":
            return self.settings.tesseract_dutch_lang
        else:
            return self.settings.tesseract_english_lang

    def extract_text(self, image_path: str, language: str = "nl") -> dict:
        """
        Extract text from an image using Tesseract OCR.

        Args:
            image_path: Path to the image file
            language: Language code (default: 'nl' for Dutch)

        Returns:
            Dictionary with: source, raw_text, language_used, warnings, latency_ms
        """
        warnings = []
        start = time.time()

        if not self.is_available():
            return {
                "source": "tesseract",
                "raw_text": "",
                "language_used": language,
                "warnings": ["Tesseract is not available"],
                "latency_ms": 0
            }

        try:
            lang_code = self._get_lang_code(language)

            installed = self.get_installed_languages()
            if installed and lang_code not in installed:
                lang_code = self.settings.tesseract_english_lang
                warnings.append(
                    f"Language '{language}' not available, falling back to English"
                )

            text = pytesseract.image_to_string(
                image_path,
                lang=lang_code,
                config="--psm 6"
            )

            # Get detailed data for additional info
            data = pytesseract.image_to_data(
                image_path,
                lang=lang_code,
                output_type=Output.DICT
            )

            detected_lang = self._detect_language(text)
            ocr_ms = round((time.time() - start) * 1000)

            return {
                "source": "tesseract",
                "raw_text": text.strip(),
                "language_used": detected_lang if detected_lang else lang_code,
                "warnings": warnings,
                "latency_ms": ocr_ms
            }

        except Exception as e:
            warnings.append(f"OCR extraction failed: {str(e)}")
            return {
                "source": "tesseract",
                "raw_text": "",
                "language_used": language,
                "warnings": warnings,
                "latency_ms": 0
            }

    def _detect_language(self, text: str) -> Optional[str]:
        """Simple language detection based on common words."""
        if not text:
            return None

        text_lower = text.lower()

        dutch_words = [
            "de", "het", "een", "van", "en", "in", "is", "op", "te", "voor",
            "met", "zijn", "dat", "niet", "aan", "er", "maar", "om", "ook",
            "als", "dan", "naar", "bij", "door", "uit", "nog", "wel",
            "schade", "voertuig", "verzekering", "claim", "datum", "locatie",
            "beschadigd", "bumper", "voor", "achter", "zijkant"
        ]

        english_words = [
            "the", "a", "an", "is", "in", "on", "at", "to", "for", "with",
            "of", "and", "or", "but", "from", "by", "this", "that", "it",
            "damage", "vehicle", "insurance", "claim", "date", "location",
            "front", "rear", "side", "door", "hood", "trunk"
        ]

        text_words = set(text_lower.split())

        dutch_count = sum(1 for w in dutch_words if w in text_words)
        english_count = sum(1 for w in english_words if w in text_words)

        if dutch_count > english_count:
            return "nl"
        elif english_count > dutch_count:
            return "eng"

        return None

    def get_detailed_data(self, image_path: str, language: str = "nl") -> dict:
        """Get detailed OCR data including bounding boxes and confidence."""
        try:
            lang_code = self._get_lang_code(language)
            data = pytesseract.image_to_data(
                image_path,
                lang=lang_code,
                output_type=Output.DICT
            )
            return data
        except Exception as e:
            return {"error": str(e)}
