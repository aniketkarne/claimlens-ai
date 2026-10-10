"""Configuration management for ClaimLens AI."""

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    # Default to qwen2.5:7b — the only text-reasoning model documented as
    # active in README §"Model Evaluation History" and journey.md. The prior
    # default (gemma4:e4b) was explicitly removed because it hallucinated
    # fields on EU accident forms.
    ollama_model: str = "qwen2.5:7b"

    # Tesseract Configuration
    tesseract_dutch_lang: str = "nld"
    tesseract_english_lang: str = "eng"

    # File Storage
    upload_dir: str = "uploads"
    log_dir: str = "logs"

    # Language Settings
    default_language: str = "nl"

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Pipeline Version
    pipeline_version: str = "0.1.0"

    @property
    def upload_path(self) -> Path:
        """Get the upload directory path."""
        return Path(self.upload_dir)

    @property
    def log_path(self) -> Path:
        """Get the log directory path."""
        return Path(self.log_dir)

    @property
    def claims_log_path(self) -> Path:
        """Get the claims audit log directory path."""
        return self.log_path / "claims"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the settings instance."""
    return settings
