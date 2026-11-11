"""Tests for the audit logger's secret redaction and record writing."""

import asyncio
import json

from app.services.audit_logger import AuditLogger


def _make_logger(tmp_path):
    logger = AuditLogger()
    logger.log_dir = tmp_path
    return logger


class TestRedaction:
    """Fields whose name looks like a secret must be redacted wherever they hide."""

    def test_secret_shaped_key_names_are_redacted(self):
        logger = _make_logger(None)
        record = {
            "access_token": "gho_abc123",
            "client_secret": "s3cr3t",
            "X-Api-Key": "sk-live-xyz",
            "github_token": "ghp_def",
            "auth-token": "tok",
            "OPENAI_API_KEY": "sk-proj-123",
            "user_password": "hunter2",
        }
        sanitized = logger._sanitize(record)
        for key in record:
            assert sanitized[key] == "[REDACTED]", f"{key} was not redacted"

    def test_exact_legacy_key_names_still_redacted(self):
        logger = _make_logger(None)
        record = {
            "password": "x",
            "secret": "x",
            "api_key": "x",
            "apikey": "x",
            "token": "x",
            "authorization": "x",
            "credential": "x",
        }
        sanitized = logger._sanitize(record)
        for key in record:
            assert sanitized[key] == "[REDACTED]", f"{key} was not redacted"

    def test_secrets_redacted_in_nested_dicts_and_lists(self):
        logger = _make_logger(None)
        record = {
            "audit": {
                "steps": [
                    {"step": "ocr", "auth_token": "tok"},
                    {"step": "reasoning", "ok": True},
                ],
                "env": {"OLLAMA_API_KEY": "sk"},
            }
        }
        sanitized = logger._sanitize(record)
        steps = sanitized["audit"]["steps"]
        assert steps[0]["auth_token"] == "[REDACTED]"
        assert steps[0]["step"] == "ocr"
        assert steps[1] == {"step": "reasoning", "ok": True}
        assert sanitized["audit"]["env"]["OLLAMA_API_KEY"] == "[REDACTED]"

    def test_benign_claim_content_is_preserved(self):
        logger = _make_logger(None)
        record = {
            "claim_id": "abc-123",
            "damage_description": "Scratch on the front bumper",
            "damage_location": "front",
            "accident_type": "collision",
            "policy_number": "NL-42",
            "language_notes": "Dutch form",
        }
        sanitized = logger._sanitize(record)
        assert sanitized == record

    def test_original_record_not_mutated(self):
        logger = _make_logger(None)
        record = {"access_token": "tok", "claim_id": "abc"}
        logger._sanitize(record)
        assert record["access_token"] == "tok"


class TestSaveClaim:
    def test_save_claim_redacts_and_stamps_utc(self, tmp_path):
        logger = _make_logger(tmp_path)
        claim_data = {
            "claim_id": "abc-123",
            "extraction": {"api_token": "sk-live", "raw_text": "form text"},
        }
        path = asyncio.run(logger.save_claim("abc-123", claim_data))

        with open(path, "r", encoding="utf-8") as f:
            saved = json.load(f)

        assert saved["claim_id"] == "abc-123"
        assert saved["timestamp"].endswith("Z")
        assert saved["record"]["extraction"]["api_token"] == "[REDACTED]"
        assert saved["record"]["extraction"]["raw_text"] == "form text"

    def test_get_claim_log_round_trip(self, tmp_path):
        logger = _make_logger(tmp_path)
        asyncio.run(logger.save_claim("claim-1", {"status": "processed"}))
        loaded = logger.get_claim_log("claim-1")
        assert loaded["record"]["status"] == "processed"
        assert loaded["pipeline_version"] == logger.settings.pipeline_version
