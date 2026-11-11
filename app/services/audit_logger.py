"""Audit logger - persists claim records as JSON files."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from app.config import get_settings


class AuditLogger:
    """Handles audit logging for claim processing."""

    def __init__(self):
        self.settings = get_settings()
        self.log_dir = self.settings.claims_log_path

    async def save_claim(self, claim_id: str, claim_data: Dict[str, Any]) -> str:
        """
        Save a complete claim record as a JSON file.

        Args:
            claim_id: Unique claim identifier
            claim_data: Full claim processing result

        Returns:
            Path to the saved audit log file
        """
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Add metadata
        audit_record = {
            "claim_id": claim_id,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "pipeline_version": self.settings.pipeline_version,
            "record": claim_data
        }

        # Remove sensitive fields if any
        audit_record = self._sanitize(audit_record)

        # Save to file
        filename = f"{claim_id}.json"
        file_path = self.log_dir / filename

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(audit_record, f, indent=2, default=str)

        return str(file_path)

    def _sanitize(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove sensitive information from audit record.

        Any field whose name contains a secret-shaped word is redacted,
        in any nesting, in dicts and in lists. Exact-name matching is not
        enough: fields like ``access_token``, ``client_secret`` or
        ``X-Api-Key`` carry secrets too and used to pass through untouched.

        Args:
            record: The audit record to sanitize

        Returns:
            Sanitized record
        """
        # Make a copy to avoid modifying original
        sanitized = json.loads(json.dumps(record, default=str))

        # Remove any field that looks like a secret
        secrets_patterns = [
            "password", "secret", "apikey", "token",
            "authorization", "credential"
        ]

        def is_secret_key(key: str) -> bool:
            # Compare on a normalized form so api_key / api-key / apiKey
            # all match the same pattern.
            normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
            return any(pattern in normalized for pattern in secrets_patterns)

        def remove_secrets(obj):
            if isinstance(obj, dict):
                return {
                    k: "[REDACTED]" if is_secret_key(k) else remove_secrets(v)
                    for k, v in obj.items()
                }
            elif isinstance(obj, list):
                return [remove_secrets(item) for item in obj]
            else:
                return obj

        return remove_secrets(sanitized)

    def get_claim_log(self, claim_id: str) -> Dict[str, Any]:
        """
        Retrieve a claim audit log by ID.

        Args:
            claim_id: The claim ID to look up

        Returns:
            The audit log record

        Raises:
            FileNotFoundError: If the claim log doesn't exist
        """
        filename = f"{claim_id}.json"
        file_path = self.log_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Claim log not found: {claim_id}")

        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_claims(self, limit: int = 100) -> list:
        """
        List recent claim audit logs.

        Args:
            limit: Maximum number of logs to return

        Returns:
            List of claim IDs and timestamps
        """
        if not self.log_dir.exists():
            return []

        log_files = sorted(
            self.log_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )[:limit]

        results = []
        for file_path in log_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    results.append({
                        "claim_id": data.get("claim_id"),
                        "timestamp": data.get("timestamp"),
                        "status": data.get("record", {}).get("status")
                    })
            except Exception:
                results.append({
                    "claim_id": file_path.stem,
                    "error": "Failed to parse"
                })

        return results
