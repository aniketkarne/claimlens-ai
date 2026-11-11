"""Tests for upload size limits and PII retention sweeping in DocumentLoader."""

import asyncio
import io
import os
import time
from datetime import timedelta

import pytest
from fastapi import HTTPException

from app.services.document_loader import DocumentLoader


class FakeUpload:
    """Duck-typed stand-in for fastapi.UploadFile (content_type + async read)."""

    def __init__(self, data: bytes, content_type: str = "image/jpeg"):
        self._buf = io.BytesIO(data)
        self.content_type = content_type
        self.filename = "claim_form.jpg"

    async def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)


def _make_loader(tmp_path):
    loader = DocumentLoader()
    loader.upload_dir = tmp_path
    return loader


class TestSaveUpload:
    def test_writes_file_with_extension_from_content_type(self, tmp_path):
        loader = _make_loader(tmp_path)
        path = asyncio.run(loader.save_upload(FakeUpload(b"formbytes"), "claim-1"))
        assert path == tmp_path / "claim-1.jpg"
        assert path.read_bytes() == b"formbytes"

    def test_pdf_content_type_maps_to_pdf_extension(self, tmp_path):
        loader = _make_loader(tmp_path)
        path = asyncio.run(
            loader.save_upload(FakeUpload(b"%PDF", "application/pdf"), "claim-2")
        )
        assert path == tmp_path / "claim-2.pdf"

    def test_oversized_upload_rejected_with_413_and_no_partial_file(self, tmp_path):
        loader = _make_loader(tmp_path)
        loader.max_upload_mb = 1
        big = b"x" * (1024 * 1024 + 1)
        with pytest.raises(HTTPException) as exc:
            asyncio.run(loader.save_upload(FakeUpload(big), "claim-big"))
        assert exc.value.status_code == 413
        assert not (tmp_path / "claim-big.jpg").exists()
        assert list(tmp_path.iterdir()) == []

    def test_upload_at_exact_limit_is_accepted(self, tmp_path):
        loader = _make_loader(tmp_path)
        loader.max_upload_mb = 1
        data = b"x" * (1024 * 1024)
        path = asyncio.run(loader.save_upload(FakeUpload(data), "claim-max"))
        assert path.exists()
        assert path.stat().st_size == 1024 * 1024

    def test_save_upload_sweeps_stale_artifacts(self, tmp_path):
        loader = _make_loader(tmp_path)
        loader.retention = timedelta(hours=1)
        stale = tmp_path / "old-claim.jpg"
        stale.write_bytes(b"pii")
        old = time.time() - 2 * 3600
        os.utime(stale, (old, old))

        asyncio.run(loader.save_upload(FakeUpload(b"new"), "fresh-claim"))

        assert not stale.exists()
        assert (tmp_path / "fresh-claim.jpg").exists()


class TestSweepStale:
    def test_removes_old_files_keeps_fresh(self, tmp_path):
        loader = _make_loader(tmp_path)
        old_file = tmp_path / "old.png"
        fresh_file = tmp_path / "fresh.png"
        old_file.write_bytes(b"old pii")
        fresh_file.write_bytes(b"fresh pii")
        past = time.time() - 48 * 3600
        os.utime(old_file, (past, past))

        removed = loader.sweep_stale(timedelta(hours=24))

        assert removed == [str(old_file)]
        assert not old_file.exists()
        assert fresh_file.exists()

    def test_uses_configured_retention_by_default(self, tmp_path):
        loader = _make_loader(tmp_path)
        loader.retention = timedelta(hours=24)
        old_file = tmp_path / "expired.jpg"
        old_file.write_bytes(b"pii")
        past = time.time() - 25 * 3600
        os.utime(old_file, (past, past))

        removed = loader.sweep_stale()

        assert removed == [str(old_file)]

    def test_missing_upload_dir_returns_empty(self, tmp_path):
        loader = _make_loader(tmp_path / "does-not-exist")
        assert loader.sweep_stale() == []

    def test_subdirectories_are_left_alone(self, tmp_path):
        loader = _make_loader(tmp_path)
        subdir = tmp_path / "nested"
        subdir.mkdir()
        past = time.time() - 48 * 3600
        os.utime(subdir, (past, past))

        removed = loader.sweep_stale(timedelta(hours=1))

        assert removed == []
        assert subdir.exists()


class TestUploadLimitAPI:
    """The 413 must survive the route handlers, not become a 500."""

    def test_oversized_upload_returns_413_end_to_end(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient

        from app.config import get_settings
        from app.main import app

        settings = get_settings()
        monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))
        monkeypatch.setattr(settings, "log_dir", str(tmp_path / "logs"))
        monkeypatch.setattr(settings, "max_upload_mb", 1)

        client = TestClient(app)
        resp = client.post(
            "/api/claims/analyze",
            files={"file": ("form.jpg", b"x" * (1024 * 1024 + 1), "image/jpeg")},
        )

        assert resp.status_code == 413
        assert "limit is 1 MB" in resp.json()["detail"]
        uploads = tmp_path / "uploads"
        assert not uploads.exists() or list(uploads.iterdir()) == []
