"""Regression test for the ``/api/claims/analyze-vision`` route.

``qwen_vl_extractor.extract()`` returns ``structured_output: None`` whenever
the model output cannot be parsed as JSON — which happens whenever the
vision model answers with prose instead of JSON. The route then did

    structured = vision_result.get("structured_output", {}) if vision_result else {}

``.get(key, default)`` only applies the default when the key is *absent*;
``structured_output`` was present and ``None``, so ``structured`` became
``None`` and the very next line raised

    AttributeError: 'NoneType' object has no attribute 'get'

The route builds the fusion payload from ``structured.get("accident_details",
{})``, so a single unparseable model reply turned the whole endpoint into a
500.
"""

import pytest


class _FakeVisionExtractor:
    """A vision extractor whose model output could not be parsed."""

    def extract(self, image_path):
        return {
            "source": "qwen3-vl:4b",
            "structured_output": None,
            "raw_output": "Sorry, I could not read this form.",
            "warnings": ["Failed to parse JSON from model output"],
            "latency_seconds": 1.2,
        }


@pytest.fixture
def client(monkeypatch, tmp_path):
    """A TestClient with the vision extractor replaced by the failing stub."""
    pytest.importorskip("fastapi", reason="the route needs FastAPI")
    pytest.importorskip("httpx", reason="the route needs the HTTP client")

    from fastapi.testclient import TestClient

    try:
        from app.api import routes
    except ImportError as exc:  # pragma: no cover - depends on installed extras
        pytest.skip(f"the route module needs extras that are not installed: {exc}")

    monkeypatch.setattr(routes, "get_qwen_vl", lambda: _FakeVisionExtractor())
    monkeypatch.setattr(routes.settings, "upload_dir", str(tmp_path / "uploads"))

    return TestClient(routes.router)


def test_unparseable_vision_output_does_not_crash_the_route(client):
    """The route must answer with a decision, not a 500."""
    response = client.post(
        "/claims/analyze-vision",
        files={"file": ("claim.jpg", b"fake-image-bytes", "image/jpeg")},
    )

    assert response.status_code == 200, response.text

    body = response.json()
    assert body["decision"]["recommended_action"] in (
        "auto_process",
        "request_more_info",
        "manual_review",
        "reject_not_supported",
    )
    assert body["extraction"]["vision"]["structured_output"] is None
