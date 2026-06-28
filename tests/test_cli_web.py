"""Contract tests for the CLI (Typer) and the web app (FastAPI TestClient)."""
from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from cardiag.cli import app

runner = CliRunner()


# ------------------------------------------------------------------- CLI
def test_cli_version():
    res = runner.invoke(app, ["version"])
    assert res.exit_code == 0
    assert "0.1.0" in res.stdout


def test_cli_help_lists_commands():
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    for cmd in ("diagnose", "triage", "clean", "serve", "scrape", "train"):
        assert cmd in res.stdout


def test_cli_diagnose_missing_model_errors(tone_wav):
    res = runner.invoke(app, ["diagnose", tone_wav, "--model", "/no/such.joblib"])
    assert res.exit_code != 0


def test_cli_clean_outputs_json(tone_wav):
    res = runner.invoke(app, ["clean", tone_wav, "--no-music-gate"])
    assert res.exit_code == 0
    payload = json.loads(res.stdout)
    assert "segments" in payload and payload["is_empty"] is False


# ------------------------------------------------------------------- web
@pytest.fixture
def client():
    pytest.importorskip("fastapi", reason="install the [web] extra to run web tests")
    from fastapi.testclient import TestClient

    from cardiag.web.app import app as web_app
    return TestClient(web_app)


def test_web_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "cardiag" in r.text


def test_web_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_web_rejects_non_audio_with_400(client, tmp_path, monkeypatch):
    import cardiag.web.app as webmod
    webmod._classifier.cache_clear()
    monkeypatch.setattr(webmod, "_classifier", lambda: None)
    bad = tmp_path / "x.wav"
    bad.write_text("definitely not audio")
    with open(bad, "rb") as fh:
        r = client.post("/diagnose", files={"file": ("x.wav", fh, "audio/wav")})
    assert r.status_code == 400 and "error" in r.json()


def test_web_rejects_empty_upload_with_400(client, tmp_path, monkeypatch):
    import cardiag.web.app as webmod
    webmod._classifier.cache_clear()
    monkeypatch.setattr(webmod, "_classifier", lambda: None)
    empty = tmp_path / "e.wav"
    empty.write_bytes(b"")
    with open(empty, "rb") as fh:
        r = client.post("/diagnose", files={"file": ("e.wav", fh, "audio/wav")})
    assert r.status_code == 400


def test_web_diagnose_endpoint_cleaning_path(client, tone_wav, monkeypatch):
    # force "no model loaded" so the endpoint returns the cleaning result
    # (deterministic, CLAP-free)
    import cardiag.web.app as webmod
    webmod._classifier.cache_clear()
    monkeypatch.setattr(webmod, "_classifier", lambda: None)
    with open(tone_wav, "rb") as fh:
        r = client.post("/diagnose", files={"file": ("tone.wav", fh, "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is False
    assert "cleaning" in body
