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


def test_cli_inspect_bad_audio_is_clean_error(tmp_path):
    # fuzz regression: inspect on a non-audio file -> clean message, never a traceback
    bad = tmp_path / "x.wav"
    bad.write_text("definitely not audio")
    res = runner.invoke(app, ["inspect", str(bad), "--no-clap"])
    assert res.exit_code == 1
    assert "Traceback" not in res.stdout and "Traceback" not in (res.stderr or "")


def test_cli_serve_invalid_model_fails_fast(tmp_path):
    # fuzz regression: serve --model <not a model> exits before starting uvicorn
    bad = tmp_path / "m.joblib"
    bad.write_text("not a real model")
    res = runner.invoke(app, ["serve", "--model", str(bad)])
    assert res.exit_code == 1


def test_cli_scrape_rejects_negative_counts():
    # fuzz regression: negative caps are a usage error, not a silent slice
    assert runner.invoke(app, ["scrape", "youtube", "--max-videos", "-5"]).exit_code == 2


def test_cli_ingest_listed_and_validates_kind():
    # the bring-your-own-audio command exists and rejects a bad --kind before any work
    assert "ingest" in runner.invoke(app, ["--help"]).stdout
    bad = runner.invoke(app, ["ingest", "/tmp", "--kind", "broken"])
    assert bad.exit_code != 0 and "fault" in (bad.stdout + (bad.stderr or ""))


def test_ingest_dir_missing_folder_is_clean_error():
    # ingest on a non-existent folder -> friendly SystemExit, never a traceback
    import pytest

    from cardiag.pipeline import build
    with pytest.raises(SystemExit):
        build.ingest_dir("/no/such/folder", kind="fault")


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


def _events(text):
    return [ln[len("event: "):] for ln in text.splitlines() if ln.startswith("event: ")]


def test_stream_requires_input(client):
    # neither file nor url -> a clean SSE 'error' event, not a crash
    r = client.post("/api/diagnose/stream", data={})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert "error" in _events(r.text)


def test_stream_bad_audio_emits_error_event(client, tmp_path):
    # explain() reports unreadable audio cleanly (fails at load, before any CLAP)
    bad = tmp_path / "x.wav"
    bad.write_text("not audio at all")
    with open(bad, "rb") as fh:
        r = client.post("/api/diagnose/stream", files={"file": ("x.wav", fh, "audio/wav")})
    assert r.status_code == 200
    assert "error" in _events(r.text)


def test_audio_endpoint_rejects_nonhex_and_missing(client):
    # the cached-audio server must validate ids (no path traversal) and 404 unknown
    assert client.get("/api/audio/zzzz").status_code == 400          # non-hex id
    assert client.get("/api/audio/deadbeefdeadbeef").status_code == 404  # valid hex, no file


def test_favicon_served(client):
    r = client.get("/favicon.svg")
    assert r.status_code == 200 and "svg" in r.headers["content-type"]


def test_explain_endpoint_validates_input(client):
    # no input -> 400; a non-hex audio_id -> 400 (no path traversal); both CLAP-free
    assert client.post("/api/explain", data={}).status_code == 400
    assert client.post("/api/explain", data={"audio_id": "../x"}).status_code == 400


def test_url_fetch_ssrf_guard():
    # the link-fetch feature must refuse non-media / internal / non-http URLs BEFORE
    # yt-dlp ever runs (regression guard for the SSRF fix)
    from cardiag.web import explain
    for bad in ("http://169.254.169.254/latest/meta-data/",   # cloud metadata
                "file:///etc/passwd", "http://localhost:6379/",
                "https://evil.example.com/x", "ftp://x/y", "not a url"):
        with pytest.raises(ValueError):
            explain._validate_url(bad)


def test_demo_clip_is_bundled():
    # a fresh clone must have something to diagnose offline
    from cardiag import paths
    assert paths.DEMO_CLIP.exists() and paths.DEMO_CLIP.stat().st_size > 1000


def test_explain_on_garbage_is_not_500(client, tmp_path):
    # fuzz regression: unreadable audio must NOT 500 (saliency returns available:False)
    bad = tmp_path / "x.wav"
    bad.write_bytes(b"definitely not audio" * 50)
    with open(bad, "rb") as fh:
        r = client.post("/api/explain", files={"file": ("x.wav", fh, "audio/wav")})
    assert r.status_code == 200 and r.json().get("available") is False


def test_diagnose_error_does_not_leak_temp_path(client, tmp_path, monkeypatch):
    # fuzz regression: the error body must not echo the server's temp path
    import cardiag.web.app as webmod
    webmod._classifier.cache_clear()
    monkeypatch.setattr(webmod, "_classifier", lambda: None)
    bad = tmp_path / "x.wav"
    bad.write_bytes(b"\x00" * 8)
    with open(bad, "rb") as fh:
        r = client.post("/diagnose", files={"file": ("x.wav", fh, "audio/wav")})
    assert r.status_code == 400
    err = r.json().get("error", "")
    assert "/var/folders" not in err and "/tmp" not in err and "/private" not in err


def test_saliency_degrades_on_bad_input():
    # fuzz regression: occlusion_saliency never raises on missing/unreadable audio
    from cardiag.audio import saliency
    assert saliency.occlusion_saliency("/no/such/file.wav")["available"] is False
