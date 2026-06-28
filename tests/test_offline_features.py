"""Offline tests for the novice-UX features: fixtures training + inspect report.
Both run with no network and no CLAP download."""
from __future__ import annotations

import joblib
import pytest


def test_train_from_fixtures_offline(monkeypatch, tmp_path):
    """Bundled embeddings -> a real model, no scrape, no CLAP."""
    monkeypatch.setenv("CARDIAG_DATA", str(tmp_path))
    import importlib

    import cardiag.paths as p
    importlib.reload(p)
    from cardiag.pipeline import build
    importlib.reload(build)

    if not build.FIXTURES.joinpath("embeddings.npz").exists():
        pytest.skip("fixture embeddings not built")

    report = build.train_from_fixtures(min_class=2)
    assert "kind" in report and "cause" in report and "triage" in report
    art = joblib.load(p.MODEL_CLAP)
    assert set(art["heads"]) == {"kind", "knock", "cause"}
    assert "fault" in art["heads"]["kind"].classes_
    triage = joblib.load(p.MODEL_TRIAGE)
    assert set(triage["classes"]) <= {"engine", "chassis"}
    importlib.reload(p)


def test_inspect_report_offline(tone_wav, tmp_path):
    """inspect renders a self-contained HTML (no_clap path = no CLAP needed)."""
    pytest.importorskip("matplotlib")
    from cardiag import inspect as inspect_mod
    out = tmp_path / "r.html"
    inspect_mod.report([tone_wav], out_path=str(out), with_clap=False)
    html = out.read_text()
    assert "pipeline inspection" in html
    assert "data:image/png;base64," in html      # spectrogram embedded
    assert "data:audio/wav;base64," in html       # audio player embedded
    assert "kept" in html                          # cascade decision line
