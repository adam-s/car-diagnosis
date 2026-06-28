"""Snapshot regression tests — golden files for deterministic pipeline behavior.

These let us iterate aggressively and catch *behavior drift* instantly: if a
change alters the cascade's decisions, the label logic, or the fixture-trained
model's classes, the snapshot mismatches and the test fails. Offline + CLAP-free.

Regenerate goldens intentionally with:  UPDATE_SNAPSHOTS=1 pytest tests/test_snapshots.py
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

SNAP = Path(__file__).parent / "snapshots"


def _assert_snapshot(name, value):
    SNAP.mkdir(exist_ok=True)
    f = SNAP / f"{name}.json"
    payload = json.dumps(value, indent=2, sort_keys=True)
    if os.environ.get("UPDATE_SNAPSHOTS") or not f.exists():
        f.write_text(payload)
        pytest.skip(f"wrote snapshot {name}")
    assert payload == f.read_text(), (
        f"snapshot '{name}' drifted — inspect the change; if intended, "
        f"regenerate with UPDATE_SNAPSHOTS=1")


def _fixed_tone(sr=16_000):
    """A fully deterministic signal: 0.5s quiet, 1.5s 220Hz tone, 1s quiet."""
    t = np.arange(int(sr * 3)) / sr
    y = np.zeros_like(t, dtype=np.float32)
    on = (t >= 0.5) & (t < 2.0)
    y[on] = 0.4 * np.sin(2 * np.pi * 220 * t[on]).astype(np.float32)
    return y, sr


def test_cascade_decisions_snapshot():
    from cardiag.audio.cascade import candidate_regions
    y, sr = _fixed_tone()
    regions = candidate_regions(y, sr)
    _assert_snapshot("cascade_regions",
                     {"n": len(regions), "regions": [[round(s, 1), round(e, 1)]
                                                     for s, e in regions]})


def test_label_distribution_snapshot():
    from cardiag.pipeline import build
    npz = build.FIXTURES / "embeddings.npz"
    if not npz.exists():
        pytest.skip("fixtures not built")
    z = np.load(npz, allow_pickle=True)
    rows = [{"kind": str(k), "l1": str(l), "cause": str(c)}
            for k, l, c in zip(z["kind"], z["l1"], z["cause"])]
    dist = {
        "kind": dict(sorted(Counter(r["kind"] for r in rows).items())),
        "knock": dict(sorted(Counter(
            build._knock_of(r) or "none" for r in rows).items())),
        "cause": dict(sorted(Counter(r["cause"] or "none" for r in rows).items())),
    }
    _assert_snapshot("label_distribution", dist)


def test_fixture_model_classes_snapshot(monkeypatch, tmp_path):
    monkeypatch.setenv("CARDIAG_DATA", str(tmp_path))
    import importlib

    import cardiag.paths as p
    importlib.reload(p)
    from cardiag.pipeline import build
    importlib.reload(build)
    if not build.FIXTURES.joinpath("embeddings.npz").exists():
        pytest.skip("fixtures not built")
    report = build.train_from_fixtures(min_class=2)
    classes = {head: sorted(report[head].get("classes", []))
               for head in ("kind", "knock", "cause", "triage")}
    importlib.reload(p)
    _assert_snapshot("fixture_model_classes", classes)
