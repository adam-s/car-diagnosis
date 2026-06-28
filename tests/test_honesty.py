"""Regression tests for the two highest-severity adversary findings:
  1. re-scraping must NEVER wipe an existing corpus (data loss)
  2. a degenerate (single-class) model must NOT report a confident verdict
Both offline, no CLAP."""
from __future__ import annotations

import json

import numpy as np
import pytest

from cardiag.pipeline import build


# ---------------------------------------------------- data-loss (no truncation)
def test_write_corpus_empty_run_keeps_existing(tmp_path):
    base = tmp_path / "youtube"
    rows = [{"clip_id": "v_00", "wav": "/x.wav", "kind": "fault"},
            {"clip_id": "v_01", "wav": "/y.wav", "kind": "normal"}]
    build._write_corpus(rows, base)
    # a later FAILED scrape yields nothing -> must keep the 2 existing
    n = build._write_corpus([], base)
    assert n == 2
    kept = [json.loads(l) for l in open(base / "corpus.jsonl")]
    assert {r["clip_id"] for r in kept} == {"v_00", "v_01"}


def test_write_corpus_is_additive_and_deduped(tmp_path):
    base = tmp_path / "youtube"
    build._write_corpus([{"clip_id": "a", "wav": "/a.wav", "kind": "fault"}], base)
    n = build._write_corpus([{"clip_id": "a", "wav": "/a.wav", "kind": "fault"},
                             {"clip_id": "b", "wav": "/b.wav", "kind": "normal"}], base)
    assert n == 2                       # 'a' deduped, 'b' added; nothing lost


# ---------------------------------------------------- honesty (no fake verdict)
def test_degenerate_kind_head_yields_uncertain(tmp_path, monkeypatch):
    monkeypatch.setenv("CARDIAG_DATA", str(tmp_path))
    import importlib

    import cardiag.paths as p
    importlib.reload(p)
    importlib.reload(build)

    rng = np.random.default_rng(0)
    rows, embed = [], {}
    for i in range(40):
        cid = f"c{i}"
        # kind is ALL fault -> kind head is degenerate (single class)
        # but give knock + cause two classes each so training still succeeds
        l1 = "knocking or clunking noise" if i % 2 else "normal smooth engine idle"
        cause = "brakes" if i % 2 else "belt"
        rows.append({"clip_id": cid, "video": f"v{i}", "kind": "fault",
                     "l1": l1, "cause": cause})
        embed[cid] = rng.standard_normal(512)
    build._train_heads(rows, embed, min_class=2, cause_fn=lambda r: r.get("cause"))

    from cardiag import Classifier, Verdict
    clf = Classifier.load()
    vec = rng.standard_normal(512)
    monkeypatch.setattr("cardiag.inference.classifier.embed_windows", lambda *a, **k: vec)
    d = clf.diagnose("x.wav", clean_audio=False)
    assert d.verdict is Verdict.UNCERTAIN          # NOT a confident FAULT
    assert "single class" in d.note
    importlib.reload(p)


def test_all_degenerate_refuses_to_write_model(tmp_path, monkeypatch):
    monkeypatch.setenv("CARDIAG_DATA", str(tmp_path))
    import importlib

    import cardiag.paths as p
    importlib.reload(p)
    importlib.reload(build)
    rng = np.random.default_rng(1)
    rows = [{"clip_id": f"c{i}", "video": f"v{i}", "kind": "fault",
             "l1": "grinding noise", "cause": "brakes"} for i in range(10)]
    embed = {r["clip_id"]: rng.standard_normal(512) for r in rows}
    with pytest.raises(SystemExit):                 # every head constant -> refuse
        build._train_heads(rows, embed, min_class=2, cause_fn=lambda r: r.get("cause"))
    importlib.reload(p)
