"""Unit tests for the from-scratch training label logic (no network, no CLAP)."""
from __future__ import annotations

from cardiag.pipeline import build


def test_knock_label_from_l1():
    assert build._knock_of({"l1": "knocking or clunking noise"}) == "knock"
    assert build._knock_of({"l1": "normal smooth engine idle"}) == "normal_idle"
    assert build._knock_of({"l1": "squealing or squeaking noise"}) is None


def test_cause_label_from_title_keywords():
    # l2_candidates are produced by the title keyword map at scrape time
    assert build._cause_of({"l2_candidates": ["brakes"]}) == "brakes"
    assert build._cause_of({"l2_candidates": ["wheel bearing"]}) == "wheel_bearing"
    assert build._cause_of({"l2_candidates": []}) is None


def test_triage_label_only_for_faults():
    assert build._triage_of({"kind": "fault", "l2_candidates": ["brakes"]}) == "chassis"
    assert build._triage_of({"kind": "fault", "l2_candidates": ["belt"]}) == "engine"
    assert build._triage_of({"kind": "normal", "l2_candidates": ["brakes"]}) is None


def test_fit_degenerate_single_class_returns_dummy():
    import numpy as np
    rows = [{"clip_id": f"c{i}", "video": f"v{i}", "k": "fault"} for i in range(5)]
    embed = {r["clip_id"]: np.zeros(512) for r in rows}
    clf, report = build._fit(rows, lambda r: r["k"], embed, min_class=1)
    assert report["degenerate"] is True
    assert clf.predict([np.zeros(512)])[0] == "fault"


def test_fit_two_classes_trains_real_head():
    import numpy as np
    rng = np.random.default_rng(0)
    rows, embed = [], {}
    for i in range(40):
        cid = f"c{i}"
        lbl = "fault" if i % 2 else "normal"
        rows.append({"clip_id": cid, "video": f"v{i}", "k": lbl})
        # separable embeddings so the head is well-defined
        embed[cid] = rng.standard_normal(512) + (3 if lbl == "fault" else -3)
    clf, report = build._fit(rows, lambda r: r["k"], embed, min_class=2)
    assert set(report["classes"]) == {"fault", "normal"}
    assert hasattr(clf, "predict_proba")
