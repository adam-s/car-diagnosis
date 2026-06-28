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
    clf, report, temp = build._fit(rows, lambda r: r["k"], embed, min_class=1)
    assert report["degenerate"] is True
    assert temp == 1.0                       # no calibration for a constant head
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
    clf, report, temp = build._fit(rows, lambda r: r["k"], embed, min_class=2)
    assert set(report["classes"]) == {"fault", "normal"}
    assert hasattr(clf, "predict_proba")
    assert "cv_bal_acc" in report and report["cv_folds"] > 0   # honest grouped-CV report
    assert temp > 0                                            # a fitted temperature


def test_fit_drops_sentinel_labels():
    # fuzz regression: '', 'None', 'nan' must not become real classes
    import numpy as np
    rng = np.random.default_rng(0)
    rows = [{"clip_id": f"c{i}", "video": f"v{i}", "k": ("fault" if i % 2 else "")}
            for i in range(20)]
    embed = {r["clip_id"]: rng.standard_normal(512) for r in rows}
    _, report, _ = build._fit(rows, lambda r: r["k"], embed, min_class=2)
    assert report.get("degenerate") is True               # only 'fault' is real -> 1 class
    assert "" not in report.get("classes", [])


def test_fit_rejects_nonfinite_embeddings():
    # fuzz regression: NaN/Inf embeddings give a clean SystemExit, not a sklearn crash
    import numpy as np
    import pytest
    rows = [{"clip_id": f"c{i}", "video": f"v{i}", "k": ("fault" if i % 2 else "normal")}
            for i in range(8)]
    embed = {r["clip_id"]: np.full(512, np.nan) for r in rows}
    with pytest.raises(SystemExit):
        build._fit(rows, lambda r: r["k"], embed, min_class=2)


def test_cv_report_no_usable_fold_is_none_not_nan():
    # fuzz regression: an un-splittable head reports None (valid JSON), not NaN
    import numpy as np
    X = np.random.default_rng(0).standard_normal((4, 512))
    y = np.array(["a", "a", "b", "b"])
    g = np.array(["v1", "v1", "v2", "v2"])
    rep = build._cv_report(X, y, g)
    assert rep["cv_folds"] == 0 and rep["cv_bal_acc"] is None and "cv_unreliable" in rep
