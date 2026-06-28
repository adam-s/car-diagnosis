"""Contract tests for the inference layer. A fixture model + patched embedding
exercise the full Classifier/Triage wiring deterministically, without CLAP."""
from __future__ import annotations

import pytest

from cardiag import Classifier, TriageClassifier
from cardiag.types import Band, Diagnosis, TriageResult, Verdict


def test_classifier_load_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Classifier.load(tmp_path / "nope.joblib")


def test_diagnose_returns_well_formed_diagnosis(fixture_model, fixed_embedding, tone_wav):
    clf = Classifier.load(fixture_model["clap"])
    d = clf.diagnose(tone_wav, clean_audio=False)
    assert isinstance(d, Diagnosis)
    assert isinstance(d.verdict, Verdict)
    assert 0.0 <= d.fault_probability <= 1.0
    assert 0.0 <= d.engine_knock_probability <= 1.0
    assert 1 <= len(d.causes) <= 3
    assert all(0.0 <= c.p <= 1.0 for c in d.causes)
    # causes ranked descending
    ps = [c.p for c in d.causes]
    assert ps == sorted(ps, reverse=True)


def test_verdict_thresholds(fixture_model, fixed_embedding, tone_wav, monkeypatch):
    clf = Classifier.load(fixture_model["clap"])

    def force_fault(p):
        monkeypatch.setattr(clf.heads["kind"], "predict_proba",
                            lambda X: [[p, 1 - p]])  # classes_ = [fault, normal]
    force_fault(0.9)
    assert clf.diagnose(tone_wav, clean_audio=False).verdict is Verdict.FAULT
    force_fault(0.5)
    assert clf.diagnose(tone_wav, clean_audio=False).verdict is Verdict.UNCERTAIN
    force_fault(0.2)
    assert clf.diagnose(tone_wav, clean_audio=False).verdict is Verdict.NORMAL


def test_diagnosis_json_roundtrip(fixture_model, fixed_embedding, tone_wav):
    import json
    d = Classifier.load(fixture_model["clap"]).diagnose(tone_wav, clean_audio=False)
    payload = json.loads(json.dumps(d.to_dict()))
    assert payload["verdict"] in {"fault", "normal", "uncertain"}


def test_triage_returns_band(fixture_model, fixed_embedding, tone_wav):
    t = TriageClassifier.load(fixture_model["triage"]).triage(tone_wav)
    assert isinstance(t, TriageResult)
    assert isinstance(t.band, Band)
    assert t.triage in {"engine", "chassis"}
    assert abs(sum(t.probabilities.values()) - 1.0) < 1e-6
