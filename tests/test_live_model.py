"""Live test: real model + real CLAP on a real clip.

Excluded from the default run (needs the ~2GB CLAP weights and a trained model
in data/training/). Run explicitly:

    pytest -m live
"""
from __future__ import annotations

from pathlib import Path

import pytest

from cardiag import Classifier, Verdict, clean

pytestmark = [pytest.mark.live, pytest.mark.model]

_REPO = Path(__file__).resolve().parent.parent
_PROOF_CLIPS = sorted((_REPO / "proofs").glob("*.wav"))


@pytest.mark.skipif(not (_REPO / "data/training/best_model_clap.joblib").exists(),
                    reason="no trained model in data/training/")
@pytest.mark.skipif(not _PROOF_CLIPS, reason="no proof clips in proofs/")
def test_real_diagnose_on_real_clip():
    clf = Classifier.load()
    d = clf.diagnose(str(_PROOF_CLIPS[0]))
    assert isinstance(d.verdict, Verdict)
    assert 0.0 <= d.fault_probability <= 1.0
    assert d.causes


@pytest.mark.skipif(not _PROOF_CLIPS, reason="no proof clips in proofs/")
def test_real_clean_isolates_audio():
    res = clean(str(_PROOF_CLIPS[0]))
    # real clip -> cleaning produces a structured result (segments may be empty
    # if the whole clip is uniform, but the contract must hold)
    assert res.total_seconds > 0
    assert 0.0 <= res.music_probability <= 1.0
