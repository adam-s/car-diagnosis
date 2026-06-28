"""Shared fixtures. Deterministic synthetic audio + a tiny fixture model so the
unit/contract/integration tests never need CLAP weights or the network."""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

SR = 48_000


def _tone_burst(freq=220.0, sr=SR, total_s=4.0, on=(1.0, 2.5)):
    """A sine burst surrounded by near-silence — a clean cascade target."""
    t = np.linspace(0, total_s, int(sr * total_s), endpoint=False)
    y = 0.001 * np.random.default_rng(0).standard_normal(len(t))  # quiet floor
    s, e = int(on[0] * sr), int(on[1] * sr)
    y[s:e] += 0.4 * np.sin(2 * np.pi * freq * t[s:e])
    return y.astype(np.float32)


@pytest.fixture
def tone_wav(tmp_path):
    """Path to a synthetic WAV with one loud non-speech span."""
    p = tmp_path / "tone.wav"
    sf.write(p, _tone_burst(), SR)
    return str(p)


@pytest.fixture
def silence_wav(tmp_path):
    """Path to a near-silent WAV (cascade should isolate nothing)."""
    p = tmp_path / "silence.wav"
    sf.write(p, (1e-4 * np.random.default_rng(1).standard_normal(SR * 3)).astype(np.float32), SR)
    return str(p)


@pytest.fixture
def fixture_model(tmp_path):
    """A joblib artifact with the SAME shape as the real model (kind/knock/cause
    heads + triage), trained on random embeddings. Lets us exercise the full
    Classifier/Triage wiring without CLAP."""
    import joblib
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(42)
    X = rng.standard_normal((60, 512))

    def head(labels):
        y = np.array([labels[i % len(labels)] for i in range(len(X))])
        return LogisticRegression(max_iter=200).fit(X, y)

    heads = {
        "kind": head(["fault", "normal"]),
        "knock": head(["knock", "normal_idle"]),
        "cause": head(["brakes", "belt", "power_steering", "accessories",
                       "fuel_ignition", "low_oil"]),
    }
    clap_path = tmp_path / "best_model_clap.joblib"
    joblib.dump({"heads": heads, "emb": "clap"}, clap_path)

    triage = LogisticRegression(max_iter=200).fit(
        X, np.array([["chassis", "engine"][i % 2] for i in range(len(X))]))
    triage_path = tmp_path / "triage_model.joblib"
    joblib.dump({"model": triage, "classes": ["chassis", "engine"]}, triage_path)
    return {"clap": str(clap_path), "triage": str(triage_path)}


@pytest.fixture
def fixed_embedding(monkeypatch):
    """Patch the shared embedding contract so diagnose()/triage() are
    deterministic and CLAP-free. Returns one span vector through model_vectors()
    (the single seam train and serve share). Returns the vector used."""
    from cardiag.audio.embed import EmbedResult

    vec = np.random.default_rng(7).standard_normal(512)
    vec = vec / np.linalg.norm(vec)

    def _fake(path, **k):
        return EmbedResult(vectors=vec[None, :], segments=[], clean_result=None,
                           source="windows")

    monkeypatch.setattr("cardiag.inference.classifier.model_vectors", _fake)
    monkeypatch.setattr("cardiag.inference.triage.model_vectors", _fake)
    return vec
