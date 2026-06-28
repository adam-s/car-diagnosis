"""The train/serve embedding contract (cardiag.audio.embed).

These lock in the anti-skew invariant: every vector a head ever sees — at train
or at serve — is a single-span embed_clip() vector, and multi-span recordings are
pooled in PROBABILITY space, never by averaging embeddings. All offline (no CLAP).
"""
from __future__ import annotations

import numpy as np

from cardiag.audio import embed
from cardiag.inference.classifier import _proba


# ---- probability pooling (not embedding averaging) -------------------------
class _FakeHead:
    classes_ = np.array(["fault", "normal"])

    def predict_proba(self, X):
        # one distinct row per span; the contract must AVERAGE these
        return np.array([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]])[: len(X)]


def test_proba_pools_across_spans_in_probability_space():
    X = np.zeros((3, 512))                     # 3 spans
    out = _proba(_FakeHead(), X)
    # mean of the three fault columns = (0.9+0.1+0.5)/3 = 0.5
    assert abs(out["fault"] - 0.5) < 1e-9
    assert abs(out["normal"] - 0.5) < 1e-9
    assert abs(sum(out.values()) - 1.0) < 1e-9


def test_proba_single_span_is_that_span():
    X = np.zeros((1, 512))
    out = _proba(_FakeHead(), X)
    assert out["fault"] == 0.9 and out["normal"] == 0.1


# ---- model_vectors: one row per isolated span, each via embed_clip ---------
class _Res:
    """Stand-in for CleanResult with two isolated spans."""
    isolated = [np.zeros(48_000, np.float32), np.zeros(48_000, np.float32)]
    segments = ["s0", "s1"]
    sr = 48_000


def test_model_vectors_is_one_row_per_isolated_span(monkeypatch):
    monkeypatch.setattr(embed, "clean", lambda p: _Res())
    # embed_clips is the only CLAP call — stub it to a recognizable matrix
    monkeypatch.setattr(embed, "embed_clips",
                        lambda ys, sr=48_000: np.arange(len(ys) * 4).reshape(len(ys), 4))
    ev = embed.model_vectors("clip.wav", clean_audio=True)
    assert ev.source == "isolated"
    assert ev.vectors.shape == (2, 4)          # 2 spans -> 2 rows, NOT 1 averaged
    assert ev.segments == ["s0", "s1"]
    assert ev.n == 2


def test_model_vectors_falls_back_to_windows_when_nothing_isolated(monkeypatch):
    class _Empty:
        isolated = []
        segments = []
        sr = 48_000
    monkeypatch.setattr(embed, "clean", lambda p: _Empty())
    monkeypatch.setattr(embed, "_window_vectors",
                        lambda path, win_s=10.0: np.ones((3, 4)))
    ev = embed.model_vectors("clip.wav", clean_audio=True)
    assert ev.source == "windows"
    assert ev.vectors.shape == (3, 4)          # still one row per window
    assert ev.clean_result is not None         # kept for the "isolated nothing" note
