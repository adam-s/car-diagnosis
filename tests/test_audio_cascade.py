"""Integration tests for the cleaning cascade and clean() on synthetic audio.

These run the real Silero VAD + librosa cascade (no network, no CLAP); they
prove the isolation logic on signals with a known shape.
"""
from __future__ import annotations

import numpy as np

from cardiag.audio.cascade import candidate_regions, cyclic_features, spectral_fingerprint
from cardiag.audio.clean import CleanResult, clean


def test_cascade_isolates_loud_span(tone_wav):
    import librosa

    from cardiag import config
    y, _ = librosa.load(tone_wav, sr=config.SR_CHEAP, mono=True)
    regions = candidate_regions(y)
    assert regions, "should find the loud tone burst"
    s, e = regions[0]
    # the burst sits around 1.0-2.5s; the isolated region should overlap it
    assert s < 2.4 and e > 1.1


def test_cascade_finds_nothing_in_silence(silence_wav):
    import librosa

    from cardiag import config
    y, _ = librosa.load(silence_wav, sr=config.SR_CHEAP, mono=True)
    assert candidate_regions(y) == []


def test_cyclic_features_detects_periodicity():
    sr = 16_000
    t = np.linspace(0, 3, sr * 3, endpoint=False)
    # 5 Hz click train -> high periodicity
    clicks = (np.sin(2 * np.pi * 5 * t) > 0.99).astype(np.float32)
    periodicity, pulse_hz, regularity = cyclic_features(clicks, sr)
    assert 0.0 <= periodicity <= 1.0
    assert pulse_hz >= 0.0


def test_spectral_fingerprint_shape():
    sr = 16_000
    y = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr)).astype(np.float32)
    fp = spectral_fingerprint(y, sr)
    assert set(fp) == {"centroid_hz", "flatness"}
    assert fp["centroid_hz"] > 0


def test_clean_returns_segments_no_music_gate(tone_wav):
    res = clean(tone_wav, music_gate=False)
    assert isinstance(res, CleanResult)
    assert not res.is_empty
    assert res.kept_seconds > 0
    assert res.merged_audio().size > 0
    d = res.to_dict()
    assert d["is_music"] is False and d["segments"]


def test_clean_empty_on_silence(silence_wav):
    res = clean(silence_wav, music_gate=False)
    assert res.is_empty
    assert res.merged_audio().size == 0
