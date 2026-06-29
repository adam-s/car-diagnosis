"""Robustness: the public API must handle hostile/degenerate audio gracefully:
a clean result or a clear error, never an obscure crash. Offline (no CLAP)."""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from cardiag import clean
from cardiag.audio.clean import CleanResult

SR = 48_000


def _write(tmp_path, name, y, sr=SR):
    p = tmp_path / name
    sf.write(p, y, sr)
    return str(p)


def test_clean_silence(tmp_path):
    p = _write(tmp_path, "s.wav", np.zeros(SR * 2, dtype=np.float32))
    res = clean(p, music_gate=False)
    assert isinstance(res, CleanResult) and res.is_empty


def test_clean_subsecond(tmp_path):
    y = np.random.default_rng(0).standard_normal(2000).astype(np.float32)
    p = _write(tmp_path, "tiny.wav", y)
    res = clean(p, music_gate=False)              # 0.04s, must not crash
    assert isinstance(res, CleanResult)


def test_clean_stereo_is_handled(tmp_path):
    y = (0.3 * np.random.default_rng(1).standard_normal((SR, 2))).astype(np.float32)
    p = _write(tmp_path, "stereo.wav", y)
    res = clean(p, music_gate=False)               # 2-channel -> downmixed
    assert isinstance(res, CleanResult)


def test_clean_extreme_clipping(tmp_path):
    # a heavily clipping / extreme-amplitude clip must not crash the cascade
    y = np.clip(np.random.default_rng(3).standard_normal(SR) * 50, -1e6, 1e6).astype(np.float32)
    p = _write(tmp_path, "loud.wav", y / np.abs(y).max())
    assert isinstance(clean(p, music_gate=False), CleanResult)


def test_clean_nonexistent_file_clear_error():
    with pytest.raises((FileNotFoundError, ValueError, RuntimeError)):
        clean("/no/such/file.wav", music_gate=False)


def test_clean_non_audio_file_clear_error(tmp_path):
    bad = tmp_path / "notaudio.wav"
    bad.write_text("this is not audio")
    with pytest.raises((ValueError, RuntimeError)):
        clean(str(bad), music_gate=False)


def test_clean_empty_file_clear_error(tmp_path):
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        clean(str(empty), music_gate=False)
