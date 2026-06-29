"""The cheap cleaning cascade: CPU, ~free, 80-200x realtime.

It discards silence, speech, and broadband static so the expensive CLAP tier
only ever sees the few surviving percent of audio. Measured bake-off: webrtcvad
was disqualified (it calls mechanical noise "speech"); Silero VAD works (speech
0.87-0.98 coverage, mechanical regions <=0.43).

The same cascade runs at corpus-build time and at inference (see
:func:`cardiag.audio.clean.clean`), which is what keeps inference matched to
training.
"""
from __future__ import annotations

import librosa
import numpy as np
import torch

from cardiag import config

_STEP = 0.030
_SILERO = None


def _vad():
    global _SILERO
    if _SILERO is None:
        from silero_vad import load_silero_vad
        _SILERO = load_silero_vad()
    return _SILERO


def candidate_regions(y16, sr: int = config.SR_CHEAP, return_speech_frac: bool = False):
    """Cheap tiers 0-2: loud, non-speech, non-static spans, as ``(start, end)``
    seconds. With ``return_speech_frac`` also returns the clip-level speech
    fraction, a shop-context prior (talky repair video vs clean compilation)."""
    from silero_vad import get_speech_timestamps
    speech = get_speech_timestamps(torch.from_numpy(y16), _vad(),
                                   sampling_rate=sr, return_seconds=True)
    hop = int(_STEP * sr)
    rms = librosa.feature.rms(y=y16, frame_length=hop * 2, hop_length=hop)[0]
    loud = rms > max(0.005, float(np.percentile(rms, 20)) * 1.5)
    sp = np.zeros(len(loud), dtype=bool)
    for t in speech:
        sp[int(t["start"] / _STEP):int(t["end"] / _STEP) + 1] = True
    keep = loud & ~sp[:len(loud)]

    regions, cur = [], None
    for i, k in enumerate(keep):
        t = i * _STEP
        if k:
            cur = [t, t + _STEP] if cur is None else [cur[0], t + _STEP]
        elif cur:
            regions.append(cur)
            cur = None
    if cur:
        regions.append(cur)
    merged = []
    for s, e in regions:
        if merged and s - merged[-1][1] < 0.5:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    out = []
    for s, e in merged:
        if e - s < config.MIN_REGION_S:
            continue
        seg = y16[int(s * sr):int(e * sr)]
        if float(np.mean(librosa.feature.spectral_flatness(y=seg))) > \
                config.SPECTRAL_FLATNESS_MAX:
            continue
        cov = sum(max(0, min(e, t["end"]) - max(s, t["start"]))
                  for t in speech) / (e - s)
        if cov > config.MAX_SPEECH_COV:
            continue
        out.append((round(s, 2), round(e, 2)))
    if return_speech_frac:
        total = len(y16) / sr
        sp_frac = sum(t["end"] - t["start"] for t in speech) / max(1e-9, total)
        return out, round(sp_frac, 3)
    return out


def cyclic_features(y, sr):
    """Cyclic structure: many faults repeat at rotation rate (thump/click
    ~1-15 Hz). Returns ``(periodicity 0-1, pulse_hz, regularity 0-1)``."""
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)
    if env.std() < 1e-6 or len(env) < 16:
        return 0.0, 0.0, 0.0
    env = (env - env.mean()) / (env.std() + 1e-9)
    fps = sr / 256.0
    ac = np.correlate(env, env, "full")[len(env) - 1:]
    ac = ac / (ac[0] + 1e-9)
    lo, hi = int(fps / 15), min(int(fps / 0.8), len(ac) - 1)
    if hi <= lo:
        return 0.0, 0.0, 0.0
    lag = lo + int(np.argmax(ac[lo:hi]))
    periodicity, pulse_hz = float(ac[lag]), fps / lag
    peaks = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=5,
                                   post_avg=5, delta=0.5, wait=int(fps * 0.04))
    if len(peaks) >= 3:
        ioi = np.diff(peaks) / fps
        regularity = float(max(0.0, 1.0 - ioi.std() / (ioi.mean() + 1e-9)))
    else:
        regularity = 0.0
    return periodicity, pulse_hz, regularity


def spectral_fingerprint(y, sr):
    """Cheap FFT character for validation cross-checks."""
    return {
        "centroid_hz": round(float(np.mean(
            librosa.feature.spectral_centroid(y=y, sr=sr))), 1),
        "flatness": round(float(np.mean(
            librosa.feature.spectral_flatness(y=y))), 4),
    }
