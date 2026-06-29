"""Occlusion saliency: *why* did the model say that?

A rigorous, model-agnostic explanation (occlusion sensitivity, à la Zeiler &
Fergus 2014 / RISE) for the audio diagnosis: systematically mask a grid of
time × frequency tiles of the clip, re-embed each masked variant with CLAP, run
the calibrated fault/normal head, and measure how much the fault probability
moves. A tile whose removal collapses the verdict is the evidence the model relied
on. The result is a time-frequency heatmap, aligned to the spectrogram the UI
already shows, plus a one-sentence headline ("masking 1.2–1.8s, 200–800 Hz changes
fault probability by −0.42").

This is honest by construction: it perturbs the *real* model and reports the real
effect: no surrogate, no hand-wavy attention.
"""
from __future__ import annotations

import numpy as np

from cardiag import config

_N_FFT = 1024
_HOP = 256
MAX_S = 10.0                      # CLAP embeds ~10s; cap so the map stays faithful


def _p_fault_vec(clf, X) -> np.ndarray:
    """Calibrated P(fault) for each row of X (applies the head's temperature)."""
    head = clf.heads["kind"]
    classes = list(getattr(head, "classes_", []))
    if "fault" not in classes:
        return np.full(len(X), np.nan)
    fi = classes.index("fault")
    T = float(clf.temps.get("kind", 1.0)) if hasattr(clf, "temps") else 1.0
    if T != 1.0 and hasattr(head, "decision_function"):
        d = np.asarray(head.decision_function(X))           # binary: >0 favours classes_[1]
        p1 = 1.0 / (1.0 + np.exp(-d / T))                   # P(classes_[1])
        return p1 if fi == 1 else (1.0 - p1)
    return np.asarray(head.predict_proba(X))[:, fi]


def occlusion_saliency(path, model_path=None, n_time: int = 12, n_freq: int = 6) -> dict:
    """Return a time × frequency saliency map for the fault/normal verdict on
    ``path``. Each cell is how much *removing* that audio region moves the
    probability toward the verdict (positive = the model leaned on it)."""
    from pathlib import Path

    import librosa

    from cardiag.audio.embed import embed_clip, embed_clips
    from cardiag.inference.classifier import Classifier, _usable
    clf = Classifier.load(model_path)
    if not _usable(clf.heads["kind"]):
        return {"available": False,
                "reason": "the fault/normal head is degenerate — nothing to explain"}

    if not Path(path).exists():
        return {"available": False, "reason": "could not read audio"}
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, _ = librosa.load(str(path), sr=config.SR_CLAP, mono=True)
    except Exception:
        return {"available": False, "reason": "could not read audio"}
    y = np.nan_to_num(y).astype(np.float32)
    sr = config.SR_CLAP
    if len(y) > int(MAX_S * sr):
        y = y[: int(MAX_S * sr)]
    dur = len(y) / sr
    if dur < 0.4 or float(np.max(np.abs(y))) < 1e-3:
        return {"available": False, "reason": "clip too short or near-silent to explain"}

    base_p = float(_p_fault_vec(clf, embed_clip(y)[None, :])[0])
    if not np.isfinite(base_p):
        return {"available": False, "reason": "nothing to explain for this clip"}
    verdict = "fault" if base_p >= 0.5 else "normal"

    D = librosa.stft(y, n_fft=_N_FFT, hop_length=_HOP)        # (F_bins, T_frames)
    Fb, Tf = D.shape
    t_edges = np.linspace(0, Tf, n_time + 1).astype(int)
    f_edges = np.linspace(0, Fb, n_freq + 1).astype(int)

    variants, coords = [], []
    for ti in range(n_time):
        for fi in range(n_freq):
            Dm = D.copy()
            Dm[f_edges[fi]:f_edges[fi + 1], t_edges[ti]:t_edges[ti + 1]] = 0
            ym = librosa.istft(Dm, hop_length=_HOP, length=len(y)).astype(np.float32)
            variants.append(ym)
            coords.append((ti, fi))

    ps = _p_fault_vec(clf, embed_clips(variants))            # batched CLAP embed
    imp = np.zeros((n_time, n_freq))
    for (ti, fi), p in zip(coords, ps):
        # importance toward the SHOWN verdict: removing supporting evidence should
        # move the probability away from the verdict.
        imp[ti, fi] = (base_p - p) if verdict == "fault" else (p - base_p)

    times = [round(float(t_edges[i] / Tf * dur), 3) for i in range(n_time + 1)]
    freqs = [round(float(f_edges[i] / Fb * (sr / 2)), 1) for i in range(n_freq + 1)]

    # marginal importance over time and over frequency (sum amplifies the per-tile
    # signal; the verdict leans on regions where these are largest)
    time_imp = imp.sum(axis=1)
    freq_imp = imp.sum(axis=0)

    # headline: the single tile the verdict leans on most
    ti, fi = np.unravel_index(int(np.argmax(imp)), imp.shape)
    top = {"t0": times[ti], "t1": times[ti + 1], "f0": freqs[fi], "f1": freqs[fi + 1],
           "delta": round(float(imp[ti, fi]), 3),
           "p_without": round(float(ps[ti * n_freq + fi]), 3)}
    return {
        "available": True, "verdict": verdict, "base_p": round(base_p, 3),
        "duration": round(dur, 3), "n_time": n_time, "n_freq": n_freq,
        "time_edges": times, "freq_edges": freqs,
        "map": np.round(imp, 4).tolist(),
        "time_importance": np.round(time_imp, 4).tolist(),
        "freq_importance": np.round(freq_imp, 4).tolist(),
        "top": top,
    }
