"""Rich, physics-informed DSP features for fault discrimination.

CLAP's global embedding blurs the cues that actually separate fault subtypes.
The machinery-fault literature (envelope analysis, bearing characteristic
frequencies, transient/percussive structure) says the discriminative signal is:
  - MODULATION RATE (rod knock ~crank rate, lifter tick ~cam rate = ~half;
    bearing faults at characteristic frequencies) -> Hilbert envelope spectrum
  - FREQUENCY BAND BALANCE (rod = low, lifter = higher) -> band-energy ratios
  - IMPULSIVENESS (knocks/ticks are percussive transients) -> crest factor,
    envelope kurtosis, harmonic/percussive energy ratio (HPSS)
  - TIMBRE -> MFCC means

This caches a ~38-d vector per clip; concat with CLAP in the sub-type experiments
(z-scored). Pre-emphasis is applied first (boosts the high-frequency content the
user flagged, where tick/lifter energy lives).

    uv run training/features/dsp_features2.py     (resumable)
"""
import json
import time

import librosa
import numpy as np
from scipy.signal import hilbert

from cardiag import paths

DATA = paths.TRAIN_DATA
CACHE = DATA / "dsp_features2.npz"
SR = 22050
MAX_S = 6.0
MANIFESTS = ["train", "val", "test", "external_eval", "anchors_fewshot"]


def needed_ids():
    """Only clips the sub-type experiments touch: sub-labeled + verified eval.
    (Computing rich DSP for all 10k is wasteful; HPSS-free keeps this fast.)"""
    ids = set()
    p = DATA / "corpus_textmined_labels.jsonl"
    if p.exists():
        ids |= {json.loads(l)["eid"] for l in open(p)}
    for l in open(DATA / "external_eval.jsonl"):
        ids.add(json.loads(l)["id"])
    return ids


def envelope_spectrum(y, sr):
    """Dominant amplitude-modulation rate + harmonic richness (bearing/knock)."""
    env = np.abs(hilbert(y))
    env = env - env.mean()
    # modulation band 2-200 Hz
    n = len(env)
    sp = np.abs(np.fft.rfft(env * np.hanning(n)))
    f = np.fft.rfftfreq(n, 1 / sr)
    band = (f >= 2) & (f <= 200)
    if not band.any() or sp[band].max() < 1e-9:
        return [0.0, 0.0, 0.0, 0.0]
    sb, fb = sp[band], f[band]
    peak = int(np.argmax(sb))
    f0 = float(fb[peak])
    strength = float(sb[peak] / (sb.mean() + 1e-9))
    # energy at 2x f0 (harmonic) relative to f0 -> impulsive train richness
    h2 = float(np.interp(2 * f0, fb, sb) / (sb[peak] + 1e-9))
    # spectral flatness of envelope spectrum (tonal modulation vs noisy)
    flat = float(np.exp(np.mean(np.log(sb + 1e-9))) / (sb.mean() + 1e-9))
    return [f0, strength, h2, flat]


def band_ratios(y, sr):
    S = np.abs(librosa.stft(y, n_fft=1024)) ** 2
    f = librosa.fft_frequencies(sr=sr, n_fft=1024)
    edges = [0, 250, 800, 2000, 5000, sr / 2]
    tot = S.sum() + 1e-9
    return [float(S[(f >= a) & (f < b)].sum() / tot)
            for a, b in zip(edges[:-1], edges[1:])]


def feats(path):
    try:
        dur = librosa.get_duration(path=str(path))
        off = max(0.0, (dur - MAX_S) / 2)
        y, _ = librosa.load(str(path), sr=SR, mono=True, offset=off,
                            duration=MAX_S)
    except Exception:
        return None
    if len(y) < SR // 2:
        return None
    y = librosa.effects.preemphasis(y, coef=0.97)  # boost highs (tick/lifter)
    y = y / (np.abs(y).max() + 1e-9)

    out = []
    out += envelope_spectrum(y, SR)                       # 4
    out += band_ratios(y, SR)                             # 5
    # impulsiveness
    env = np.abs(hilbert(y))
    rms = float(np.sqrt(np.mean(y ** 2)))
    out += [float(np.max(np.abs(y)) / (rms + 1e-9)),       # crest factor
            float(((env - env.mean()) ** 4).mean()
                  / ((env.var() + 1e-9) ** 2)),            # env kurtosis
            float(np.mean(librosa.feature.zero_crossing_rate(y)))]  # zcr  -> 3
    # percussiveness proxy (cheap, HPSS-free): onset-flux energy fraction
    flux = librosa.onset.onset_strength(y=y, sr=SR)
    out += [float(flux.mean() / (np.abs(y).mean() * SR + 1e-9) * 1e3),
            float(flux.std() / (flux.mean() + 1e-9))]          # 2
    # spectral shape
    out += [float(np.mean(librosa.feature.spectral_centroid(y=y, sr=SR))),
            float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=SR))),
            float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=SR))),
            float(np.mean(librosa.feature.spectral_flatness(y=y)))]  # 4
    out += [float(x) for x in
            np.mean(librosa.feature.spectral_contrast(y=y, sr=SR), axis=1)]  # 7
    # timbre
    out += [float(x) for x in
            np.mean(librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13), axis=1)]  # 13
    return np.array(out, dtype=np.float32)                # total 37


def main():
    want = needed_ids()
    todo, seen = [], set()
    for name in MANIFESTS:
        for line in open(DATA / f"{name}.jsonl"):
            r = json.loads(line)
            if r["id"] in want and r["id"] not in seen:
                seen.add(r["id"])
                todo.append((r["id"], r["path"]))
    print(f"need {len(want)} ids; {len(todo)} resolvable to clip paths")
    ids, X = [], []
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        ids, X = list(z["ids"]), list(z["X"])
        have = set(ids)
        todo = [t for t in todo if t[0] not in have]
        print(f"resuming: {len(have)} cached, {len(todo)} to go")
    t0, fail = time.time(), 0
    for k, (cid, path) in enumerate(todo):
        v = feats(paths.resolve_clip(path))
        if v is None:
            fail += 1
            continue
        ids.append(cid)
        X.append(v)
        if k % 500 == 0:
            np.savez(CACHE, ids=np.array(ids), X=np.array(X))
            print(f"  {len(ids)} done ({(k+1)/max(1e-9,time.time()-t0):.1f}/s,"
                  f" {fail} fail, dim {len(v)})", flush=True)
    np.savez(CACHE, ids=np.array(ids), X=np.array(X))
    print(f"done: {len(ids)} rows dim {len(X[0]) if X else 0} "
          f"({fail} fail, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
