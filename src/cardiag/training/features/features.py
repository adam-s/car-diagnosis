"""Cache cheap hand DSP features (cyclic + spectral) per manifest clip.

Motivation: CLAP embeddings are knock-blind (0% verified knock recall) because
mel/CLIP-style features wash out the low-rate impulsive structure that
distinguishes knock/tick from idle. The pipeline already computes cyclic
features (periodicity, pulse_hz, regularity) that target exactly this: cache
them for every clip so iterate.py can concat them onto any embedding.

Output (data/training/dsp_features.npz): ids, X = [periodicity, pulse_hz,
regularity, centroid_hz, flatness, zcr, rms]  (7 scalars).

Usage: uv run training/features/features.py   (resumable)
"""
import json
import time

import librosa
import numpy as np

from cardiag import paths

DATA = paths.TRAIN_DATA
CACHE = DATA / "dsp_features.npz"
SR = 22050
MAX_S = 10.0
MANIFESTS = ["train", "val", "test", "external_eval", "anchors_fewshot"]


# Inlined from ingest/youtube/audio.py (importing it pulls torch/silero/transformers).
def cyclic_features(y, sr):
    """Cyclic structure: faults that repeat at rotation rate (1-15 Hz)."""
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
    return {
        "centroid_hz": round(float(np.mean(
            librosa.feature.spectral_centroid(y=y, sr=sr))), 1),
        "flatness": round(float(np.mean(
            librosa.feature.spectral_flatness(y=y))), 4),
    }


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
    per, hz, reg = cyclic_features(y, SR)
    sp = spectral_fingerprint(y, SR)
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    rms = float(np.mean(librosa.feature.rms(y=y)))
    return np.array([per, hz, reg, sp["centroid_hz"], sp["flatness"],
                     zcr, rms], dtype=np.float32)


def main():
    todo, seen = [], set()
    for name in MANIFESTS:
        for line in open(DATA / f"{name}.jsonl"):
            r = json.loads(line)
            if r["id"] not in seen:
                seen.add(r["id"])
                todo.append((r["id"], r["path"]))
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
                  f" {fail} failed)", flush=True)
    np.savez(CACHE, ids=np.array(ids), X=np.array(X))
    print(f"done: {len(ids)} feature rows ({fail} failed, "
          f"{time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
