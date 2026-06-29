"""Turn scraped Reddit audio into clean fault CLIPS: same cascade as YouTube.

Reddit gives one short phone-mic video per post (the fault sound) + a crowd
diagnosis (label_reddit.py). This runs the YouTube cascade (Silero VAD + energy
+ spectral) to isolate the non-speech mechanical spans, scores them with CLAP
(L1 sound-type + mechanical-confirm), and writes clip wavs + a ledger, so
Reddit clips look exactly like YouTube/TikTok clips for the unified set.

    python -m cardiag.ingest.reddit.pipeline
Writes data/reddit/clips/<post>/clip_NN.wav + data/reddit/corpus.jsonl
"""
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from cardiag import config, paths
from cardiag.audio.cascade import candidate_regions, cyclic_features, spectral_fingerprint
from cardiag.audio.clap import Clap

DATA = paths.REDDIT_DATA
AUDIO = DATA / "audio"
CLIPS = DATA / "clips"
LEDGER = DATA / "corpus.jsonl"


def process(fullname, clap):
    wav = AUDIO / f"{fullname}.wav"
    if not wav.exists():
        return []
    try:
        y16, _ = librosa.load(str(wav), sr=config.SR_CHEAP, mono=True)
    except Exception:
        return []
    regions = candidate_regions(y16)            # phone videos: usually no talk
    if not regions:                              # fall back to the whole clip
        dur = len(y16) / config.SR_CHEAP
        if dur < config.MIN_REGION_S:
            return []
        regions = [(0.0, round(dur, 2))]
    # window long regions into uniform ~5s clips (a 30s grind -> 6 clips), so
    # Reddit clips match the curated YT/TT clip size instead of being huge.
    CLIP_S = 5.0
    windows = []
    for s, e in regions:
        if e - s <= CLIP_S * 1.4:
            windows.append((s, e))
        else:
            t = s
            while t < e - config.MIN_REGION_S:
                windows.append((round(t, 2), round(min(t + CLIP_S, e), 2)))
                t += CLIP_S
    regions = windows
    y48, sr = librosa.load(str(wav), sr=config.SR_CLAP, mono=True)
    clips = [y48[int(s * sr):int(e * sr)] for s, e in regions]
    conf = clap.score(clips, config.CONFIRM_KEEP + config.CONFIRM_DROP)
    l1p = clap.score(clips, config.L1_PROMPTS)

    out_dir = CLIPS / fullname
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = []
    for i, ((s, e), cp, lp) in enumerate(zip(regions, conf, l1p)):
        mech = float(cp[:len(config.CONFIRM_KEEP)].sum())
        if mech < config.MECH_REJECT_BELOW:      # music/speech/static -> drop
            continue
        o = np.argsort(-lp)
        y22 = librosa.resample(clips[i], orig_sr=sr, target_sr=22050)
        per, hz, reg = cyclic_features(y22, 22050)
        f = out_dir / f"clip_{i:02d}.wav"
        sf.write(f, clips[i], sr)
        recs.append({
            "clip_id": f"{fullname}_{i:02d}", "video": fullname,
            "platform": "reddit", "start": s, "end": e,
            "l1": config.L1_PROMPTS[o[0]][2:],
            "l1_conf": round(float(lp[o[0]]), 3),
            "l1_margin": round(float(lp[o[0]] - lp[o[1]]), 3),
            "mech_confirm": round(mech, 3),
            "cyclic": {"periodicity": round(per, 3), "pulse_hz": round(hz, 2),
                       "regularity": round(reg, 3)},
            "spectral": spectral_fingerprint(y22, 22050),
            "file": str(Path("data") / f.relative_to(paths.REDDIT_DATA))})
    if not recs:
        try:
            out_dir.rmdir()
        except OSError:
            pass
    return recs


def main():
    posts = [json.loads(l) for l in open(DATA / "posts.jsonl")]
    done = set()
    if LEDGER.exists():
        done = {json.loads(l)["video"] for l in open(LEDGER)}
    todo = [p for p in posts if p["fullname"] not in done
            and (AUDIO / f"{p['fullname']}.wav").exists()]
    print(f"{len(todo)} reddit posts -> clips ({len(done)} done)", flush=True)
    clap = Clap()
    n_clip = 0
    with open(LEDGER, "a") as led:
        for i, p in enumerate(todo, 1):
            for r in process(p["fullname"], clap):
                led.write(json.dumps(r) + "\n")
                n_clip += 1
            led.flush()
            if i % 25 == 0:
                print(f"  {i}/{len(todo)} posts, {n_clip} clips", flush=True)
    print(f"done: {n_clip} reddit clips -> {LEDGER}")


if __name__ == "__main__":
    main()
