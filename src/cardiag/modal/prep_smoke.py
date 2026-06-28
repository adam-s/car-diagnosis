"""Sample a balanced curated subset and stage 16k PCM16 wavs for the
Qwen2-Audio second-opinion smoke test (modal/modal_audio.py).

Picks up to 25 engine + 25 chassis clips (capped per video for diversity),
resamples to 16 kHz mono PCM16 (what Qwen2-Audio expects), and writes
data/training/smoke/<eid>.wav + manifest.jsonl {id, label, part, wav}.

    uv run modal/prep_smoke.py
"""
import json
import random
from collections import Counter

import librosa
import numpy as np
import soundfile as sf

from cardiag import paths

DATA = paths.TRAIN_DATA
OUT = DATA / "smoke"
ENGINE = {"engine_internal", "valvetrain", "low_oil"}
CHASSIS = {"suspension", "driveline", "steering"}

CORPORA = [("yo", "youtube", paths.YT_DATA / "corpus.enriched.tiered.jsonl"),
           ("yo", "youtube", paths.YT_DATA / "corpus.jsonl"),
           ("ti", "tiktok", paths.TT_DATA / "corpus_labeled.tiered.jsonl"),
           ("rd", "reddit", paths.REDDIT_DATA / "corpus.jsonl")]


def path_map():
    m = {}
    for pre, plat, p in CORPORA:
        if not p.exists():
            continue
        for l in open(p):
            r = json.loads(l)
            if r.get("clip_id") and r.get("file"):
                m.setdefault(f"{pre}:{r['clip_id']}",
                             paths.resolve_clip(f"{plat}/{r['file']}"))
    return m


def main():
    random.seed(7)
    pm = path_map()
    rows = [json.loads(l)
            for l in open(DATA / "corpus_textmined_clean.jsonl")]
    random.shuffle(rows)
    OUT.mkdir(exist_ok=True)
    picked, per_video, per_class = [], Counter(), Counter()
    for o in rows:
        reg = ("engine" if o["tm_category"] in ENGINE else
               "chassis" if o["tm_category"] in CHASSIS else None)
        if not reg or per_class[reg] >= 25 or per_video[o["video"]] >= 2:
            continue
        p = pm.get(o["eid"])
        if not p or not p.exists():
            continue
        try:
            y, _ = librosa.load(str(p), sr=16000, mono=True, duration=10.0)
        except Exception:
            continue
        if len(y) < 8000:
            continue
        wav = OUT / (o["eid"].replace(":", "_") + ".wav")
        sf.write(wav, (np.clip(y, -1, 1) * 32767).astype(np.int16), 16000)
        picked.append({"id": o["eid"], "label": reg, "part": o["tm_part"],
                       "wav": str(wav)})
        per_video[o["video"]] += 1
        per_class[reg] += 1
    with open(OUT / "manifest.jsonl", "w") as fh:
        for r in picked:
            fh.write(json.dumps(r) + "\n")
    print(f"staged {len(picked)} clips {dict(per_class)} -> {OUT}")


if __name__ == "__main__":
    main()
