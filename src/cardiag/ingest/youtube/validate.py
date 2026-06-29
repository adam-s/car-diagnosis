"""QA suite for the corpus ledger: three independent checks, cheap-first:

  A. FFT physics (free): does the L1 label agree with spectral character?
  B. Cluster coherence (free): does each auto clip's nearest CLAP neighbor
     share its L1 label?
  C. Haiku plausibility (cheap): is (video title, L1 sound) mechanically
     plausible? Catches topic/label mismatches like "normal idle" clips inside
     "WHAT ROD BEARING KNOCK SOUNDS LIKE".

Fable is NOT used here: it audits the final gold sample once, by hand.

    uv run validate.py
"""
import json
import subprocess
from collections import Counter

import librosa
import numpy as np

from cardiag import config, paths
from cardiag.audio.clap import Clap

# expected spectral character per L1 label
EXPECT = {
    "grinding noise": "broadband", "hissing noise": "broadband",
    "rattling noise": "broadband", "squealing or squeaking noise": "tonal",
    "high-pitched whining noise": "tonal", "ticking or clicking noise": "impulsive",
    "knocking or clunking noise": "impulsive", "humming or droning roar": "tonal",
    "normal smooth engine idle": "lowfreq",
}


def load_auto():
    recs = [json.loads(l) for l in open(paths.YT_DATA / "corpus.jsonl")]
    return [r for r in recs if r["status"] == "auto" and r["l1"] != "shop_tool"
            and r.get("file") and paths.resolve_clip(r["file"]).exists()]


def check_fft(recs):
    ok, flags = 0, []
    for r in recs:
        flat, cent = r["spectral"]["flatness"], r["spectral"]["centroid_hz"]
        exp = EXPECT.get(r["l1"], "")
        agree = ((exp == "broadband" and flat > 0.05)
                 or (exp in ("tonal", "impulsive") and flat < 0.10)
                 or (exp == "lowfreq" and cent < 2000))
        if agree:
            ok += 1
        else:
            flags.append((r["clip_id"], r["l1"], flat, cent))
    return ok, flags


def check_clusters(recs, clap):
    clips = [librosa.load(str(paths.resolve_clip(r["file"])), sr=config.SR_CLAP,
                          mono=True)[0] for r in recs]
    embs = clap.embed(clips)
    labels = [r["l1"] for r in recs]
    sim = embs @ embs.T
    np.fill_diagonal(sim, -1)
    nn = sim.argmax(1)
    agree = sum(labels[i] == labels[nn[i]] for i in range(len(recs)))
    return agree, Counter(labels)


def check_haiku(recs, batch=25):
    items = [{"i": i, "title": (r["provenance"].get("title") or "")[:90], "sound": r["l1"]}
             for i, r in enumerate(recs) if r["provenance"].get("title")]
    verdicts = {}
    for k in range(0, len(items), batch):
        prompt = (
            "Each item: a car-noise video's title and the sound-type assigned to an "
            "audio clip from it. Is the sound-type mechanically PLAUSIBLE for the "
            "topic (true unless clearly contradictory)? Reply ONLY a JSON array of "
            '{"i":int,"plausible":bool}.\n' + json.dumps(items[k:k + batch]))
        out = subprocess.run(["claude", "-p", "--model", config.HAIKU_MODEL, prompt],
                             capture_output=True, text=True, timeout=120).stdout
        try:
            for o in json.loads(out[out.index("["):out.rindex("]") + 1]):
                verdicts[o["i"]] = o["plausible"]
        except (ValueError, json.JSONDecodeError):
            pass
    flags = [(recs[i]["clip_id"], recs[i]["l1"],
              (recs[i]["provenance"].get("title") or "")[:60])
             for i, v in verdicts.items() if not v]
    return sum(verdicts.values()), len(verdicts), flags


def main():
    recs = load_auto()
    print(f"auto fault clips under test: {len(recs)}\n")

    ok, flags = check_fft(recs)
    print(f"A. FFT physics:      {ok}/{len(recs)} agree ({100*ok/max(1,len(recs)):.0f}%)")
    for f in flags[:6]:
        print(f"     FLAG {f}")

    clap = Clap()
    agree, counts = check_clusters(recs, clap)
    print(f"B. cluster coherence: {agree}/{len(recs)} NN-label agreement "
          f"({100*agree/max(1,len(recs)):.0f}%)")
    print(f"     label counts: {dict(counts)}")

    plaus, n, flags = check_haiku(recs)
    print(f"C. Haiku plausibility: {plaus}/{n} ({100*plaus/max(1,n):.0f}%)")
    for f in flags[:10]:
        print(f"     IMPLAUSIBLE {f}")


if __name__ == "__main__":
    main()
