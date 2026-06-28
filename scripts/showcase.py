"""Curate a showcase across sources and situations, then run the real pipeline
on every clip — proof that clean() + diagnose() handle the hard cases:

  * isolated      : a sub-span isolated from a longer clip (kept < total)
  * music_skipped : a clip the music gate flags (TikTok music-only, etc.)
  * compilation   : more than one distinct mechanical span
  * whole_clip    : nothing distinct to isolate -> graceful whole-clip fallback

Samples real clips already scraped across YouTube / TikTok / Reddit, runs the
model on each (MPS), writes proofs/showcase.jsonl, and prints a situation table.

    python scripts/showcase.py [n_per_platform=80] [SOURCE_DATA_DIR]
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

from cardiag import Classifier, clean

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "proofs" / "showcase.jsonl"
DEFAULT_SRC = Path("/Users/adamsohn/Projects/detect-study/sound-diagnostics/data")


def sample(src: Path, n_per: int) -> list[tuple[str, Path]]:
    picks: list[tuple[str, Path]] = []
    for platform in ("youtube", "tiktok", "reddit"):
        clips_dir = src / platform / "clips"
        wavs = list(clips_dir.rglob("*.wav")) if clips_dir.exists() else []
        # reddit stores audio under audio/ in some layouts
        if not wavs and (src / platform / "audio").exists():
            wavs = list((src / platform / "audio").rglob("*.wav"))
        random.Random(0).shuffle(wavs)
        picks += [(platform, w) for w in wavs[:n_per]]
    return picks


def situation(diag, cln) -> str:
    if cln.is_music:
        return "music_skipped"
    if len(cln.segments) > 1:
        return "compilation"
    if cln.segments and cln.kept_seconds < cln.total_seconds - 0.5:
        return "isolated"
    if not cln.segments:
        return "whole_clip"
    return "clean_full"


def main(n_per: int = 80, src: Path = DEFAULT_SRC) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    clips = sample(src, n_per)
    if not clips:
        print(f"no clips found under {src}")
        return
    print(f"showcase: {len(clips)} clips across platforms; loading model + CLAP…",
          flush=True)
    clf = Classifier.load()

    sit_counts: Counter = Counter()
    plat_counts: Counter = Counter()
    verdict_counts: Counter = Counter()
    t0 = time.time()
    with open(OUT, "w") as fh:
        for i, (platform, wav) in enumerate(clips):
            try:
                cln = clean(str(wav))
                diag = clf.diagnose(str(wav))
            except Exception as e:
                fh.write(json.dumps({"platform": platform, "file": str(wav),
                                     "error": f"{type(e).__name__}: {e}"}) + "\n")
                continue
            sit = situation(diag, cln)
            sit_counts[sit] += 1
            plat_counts[platform] += 1
            verdict_counts[diag.verdict.value] += 1
            fh.write(json.dumps({
                "platform": platform,
                "file": str(wav.relative_to(src)),
                "situation": sit,
                "total_s": round(cln.total_seconds, 2),
                "kept_s": cln.kept_seconds,
                "n_segments": len(cln.segments),
                "speech_fraction": cln.speech_fraction,
                "music_probability": round(cln.music_probability, 3),
                "verdict": diag.verdict.value,
                "fault_probability": diag.fault_probability,
                "knock_probability": diag.engine_knock_probability,
                "top_cause": diag.causes[0].part if diag.causes else None,
            }) + "\n")
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(clips)}  ({(time.time()-t0)/(i+1):.2f}s/clip)",
                      flush=True)

    print(f"\n=== showcase complete: {sum(sit_counts.values())} clips "
          f"in {time.time()-t0:.0f}s -> {OUT} ===")
    print("\nby platform :", dict(plat_counts))
    print("by verdict  :", dict(verdict_counts))
    print("by SITUATION:")
    for k, v in sit_counts.most_common():
        print(f"   {k:14} {v}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    s = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_SRC
    main(n, s)
