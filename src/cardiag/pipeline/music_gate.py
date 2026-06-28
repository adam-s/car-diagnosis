"""Music-contamination gate. ~25% of TikTok clips are background music with no
mechanical content (measured). CLAP separates music cleanly (pure-music clips
score ~1.0). Score every on-disk clip and write a music_score map; the tiering
step then drops high-music clips (they can never be gold, and are unusable as
training audio).

    uv run pipeline/music_gate.py tiktok    # -> data/tiktok/music_scores.json
"""
import glob
import json
import sys
from pathlib import Path

import librosa
import torch
from transformers import ClapModel, ClapProcessor

from cardiag import paths

PROMPTS = ["music or a song with a beat",
           "a mechanical car noise or engine sound",
           "a person talking", "silence or ambient noise"]
MUSIC_THRESH = 0.5


def main(platform):
    base = {"youtube": paths.YT_DATA, "tiktok": paths.TT_DATA,
            "reddit": paths.REDDIT_DATA}[platform]
    files = sorted(glob.glob(str(base / "clips" / "*" / "*.wav")))
    print(f"{platform}: scoring {len(files)} clips for music")
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(dev).eval()
    p = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

    scores = {}
    for i in range(0, len(files), 16):
        chunk = files[i:i + 16]
        clips = [librosa.load(f, sr=48000, mono=True)[0] for f in chunk]
        inp = p(text=PROMPTS, audio=clips, sampling_rate=48000,
                return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            pr = m(**inp).logits_per_audio.softmax(-1).cpu().numpy()
        for f, row in zip(chunk, pr):
            # clip_id in the ledger is "<video>_NN"; rebuild from path
            vid = Path(f).parent.name
            nn = "".join(ch for ch in Path(f).stem if ch.isdigit())
            scores[f"{vid}_{int(nn):02d}"] = {
                "music": round(float(row[0]), 3),
                "mech": round(float(row[1]), 3),
                "file": f,
            }
        if (i // 16) % 25 == 0:
            print(f"  {min(i+16, len(files))}/{len(files)}", flush=True)

    (base / "music_scores.json").write_text(json.dumps(scores, indent=2))
    music = sum(1 for v in scores.values() if v["music"] >= MUSIC_THRESH)
    print(f"\n{len(scores)} clips scored -> {base}/music_scores.json")
    print(f"music (>= {MUSIC_THRESH}): {music} ({100*music/max(1,len(scores)):.0f}%) "
          f"-> excluded from corpus/gold")
    print(f"usable mechanical: {len(scores)-music}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tiktok")
