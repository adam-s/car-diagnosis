"""CLAP-embed NEW scrape clips (from data/youtube/corpus.jsonl) not yet in the
cache, and append to clap_embeddings.npz. So the overnight-scraped clips become
trainable without re-embedding everything.

    uv run training/features/embed_new.py
"""
import json
import time

import librosa
import numpy as np
import torch

from cardiag import paths

DATA = paths.TRAIN_DATA
CACHE = DATA / "clap_embeddings.npz"
CORPUS = paths.YT_DATA / "corpus.jsonl"
SR, MAX_S, BATCH = 48000, 10.0, 24


def load_clip(path):
    try:
        dur = librosa.get_duration(path=str(path))
        off = max(0.0, (dur - MAX_S) / 2)
        y, _ = librosa.load(str(path), sr=SR, mono=True, offset=off,
                            duration=MAX_S)
        return y if len(y) >= SR // 2 else None
    except Exception:
        return None


def main():
    if not CACHE.exists():
        print("no base cache; run embed.py first")
        return
    z = np.load(CACHE, allow_pickle=True)
    ids, X = list(z["ids"]), list(z["X"])
    have = set(ids)

    todo = []
    seen = set()
    for l in open(CORPUS):
        r = json.loads(l)
        if not r.get("file"):
            continue
        eid = "yo:" + r["clip_id"]
        if eid in have or eid in seen:
            continue
        # corpus.jsonl 'file' is youtube-relative ("data/clips/..")
        p = paths.resolve_clip(f"youtube/{r['file']}")
        if p.exists():
            seen.add(eid)
            todo.append((eid, p))
    print(f"{len(todo)} new clips to embed (cache has {len(have)})", flush=True)
    if not todo:
        return

    from transformers import ClapModel, ClapProcessor
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(dev).eval()
    p = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

    t0, fail = time.time(), 0
    for i in range(0, len(todo), BATCH):
        bids, clips = [], []
        for eid, path in todo[i:i + BATCH]:
            y = load_clip(path)
            if y is None:
                fail += 1
                continue
            bids.append(eid)
            clips.append(y)
        if not clips:
            continue
        inp = p(audio=clips, sampling_rate=SR, return_tensors="pt",
                padding=True).to(dev)
        with torch.no_grad():
            out = m.get_audio_features(**inp)
            e = (out if torch.is_tensor(out)
                 else getattr(out, "audio_embeds", out.pooler_output))
            e = e.cpu().numpy()
        e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
        ids.extend(bids)
        X.extend(e)
        if (i // BATCH) % 20 == 0:
            np.savez(CACHE, ids=np.array(ids), X=np.array(X))
            print(f"  {len(ids)} total ({fail} fail)", flush=True)
    np.savez(CACHE, ids=np.array(ids), X=np.array(X))
    print(f"done: cache now {len(ids)} ({fail} fail, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
