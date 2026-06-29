"""Embed every manifest clip with frozen CLAP audio embeddings (512-d).

First-model baseline: CLAP is already local and its embeddings are the
zero-shot labeler's feature space: the trained-head delta over zero-shot
prompts is the first number Phase 2 needs. BEATs embeddings are the planned
upgrade (literature: BEATs >> CLAP for trained heads) and will be a second
embed script writing the same cache format.

Usage (from repo root, ~15-30 min on MPS, resumable):
    uv run training/features/embed.py

Writes data/training/clap_embeddings.npz  {ids: (N,), X: (N, 512)}
"""
import json
import time

import librosa
import numpy as np
import torch

from cardiag import paths

OUT = paths.TRAIN_DATA
CACHE = OUT / "clap_embeddings.npz"

MANIFESTS = ["train", "val", "test", "external_eval", "anchors_fewshot"]
SR = 48000
MAX_S = 10.0
BATCH = 24


def load_clip(path):
    try:
        dur = librosa.get_duration(path=str(path))
        off = max(0.0, (dur - MAX_S) / 2)
        y, _ = librosa.load(str(path), sr=SR, mono=True,
                            offset=off, duration=MAX_S)
        return y if len(y) >= SR // 2 else None
    except Exception:
        return None


def main():
    from transformers import ClapModel, ClapProcessor
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(dev).eval()
    proc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

    # collect unique (id, path) over all manifests
    todo, seen = [], set()
    for name in MANIFESTS:
        for line in open(OUT / f"{name}.jsonl"):
            r = json.loads(line)
            if r["id"] not in seen:
                seen.add(r["id"])
                todo.append((r["id"], r["path"]))

    # resume from an existing cache
    done_ids, done_X = [], []
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        done_ids, done_X = list(z["ids"]), list(z["X"])
        have = set(done_ids)
        todo = [t for t in todo if t[0] not in have]
        print(f"resuming: {len(have)} cached, {len(todo)} to embed")

    t0, n_fail = time.time(), 0
    for i in range(0, len(todo), BATCH):
        ids, clips = [], []
        for cid, path in todo[i:i + BATCH]:
            y = load_clip(paths.resolve_clip(path))
            if y is None:
                n_fail += 1
                continue
            ids.append(cid)
            clips.append(y)
        if not clips:
            continue
        inp = proc(audio=clips, sampling_rate=SR,
                   return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            out = model.get_audio_features(**inp)
            # transformers version drift: tensor vs BaseModelOutputWithPooling
            e = (out if torch.is_tensor(out)
                 else getattr(out, "audio_embeds", out.pooler_output))
            e = e.cpu().numpy()
        e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
        done_ids.extend(ids)
        done_X.extend(e)
        if (i // BATCH) % 20 == 0:
            np.savez(CACHE, ids=np.array(done_ids), X=np.array(done_X))
            rate = (i + BATCH) / max(1e-9, time.time() - t0)
            print(f"  {len(done_ids)} embedded "
                  f"({rate:.1f} clips/s, {n_fail} failed)", flush=True)

    np.savez(CACHE, ids=np.array(done_ids), X=np.array(done_X))
    print(f"done: {len(done_ids)} embeddings -> {CACHE} "
          f"({n_fail} failed, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
