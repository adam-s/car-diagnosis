"""Embed manifest clips with AST (Audio Spectrogram Transformer, AudioSet).

The CLAP knock-blindness motivates a backbone whose pretraining target is
sound-event recognition (AudioSet has explicit engine/knock/rattle classes).
AST mean-pooled hidden states (768-d) are the candidate. Same npz format as
embed.py, so iterate.py consumes either interchangeably (or concatenated).

Usage (from repo root, resumable):
    uv run training/features/embed_ast.py
Writes data/training/ast_embeddings.npz
"""
import json
import time

import librosa
import numpy as np
import torch

from cardiag import paths

DATA = paths.TRAIN_DATA
CACHE = DATA / "ast_embeddings.npz"
MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
SR = 16000
MAX_S = 10.0
BATCH = 16
MANIFESTS = ["train", "val", "test", "external_eval", "anchors_fewshot"]


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
    from transformers import ASTModel, AutoFeatureExtractor
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    fe = AutoFeatureExtractor.from_pretrained(MODEL)
    model = ASTModel.from_pretrained(MODEL).to(dev).eval()

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
    for i in range(0, len(todo), BATCH):
        bids, clips = [], []
        for cid, path in todo[i:i + BATCH]:
            y = load_clip(paths.resolve_clip(path))
            if y is None:
                fail += 1
                continue
            bids.append(cid)
            clips.append(y)
        if not clips:
            continue
        inp = fe(clips, sampling_rate=SR, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model(**inp).last_hidden_state.mean(1)  # mean-pool tokens
        e = out.cpu().numpy()
        ids.extend(bids)
        X.extend(e)
        if (i // BATCH) % 20 == 0:
            np.savez(CACHE, ids=np.array(ids), X=np.array(X))
            print(f"  {len(ids)} embedded "
                  f"({(i+BATCH)/max(1e-9,time.time()-t0):.1f}/s, {fail} fail)",
                  flush=True)
    np.savez(CACHE, ids=np.array(ids), X=np.array(X))
    print(f"done: {len(ids)} AST embeddings ({fail} failed, "
          f"{time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
