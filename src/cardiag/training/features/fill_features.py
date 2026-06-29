"""Fill AST + DSP2 feature caches for curated clips the old manifests missed.

embed_ast.py / dsp_features2.py resolve clip paths via the training manifests,
which predate the scaled scrape, so newer curated clips (most of the TikTok
sub-labeled set) have CLAP but no AST/DSP2. The confidence ablations need the
full intersection. This resolves paths straight from the platform corpora and
appends the missing rows to both caches.

    uv run training/features/fill_features.py
"""
import json
import time

import librosa
import numpy as np
import torch

from cardiag import paths

DATA = paths.TRAIN_DATA
from cardiag.training.features.dsp_features2 import feats as dsp_feats  # noqa: E402

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
    want = {json.loads(l)["eid"]
            for l in open(DATA / "corpus_textmined_clean.jsonl")}
    pm = path_map()
    for cache, kind in [(DATA / "ast_embeddings.npz", "ast"),
                        (DATA / "dsp_features2.npz", "dsp2")]:
        z = np.load(cache, allow_pickle=True)
        ids, X = list(z["ids"]), list(z["X"])
        todo = [(e, pm[e]) for e in sorted(want - set(ids))
                if e in pm and pm[e].exists()]
        print(f"{kind}: {len(todo)} missing curated clips", flush=True)
        if not todo:
            continue
        t0, fail = time.time(), 0
        if kind == "dsp2":
            for e, p in todo:
                v = dsp_feats(p)
                if v is None:
                    fail += 1
                    continue
                ids.append(e)
                X.append(v)
        else:
            from transformers import ASTModel, AutoFeatureExtractor
            dev = "mps" if torch.backends.mps.is_available() else "cpu"
            fe = AutoFeatureExtractor.from_pretrained(
                "MIT/ast-finetuned-audioset-10-10-0.4593")
            model = ASTModel.from_pretrained(
                "MIT/ast-finetuned-audioset-10-10-0.4593").to(dev).eval()
            for i in range(0, len(todo), 16):
                bids, clips = [], []
                for e, p in todo[i:i + 16]:
                    try:
                        y, _ = librosa.load(str(p), sr=16000, mono=True,
                                            duration=10.0)
                    except Exception:
                        y = None
                    if y is None or len(y) < 8000:
                        fail += 1
                        continue
                    bids.append(e)
                    clips.append(y)
                if not clips:
                    continue
                inp = fe(clips, sampling_rate=16000,
                         return_tensors="pt").to(dev)
                with torch.no_grad():
                    out = model(**inp).last_hidden_state.mean(1)
                ids.extend(bids)
                X.extend(out.cpu().numpy())
        np.savez(cache, ids=np.array(ids), X=np.array(X))
        print(f"  {kind}: +{len(ids) - len(z['ids'])} rows "
              f"({fail} fail, {time.time()-t0:.0f}s) -> {len(ids)} total",
              flush=True)


if __name__ == "__main__":
    main()
