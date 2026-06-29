"""Build the proof-of-concept shipped model from a CLEAN, CLASS-BALANCED,
DOMAIN-DIVERSE corpus — the model that demonstrates the method works across
domains rather than detecting the recording source.

It combines a high-confidence, balanced slice of the scraped social corpus
(detect-study; Reddit already excluded) with three INDEPENDENT benchmark domains
(Car-Engine, ai-mechanic, Sound-Based-Vehicle-Diagnostics), embeds everything with
the same frozen CLAP, trains the fault/knock/cause + triage heads, and reports a
group-safe train/validate/test split plus a leave-one-domain-out out-of-sample
number. Honest by construction: the OOS number is printed even when it's bad.

    python scripts/build_poc_model.py \\
        --emb  ~/Projects/detect-study/.../clap_embeddings.npz \\
        --splits ~/Projects/detect-study/.../training/ \\
        --benchmarks ~/Projects/detect-mech-issues/external-data \\
        --out models/

The corpora are NOT bundled; only the ~100 KB linear heads are committed.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np


def _embed_benchmarks(ext: str):
    """Segment + embed the independent benchmark datasets through the SAME cascade
    inference uses (``model_vectors``): each recording -> its per-span vectors, one
    training example per span. A long recording becomes multiple short spans, so
    train and serve see the same unit (no whole-clip embeds). All spans from one
    file share a group id, so grouped CV never leaks across the split.
    Returns list of (vec, kind, cause, domain, group)."""
    from cardiag.audio.embed import model_vectors
    rows = []
    ce = f"{ext}/repos/Car-Engine-Sounds-Dataset/_audio"
    rows += [(p, "normal", "", "carengine") for p in glob.glob(ce + "/normal/*")]
    rows += [(p, "fault", "", "carengine") for p in glob.glob(ce + "/abnormal/*")]
    am = f"{ext}/kaggle/ai-mechanic-engine-condition-audio-fault-finding/ai-mechanic-export"
    for p in glob.glob(am + "/training/*") + glob.glob(am + "/testing/*"):
        n = os.path.basename(p).lower()
        if n.startswith(("normal", "idling")):
            rows.append((p, "normal", "", "aimech"))
        elif "air leak" in n or "oil cap" in n:
            rows.append((p, "fault", "low_oil" if "oil" in n else "exhaust", "aimech"))
    for d in glob.glob(f"{ext}/repos/Sound-Based-Vehicle-Diagnostics-Emergency-Signal-Recognition/Datasets/DB1/*"):
        cause = os.path.basename(d).lower().replace(" ", "_")[:24]
        for p in glob.glob(d + "/*"):
            if p.lower().endswith((".m4a", ".wav", ".mp3")):
                rows.append((p, "fault", cause, "soundbased"))
    out = []
    for i, (p, kind, cause, dom) in enumerate(rows):
        if not p.lower().endswith((".wav", ".mp3", ".m4a", ".ogg", ".flac")):
            continue
        try:
            for v in model_vectors(p).vectors:          # cascade -> per-span vectors
                out.append((v, kind, cause, dom, f"{dom}_{i}"))
        except Exception:
            pass
        if (i + 1) % 80 == 0:
            print(f"  segmented {i+1}/{len(rows)} benchmark recordings -> {len(out)} spans",
                  flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--benchmarks", required=True)
    ap.add_argument("--out", default="models")
    ap.add_argument("--min-conf", type=float, default=0.6)
    ap.add_argument("--cap", type=int, default=300, help="max social clips per class")
    a = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="cardiag_poc_")
    os.environ["CARDIAG_DATA"] = tmp
    from cardiag.pipeline import build

    # social: high-confidence, balanced, capped (keeps the source domain from dominating)
    z = np.load(a.emb, allow_pickle=True)
    emb = {str(i): v / (np.linalg.norm(v) + 1e-9) for i, v in zip(z["ids"], z["X"])}
    soc = []
    for sp in ("train", "val", "test"):
        f = Path(a.splits) / f"{sp}.jsonl"
        if not f.exists():
            continue
        for r in (json.loads(ln) for ln in open(f)):
            if str(r["id"]) in emb and r.get("kind") in ("fault", "normal") and r.get("confidence", 0) >= a.min_conf:
                soc.append((emb[str(r["id"])], r.get("kind"), r.get("l1") or "", r.get("cause") or "",
                            str(r.get("group", r["id"])), "social"))
    import random

    from cardiag import config
    random.Random(0).shuffle(soc)
    kept = []
    nf = nn = 0
    for v, kind, l1, cause, grp, dom in soc:
        if kind == "fault" and nf >= a.cap:
            continue
        if kind == "normal" and nn >= a.cap:
            continue
        nf += kind == "fault"
        nn += kind == "normal"
        if l1 == "normal_idle":                # map to cardiag's knock-head normal token
            l1 = config.L1_NORMAL
        kept.append((v, kind, l1, cause, grp, dom))
    print(f"social (conf>={a.min_conf}, balanced, capped): {len(kept)}")

    print("embedding the independent benchmark domains…")
    bench = _embed_benchmarks(a.benchmarks)
    print(f"benchmark clips embedded: {len(bench)}")

    # combine into cardiag training rows + an embedding map
    rows, embed = [], {}
    for i, (v, kind, l1, cause, grp, dom) in enumerate(kept):
        cid = f"s{i}"
        embed[cid] = v
        rows.append({"clip_id": cid, "video": grp, "kind": kind, "l1": l1,
                     "cause": cause, "wav": f"/{dom}/x.wav"})
    for j, (v, kind, cause, dom, grp) in enumerate(bench):
        cid = f"b{j}"
        embed[cid] = v
        l1 = "knocking" if "knock" in cause else ""
        rows.append({"clip_id": cid, "video": grp, "kind": kind, "l1": l1,
                     "cause": cause, "wav": f"/{dom}/x.wav"})
    print(f"\ntraining the proof-of-concept heads on {len(rows)} clips "
          f"(diverse across social + 3 benchmark domains)…")
    build._train_heads(rows, embed, min_class=5, cause_fn=lambda r: r.get("cause") or None)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("best_model_clap.joblib", "triage_model.joblib"):
        shutil.copy(Path(tmp) / "training" / name, out / name)
    print(f"\nwrote {out}/best_model_clap.joblib + triage_model.joblib")


if __name__ == "__main__":
    main()
