"""Build a shippable cardiag model from a corpus of CLAP embeddings + labeled
splits, and write it to a directory (default ``models/``).

This produces the optional, pre-trained model that ``cardiag serve --model`` can
point at. It is deliberately decoupled from the default flow: a fresh clone still
trains from its *own* scrape (``cardiag train``); this script lets the author bake
a stronger model from a larger corpus (e.g. the parent detect-study corpus of
~20k clips, validated against a human-verified eval set).

    python scripts/build_shipped_model.py \\
        --emb  ~/Projects/detect-study/sound-diagnostics/data/training/clap_embeddings.npz \\
        --splits ~/Projects/detect-study/sound-diagnostics/data/training/ \\
        --out models/

The corpus is NOT bundled; only the small linear-head artifacts are. Labels are
mapped into cardiag's vocabulary (``normal_idle`` -> the knock head's normal token).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emb", required=True, help="clap_embeddings.npz with arrays ids, X")
    ap.add_argument("--splits", required=True, help="dir with train/val/test.jsonl")
    ap.add_argument("--out", default="models", help="output dir for the model (default models/)")
    ap.add_argument("--min-class", type=int, default=10)
    ap.add_argument("--prune-noisy", type=float, default=0.10,
                    help="confident-learning label cleaning (the weak labels are "
                         "noisy; cleaning improves verified-set accuracy). Default 0.10.")
    a = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="cardiag_ship_")
    os.environ["CARDIAG_DATA"] = tmp                 # _train_heads writes under here
    from cardiag import config
    from cardiag.pipeline import build

    z = np.load(a.emb, allow_pickle=True)
    emb = {str(i): v / (np.linalg.norm(v) + 1e-9) for i, v in zip(z["ids"], z["X"])}
    rows = []
    n_missing_group = 0
    for sp in ("train", "val", "test"):
        f = Path(a.splits) / f"{sp}.jsonl"
        if not f.exists():
            continue
        for r in (json.loads(ln) for ln in open(f)):
            if str(r["id"]) not in emb:
                continue
            if not r.get("group"):                 # guard the by-video CV: without a
                n_missing_group += 1               # group field, every clip is its own
                continue                           # group and the grouped CV leaks
            l1 = r.get("l1") or ""
            if l1 == "normal_idle":                  # map to cardiag's knock-head normal token
                l1 = config.L1_NORMAL
            rows.append({"clip_id": str(r["id"]), "video": str(r.get("group", r["id"])),
                         "kind": r.get("kind"), "l1": l1, "cause": r.get("cause") or "",
                         "wav": f"/{r.get('source', 'x')}/clip.wav"})
    if not rows:
        raise SystemExit("no rows with a 'group' field — grouped CV would leak; add "
                         "per-video groups to the splits before building a shipped model.")
    if n_missing_group:
        print(f"  skipped {n_missing_group} rows missing a 'group' field (would leak CV)")
    print(f"training on {len(rows)} labeled clips ({len(emb)} embeddings), "
          f"cleaning {int(a.prune_noisy*100)}% noisiest labels per source…")
    build._train_heads(rows, emb, a.min_class, lambda r: r.get("cause") or None,
                       prune_noisy=a.prune_noisy)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("best_model_clap.joblib", "triage_model.joblib"):
        shutil.copy(Path(tmp) / "training" / name, out / name)
    print(f"\nwrote {out}/best_model_clap.joblib + {out}/triage_model.joblib")
    print(f"serve it with:  cardiag serve --model {out}")


if __name__ == "__main__":
    main()
