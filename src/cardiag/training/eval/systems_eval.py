"""Per-SYSTEM accuracy with confidence: the honest multi-class picture.

The region work collapsed everything to engine-vs-chassis. This measures the
actual systems a user cares about (brakes, steering, suspension, driveline/
transmission, belt, exhaust, ...) as a multi-class task, with the same rigor:
high-trust labels only, CLAP+scaler+logistic(balanced)+isotonic, creator-grouped
5-fold OOF, per-class precision/recall + how confident we can be per system.

    uv run training/eval/systems_eval.py
"""
import glob
import json
from collections import Counter

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cardiag import paths

DATA = paths.TRAIN_DATA
MIN_CLIPS = 40            # drop systems too small to measure honestly


def channel_map():
    m = {}
    for p in glob.glob(str(paths.YT_DATA / "meta" / "*" / "*.info.json")):
        try:
            d = json.load(open(p))
            m[d["id"]] = d.get("channel_id") or d.get("uploader_id") or d["id"]
        except Exception:
            pass
    return m


def main():
    chan = channel_map()
    z = np.load(DATA / "clap_embeddings.npz", allow_pickle=True)
    emb = {i: v / (np.linalg.norm(v) + 1e-9) for i, v in zip(z["ids"], z["X"])}

    X, y, cre = [], [], []
    for l in open(DATA / "corpus_textmined_labels.jsonl"):
        o = json.loads(l)
        hi = o.get("tm_explicit") and (o.get("tm_trust") == "timestamp"
                                       or o.get("platform") == "tiktok")
        cat = o.get("tm_category")
        if not hi or cat in (None, "other") or o["eid"] not in emb:
            continue
        X.append(emb[o["eid"]])
        y.append(cat)
        cre.append("yt:" + chan.get(o["video"], o["video"])
                   if o["platform"] == "youtube"
                   else o["platform"][:2] + ":" + o["video"])
    X = np.array(X)
    y = np.array(y)
    cre = np.array(cre)

    keep = {c for c, n in Counter(y).items() if n >= MIN_CLIPS}
    m = np.array([t in keep for t in y])
    X, y, cre = X[m], y[m], cre[m]
    print(f"{len(y)} high-trust clips across {len(keep)} systems "
          f"(>= {MIN_CLIPS} clips each), {len(set(cre))} creators")
    for c, n in Counter(y).most_common():
        print(f"   {n:5d}  {c}")

    classes = np.array(sorted(set(y)))
    P = np.zeros((len(y), len(classes)))
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    for tr, te in skf.split(X, y, cre):
        clf = CalibratedClassifierCV(
            make_pipeline(StandardScaler(), LogisticRegression(
                max_iter=3000, class_weight="balanced")),
            method="isotonic", cv=3)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])
        idx = [list(clf.classes_).index(c) for c in classes]
        P[te] = p[:, idx]
    pred = classes[np.argmax(P, 1)]
    conf = P.max(1)
    corr = pred == y

    print(f"\noverall multi-class acc {corr.mean():.3f} "
          f"(majority {Counter(y).most_common(1)[0][1]/len(y):.3f})")
    print(f"\n{'system':16}{'n':>6}{'precision':>10}{'recall':>8}"
          f"{'  p@conf>=.80 (cov)':>20}")
    for c in classes:
        tp = int(((pred == c) & (y == c)).sum())
        pc = int((pred == c).sum())
        ac = int((y == c).sum())
        prec = tp / pc if pc else float("nan")
        rec = tp / ac if ac else float("nan")
        hi = (pred == c) & (conf >= 0.80)
        hp = (corr[hi].mean() if hi.sum() else float("nan"))
        hcov = hi.sum() / ac if ac else 0
        print(f"{c:16}{ac:6d}{prec:10.2f}{rec:8.2f}"
              f"{hp:11.2f} ({hcov:4.0%})")

    # confidence coverage overall
    order = np.argsort(-conf)
    cc = corr[order]
    for t in (0.80, 0.90):
        cov = next((k / len(cc) for k in range(len(cc), 0, -1)
                    if cc[:k].mean() >= t), 0.0)
        print(f"\noverall cov@p{int(t*100)}: {cov:.3f}  "
              f"(answer {cov:.0%} of clips at {int(t*100)}% precision)")


if __name__ == "__main__":
    main()
