"""Is the 90% real? Re-measure region triage under PROPER splits.

The sub-label experiments grouped by video; two videos from the same creator
(same mic/garage/car) can leak across folds. This re-measures the curated set
under stricter grouping:
  - by VIDEO    (what we reported)
  - by CREATOR  (YT channel / TT video / reddit post): the honest one
and with two protocols:
  - grouped 5-fold CV (mean ± fold std)
  - a FIXED held-out test: 75% of creator-groups train, 25% held out (touched
    once), calibrated confidence, precision@coverage on the held-out clips.

    uv run training/eval/split_check.py
"""
import glob
import hashlib
import json
from collections import Counter

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from cardiag import paths

DATA = paths.TRAIN_DATA
ENGINE = {"engine_internal", "valvetrain", "low_oil"}
CHASSIS = {"suspension", "driveline", "steering"}


def channel_map():
    m = {}
    for p in glob.glob(str(paths.YT_DATA / "meta" / "*" / "*.info.json")):
        try:
            d = json.load(open(p))
            m[d["id"]] = (d.get("channel_id") or d.get("uploader_id")
                          or d["id"])
        except Exception:
            pass
    return m


def cov_at(correct, conf, t):
    c = correct[np.argsort(-conf)]
    for k in range(len(c), 0, -1):
        if c[:k].mean() >= t:
            return k / len(c)
    return 0.0


def main():
    chan = channel_map()
    z = np.load(DATA / "clap_embeddings.npz", allow_pickle=True)
    emb = {i: v / (np.linalg.norm(v) + 1e-9) for i, v in zip(z["ids"], z["X"])}
    reg = lambda c: ("engine" if c in ENGINE
                     else "chassis" if c in CHASSIS else None)

    X, y, vid, cre = [], [], [], []
    for l in open(DATA / "corpus_textmined_clean.jsonl"):
        o = json.loads(l)
        r = reg(o["tm_category"])
        if not r or o["eid"] not in emb:
            continue
        X.append(emb[o["eid"]])
        y.append(r)
        vid.append(o["video"])
        if o["platform"] == "youtube":
            cre.append("yt:" + chan.get(o["video"], o["video"]))
        else:
            cre.append(o["platform"][:2] + ":" + o["video"])
    X, y = np.array(X), np.array(y)
    vid, cre = np.array(vid), np.array(cre)
    print(f"curated region set: {len(y)} clips | {len(set(vid))} videos | "
          f"{len(set(cre))} creators | balance {dict(Counter(y))}\n")

    def fit(Xtr, ytr):
        clf = CalibratedClassifierCV(
            LogisticRegression(max_iter=3000, class_weight="balanced"),
            method="isotonic", cv=3)
        try:
            return clf.fit(Xtr, ytr)
        except Exception:
            return LogisticRegression(max_iter=3000,
                                      class_weight="balanced").fit(Xtr, ytr)

    # --- grouped 5-fold CV, by video vs by creator ---
    for gname, groups in [("video", vid), ("creator", cre)]:
        ns = min(5, len(set(groups)), min(Counter(y).values()))
        skf = StratifiedGroupKFold(n_splits=ns, shuffle=True, random_state=0)
        accs, covs = [], []
        for tr, te in skf.split(X, y, groups):
            clf = fit(X[tr], y[tr])
            p = clf.predict_proba(X[te])
            pred = np.array(clf.classes_)[np.argmax(p, 1)]
            corr = (pred == y[te])
            accs.append(corr.mean())
            covs.append(cov_at(corr, p.max(1), 0.90))
        print(f"CV by {gname:8}: acc {np.mean(accs):.3f}±{np.std(accs):.3f} | "
              f"cov@p90 {np.mean(covs):.3f}±{np.std(covs):.3f}")

    # --- FIXED held-out test by creator (touched once) ---
    test_c = {c for c in set(cre)
              if int(hashlib.md5(c.encode()).hexdigest(), 16) % 4 == 0}
    tr = np.array([i for i in range(len(y)) if cre[i] not in test_c])
    te = np.array([i for i in range(len(y)) if cre[i] in test_c])
    clf = fit(X[tr], y[tr])
    p = clf.predict_proba(X[te])
    corr = (np.array(clf.classes_)[np.argmax(p, 1)] == y[te])
    print(f"\nFIXED held-out (creator 75/25): train {len(tr)} clips / "
          f"{len(set(cre[tr]))} creators, test {len(te)} clips / "
          f"{len(test_c)} creators")
    print(f"  held-out acc {corr.mean():.3f} | cov@p90 "
          f"{cov_at(corr, p.max(1), 0.90):.3f} | cov@p95 "
          f"{cov_at(corr, p.max(1), 0.95):.3f} | "
          f"majority {max(Counter(y[te]).values())/len(te):.3f}")


if __name__ == "__main__":
    main()
