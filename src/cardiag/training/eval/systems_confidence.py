"""Honest CONFIDENCE across all car systems/parts: the right question.

systems_eval measured recall (can it find every brake clip? no). The product
bar is different: when the model says "80% brakes", is it right 80%, and on how
many clips can it be that sure? A model that abstains on the hard 80% but is
reliably calibrated on the confident 20% is acceptable.

Two fixes vs systems_eval:
  - NATURAL priors (drop class_weight=balanced): balancing distorts
    probabilities to buy minority recall and wrecks calibration; we want honest
    confidence, not recall on everything.
  - measure precision@confidence + reliability (claimed->empirical), per class,
    not recall.

Levels: SYSTEM (engine/brakes/steering/suspension/driveline/...) and PART
(wheel_bearing/brakes/belt/valvetrain/rod_knock/...). Creator-grouped OOF.
Compares calibration methods, picks best by ECE, saves the artifact.

    uv run training/eval/systems_confidence.py
"""
import glob
import json
from collections import Counter

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cardiag import paths

DATA = paths.TRAIN_DATA
from cardiag.training.prep.causes import canonical_cause  # noqa: E402

MIN_CLIPS = 40


def channel_map():
    m = {}
    for p in glob.glob(str(paths.YT_DATA / "meta" / "*" / "*.info.json")):
        try:
            d = json.load(open(p))
            m[d["id"]] = d.get("channel_id") or d.get("uploader_id") or d["id"]
        except Exception:
            pass
    return m


def load(level):
    """level: 'system' (tm_category) or 'part' (canonical_cause(tm_part))."""
    chan = channel_map()
    z = np.load(DATA / "clap_embeddings.npz", allow_pickle=True)
    emb = {i: v / (np.linalg.norm(v) + 1e-9) for i, v in zip(z["ids"], z["X"])}
    X, y, cre = [], [], []
    for l in open(DATA / "corpus_textmined_labels.jsonl"):
        o = json.loads(l)
        hi = o.get("tm_explicit") and (o.get("tm_trust") == "timestamp"
                                       or o.get("platform") == "tiktok")
        if not hi or o["eid"] not in emb:
            continue
        lab = (o.get("tm_category") if level == "system"
               else canonical_cause(o.get("tm_part")))
        if lab in (None, "other"):
            continue
        X.append(emb[o["eid"]])
        y.append(lab)
        cre.append("yt:" + chan.get(o["video"], o["video"])
                   if o["platform"] == "youtube"
                   else o["platform"][:2] + ":" + o["video"])
    X, y, cre = np.array(X), np.array(y), np.array(cre)
    keep = {c for c, n in Counter(y).items() if n >= MIN_CLIPS}
    m = np.array([t in keep for t in y])
    return X[m], y[m], cre[m]


def oof(X, y, cre, balanced, method):
    classes = np.array(sorted(set(y)))
    P = np.zeros((len(y), len(classes)))
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    for tr, te in skf.split(X, y, cre):
        lr = LogisticRegression(max_iter=3000,
                                class_weight="balanced" if balanced else None)
        base = make_pipeline(StandardScaler(), lr)
        clf = (CalibratedClassifierCV(base, method=method, cv=3)
               if method else base)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])
        cc = list(clf.classes_)
        P[te] = p[:, [cc.index(c) for c in classes]]
    return classes, P


def ece(corr, conf, nb=10):
    e = 0.0
    for a in np.linspace(0, 1, nb + 1)[:-1]:
        b = a + 1 / nb
        m = (conf >= a) & (conf < b if b < 1 else conf <= b)
        if m.sum():
            e += m.sum() / len(conf) * abs(corr[m].mean() - conf[m].mean())
    return e


def cov_at(corr, conf, t):
    c = corr[np.argsort(-conf)]
    return next((k / len(c) for k in range(len(c), 0, -1)
                 if c[:k].mean() >= t), 0.0)


def run(level):
    X, y, cre = load(level)
    print(f"\n{'='*64}\n{level.upper()} level: {len(y)} clips | "
          f"{len(set(y))} classes | {len(set(cre))} creators | "
          f"majority {Counter(y).most_common(1)[0][1]/len(y):.3f}")

    best = None
    for balanced, method in [(True, "isotonic"), (False, "isotonic"),
                             (False, "sigmoid"), (False, None)]:
        classes, P = oof(X, y, cre, balanced, method)
        pred = classes[np.argmax(P, 1)]
        corr, conf = pred == y, P.max(1)
        tag = f"{'bal ' if balanced else 'nat '}{method or 'raw':8}"
        print(f"  {tag}: acc {corr.mean():.3f} | ECE {ece(corr,conf):.3f} | "
              f"cov@p80 {cov_at(corr,conf,.80):.3f} | "
              f"cov@p70 {cov_at(corr,conf,.70):.3f} | "
              f"cov@p50 {cov_at(corr,conf,.50):.3f} | maxconf {conf.max():.2f}")
        if not balanced and (best is None or ece(corr, conf) < best[0]):
            best = (ece(corr, conf), tag, classes, P)

    _, tag, classes, P = best
    pred, conf = classes[np.argmax(P, 1)], P.max(1)
    corr = pred == y
    print(f"\n  BEST (natural prior): {tag}")
    print("  reliability  claimed -> empirical:")
    for a, b in [(.3, .4), (.4, .5), (.5, .6), (.6, .7), (.7, .8), (.8, 1.01)]:
        m = (conf >= a) & (conf < b)
        if m.sum():
            print(f"    {a:.1f}-{min(b,1):.1f}: n={m.sum():4d} "
                  f"empirical {corr[m].mean():.2f}")

    print("  parts CONFIDENTLY identifiable (precision on conf>=.50 subset):")
    for c in classes:
        m = (pred == c) & (conf >= 0.50)
        tot = int((y == c).sum())
        if m.sum() >= 5:
            print(f"    {c:16} prec {corr[m].mean():.2f} on "
                  f"{m.sum():3d} confident preds  (of {tot} true)")
    # final model on ALL data, saved
    method = "isotonic" if "isotonic" in tag else (
        "sigmoid" if "sigmoid" in tag else None)
    base = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=3000))
    final = (CalibratedClassifierCV(base, method=method, cv=3)
             if method else base).fit(X, y)
    out = DATA / f"conf_model_{level}.joblib"
    joblib.dump({"model": final, "classes": list(classes), "level": level}, out)
    print(f"  saved -> {out.name}")


def main():
    for level in ("system", "part"):
        run(level)


if __name__ == "__main__":
    main()
