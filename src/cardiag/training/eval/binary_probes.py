"""Process-of-elimination probes: which yes/no questions can the audio answer?

A mechanic doesn't need the exact part; narrowing the search helps. So instead
of one N-way pick (which dilutes confidence), ask each acoustic super-group as
an independent binary with calibrated P: can we confidently INCLUDE it ("this
IS an engine-internal sound") or RULE IT OUT ("this is NOT a brake/squeal
sound")? Either direction narrows the diagnosis.

For each group reports, on the curated set, creator-grouped OOF:
  INCLUDE cov@p85 = fraction of all clips we can confidently call positive @85% prec
  EXCLUDE cov@n95 = fraction we can confidently rule out (P low, true-positive <=5%)
Calibration (ECE) gates honesty.

    uv run training/eval/binary_probes.py
"""
import glob
import json

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cardiag import paths

DATA = paths.TRAIN_DATA
CORPORA = [paths.YT_DATA / "corpus.enriched.tiered.jsonl",
           paths.YT_DATA / "corpus.jsonl",
           paths.TT_DATA / "corpus_labeled.tiered.jsonl",
           paths.REDDIT_DATA / "corpus.jsonl"]

GROUPS = {
    "engine-internal (knock/tick)": {"engine_internal", "valvetrain", "low_oil"},
    "running-gear (wheel/susp/drive)": {"driveline", "suspension"},
    "squeal/friction (brake/belt)": {"brakes", "accessory_belt"},
    "fluid/hiss (exhaust/cooling)": {"exhaust", "cooling"},
    "steering": {"steering"},
}


def channel_map():
    m = {}
    for p in glob.glob(str(paths.YT_DATA / "meta" / "*" / "*.info.json")):
        try:
            d = json.load(open(p))
            m[d["id"]] = d.get("channel_id") or d.get("uploader_id") or d["id"]
        except Exception:
            pass
    return m


def quality_map():
    q = {}
    for p in CORPORA:
        if not p.exists():
            continue
        for l in open(p):
            r = json.loads(l)
            if r.get("clip_id"):
                q[r["clip_id"]] = (r.get("mech_confirm"),
                                   float(r.get("end", 0)) - float(r.get("start", 0)))
    return q


def load():
    chan, qmap = channel_map(), quality_map()
    z = np.load(DATA / "clap_embeddings.npz", allow_pickle=True)
    emb = {i: v / (np.linalg.norm(v) + 1e-9) for i, v in zip(z["ids"], z["X"])}
    X, cat, cre = [], [], []
    for l in open(DATA / "corpus_textmined_labels.jsonl"):
        o = json.loads(l)
        hi = o.get("tm_explicit") and (o.get("tm_trust") == "timestamp"
                                       or o.get("platform") == "tiktok")
        if not hi or o["eid"] not in emb or o.get("tm_category") in (None, "other"):
            continue
        mech, dur = qmap.get(o["eid"].split(":", 1)[1], (None, 0))
        if (mech or 0) < 0.5 or dur > 12:
            continue
        X.append(emb[o["eid"]])
        cat.append(o["tm_category"])
        cre.append("yt:" + chan.get(o["video"], o["video"])
                   if o["platform"] == "youtube"
                   else o["platform"][:2] + ":" + o["video"])
    return np.array(X), np.array(cat), np.array(cre)


def oof_binary(X, y, cre):
    P = np.zeros(len(y))
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    for tr, te in skf.split(X, y, cre):
        clf = CalibratedClassifierCV(
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
            method="isotonic", cv=3)
        clf.fit(X[tr], y[tr])
        P[te] = clf.predict_proba(X[te])[:, list(clf.classes_).index(1)]
    return P


def ece(y, p, nb=10):
    e = 0.0
    for a in np.linspace(0, 1, nb + 1)[:-1]:
        b = a + 1 / nb
        m = (p >= a) & (p < (b if b < 1 else 1.01))
        if m.sum():
            e += m.sum() / len(p) * abs(y[m].mean() - p[m].mean())
    return e


def main():
    X, cat, cre = load()
    print(f"curated set: {len(cat)} clips | {len(set(cre))} creators\n")
    print(f"{'super-group':34}{'base':>6}{'ECE':>6}"
          f"{'  INCLUDE @85%prec':>18}{'  RULE-OUT @95%':>16}")
    for name, cats in GROUPS.items():
        y = np.array([1 if c in cats else 0 for c in cat])
        base = y.mean()
        if y.sum() < 30:
            print(f"{name:34}{base:6.2f}   (too few positives)")
            continue
        p = oof_binary(X, y, cre)
        # INCLUDE: high-P prefix where precision(positive)>=0.85
        order = np.argsort(-p)
        ys = y[order]
        inc = next((k / len(ys) for k in range(len(ys), 0, -1)
                    if ys[:k].mean() >= 0.85), 0.0)
        # RULE-OUT: low-P prefix where positives among them <=5% (95% truly negative)
        order2 = np.argsort(p)
        yl = y[order2]
        exc = next((k / len(yl) for k in range(len(yl), 0, -1)
                    if yl[:k].mean() <= 0.05), 0.0)
        print(f"{name:34}{base:6.2f}{ece(y,p):6.3f}"
              f"{inc:14.0%}    {exc:12.0%}")
    print("\nINCLUDE = % of ALL clips we can confidently call positive (>=85% right)")
    print("RULE-OUT = % of ALL clips we can confidently exclude (>=95% truly not it)")


if __name__ == "__main__":
    main()
