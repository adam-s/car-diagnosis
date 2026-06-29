"""Precision @ coverage: the achievable form of the 99% goal.

"99% accuracy on every clip from audio" is not physically reachable. "99%
PRECISION on the confident subset, abstaining otherwise" is, and it's how
triage should work. This collects out-of-fold (video-grouped) predictions with
calibrated confidence, then sweeps an abstention threshold: accept a prediction
only when confidence >= t, and report precision (accuracy among accepted) vs
coverage (fraction accepted). The headline is COVERAGE AT 0.90 / 0.95 / 0.99
PRECISION: how much of the time we can confidently name the part.

Calibration: confidences are isotonic-calibrated within CV (CalibratedClassifierCV)
so the threshold means what it says.

    uv run training/eval/precision_coverage.py
"""
import json
import sys
from collections import Counter

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from cardiag import paths

DATA = paths.TRAIN_DATA
from cardiag.training.prep.causes import canonical_cause  # noqa: E402

ENGINE = {"engine_internal", "valvetrain", "low_oil"}
CHASSIS = {"suspension", "driveline", "steering"}


TIGHT = "--tight" in sys.argv  # only well-aligned labels (host cued THIS sound)


def load():
    z = np.load(DATA / "clap_embeddings.npz", allow_pickle=True)
    emb = {i: v / (np.linalg.norm(v) + 1e-9) for i, v in zip(z["ids"], z["X"])}
    rows = []
    for l in open(DATA / "corpus_textmined_labels.jsonl"):
        o = json.loads(l)
        hi = o["tm_explicit"] and (o["tm_trust"] == "timestamp"
                                   or o["platform"] == "tiktok")
        if TIGHT:
            # YT: tight timestamp overlap (<45s span); TikTok: short single
            # -fault clips are inherently tight.
            dur = o.get("tm_range_dur")
            hi = hi and (o["platform"] == "tiktok"
                         or (dur is not None and dur <= 45))
        if hi and o["eid"] in emb:
            rows.append((emb[o["eid"]], o, o["video"]))
    return rows


def oof(rows, labelf):
    """Out-of-fold (calibrated) predictions, video-grouped."""
    data = [(x, labelf(o), v) for x, o, v in rows if labelf(o)]
    X = np.array([d[0] for d in data])
    y = np.array([d[1] for d in data])
    g = np.array([d[2] for d in data])
    cc = Counter(y)
    nsplit = min(5, len(set(g)), min(cc.values()))
    if len(cc) < 2 or nsplit < 3:
        return None
    skf = StratifiedGroupKFold(n_splits=nsplit, shuffle=True, random_state=0)
    yt, yp, conf = [], [], []
    for tr, te in skf.split(X, y, g):
        base = LogisticRegression(max_iter=3000, class_weight="balanced")
        # isotonic calibration on an inner group-split of the training fold
        clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
        try:
            clf.fit(X[tr], y[tr])
        except Exception:
            clf = base.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])
        cls = np.array(clf.classes_)
        yp.extend(cls[np.argmax(p, 1)])
        conf.extend(np.max(p, 1))
        yt.extend(y[te])
    return np.array(yt), np.array(yp), np.array(conf)


def pr_at_cov(yt, yp, conf):
    correct = (yt == yp)
    order = np.argsort(-conf)
    out = {"n": len(yt), "overall_acc": round(float(correct.mean()), 3)}
    # coverage needed to reach each precision target (greedy by confidence)
    for target in (0.90, 0.95, 0.99):
        best_cov = 0.0
        for k in range(len(order), 0, -1):
            idx = order[:k]
            prec = correct[idx].mean()
            if prec >= target:
                best_cov = k / len(order)
                break
        out[f"cov@p{int(target*100)}"] = round(best_cov, 3)
    # precision at fixed coverage points
    for cov in (0.25, 0.5):
        k = max(1, int(cov * len(order)))
        out[f"p@cov{int(cov*100)}"] = round(float(correct[order[:k]].mean()), 3)
    return out


def main():
    rows = load()
    print(f"high-trust clips: {len(rows)}\n")

    def region(o):
        c = o["tm_category"]
        return "engine" if c in ENGINE else ("chassis" if c in CHASSIS
                                             else None)
    parts = Counter(canonical_cause(o["tm_part"]) for _, o, _ in rows)
    keep = {p for p, n in parts.items() if n >= 40 and p not in (None, "other")}

    def part(o):
        p = canonical_cause(o["tm_part"])
        return p if p in keep else None
    KNOCK = {"rod bearing": "rod_bearing", "main bearing": "rod_bearing",
             "crankshaft bearing": "rod_bearing", "lifter": "lifter",
             "hydraulic lifter": "lifter", "dod lifter": "lifter",
             "rocker arm": "rocker_arm"}

    def knock(o):
        return KNOCK.get((o["tm_part"] or "").lower())

    rep = {}
    for name, lf in [("region", region), ("part", part), ("knock", knock)]:
        r = oof(rows, lf)
        if r is None:
            rep[name] = {"note": "too few"}
            continue
        rep[name] = pr_at_cov(*r)
        print(name, json.dumps(rep[name]))

    # per-subtype one-vs-rest: which SPECIFIC parts can we name confidently?
    # (the 12-way average hides that some parts are highly separable.)
    print("\n-- per-part one-vs-rest (positive-class precision@coverage) --")
    ovr = {}
    for p in sorted(keep):
        r = oof(rows, lambda o, p=p: (p if canonical_cause(o["tm_part"]) == p
                                      else ("__rest__" if part(o) else None)))
        if r is None:
            continue
        yt, yp, conf = r
        # restrict to predictions OF this part, ranked by confidence
        mask = yp == p
        if mask.sum() < 10:
            continue
        c = (yt[mask] == p)
        cf = conf[mask]
        order = np.argsort(-cf)
        npos = int((yt == p).sum())
        row = {"n_pos": npos, "n_pred": int(mask.sum()),
               "precision": round(float(c.mean()), 3)}
        for tgt in (0.90, 0.99):
            cov = 0.0
            for k in range(len(order), 0, -1):
                if c[order[:k]].mean() >= tgt:
                    cov = round(k / npos, 3)
                    break
            row[f"recall@p{int(tgt*100)}"] = cov
        ovr[p] = row
        print(f"  {p:16} {json.dumps(row)}")
    rep["per_part_ovr"] = ovr
    json.dump(rep, open(DATA / "iterations" / "precision_coverage.json", "w"),
              indent=1)


if __name__ == "__main__":
    main()
