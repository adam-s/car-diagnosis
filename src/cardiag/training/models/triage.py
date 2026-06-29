"""Train + iterate the validated coarse triage (engine-internal vs running-gear)
with honest calibrated confidence. The metric we improve is held-out confident
coverage: cov@p90 / cov@p80 on a FIXED creator hold-out (unseen creators), plus
creator-grouped CV for the error bar. Saves the best artifact for the demo.

  uv run training/models/triage.py --gate strict --feat clap            # baseline
  uv run training/models/triage.py --gate medium --feat clap+ast --save # an iteration

Configs let us iterate honestly: data gate (curation depth), feature stack,
calibration method, multi-window TTA flag (consumed by the embed cache).
"""
import argparse
import glob
import hashlib
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
ENGINE = {"engine_internal", "valvetrain", "low_oil"}
CHASSIS = {"suspension", "driveline", "steering"}
CORPORA = [paths.YT_DATA / "corpus.enriched.tiered.jsonl",
           paths.YT_DATA / "corpus.jsonl",
           paths.TT_DATA / "corpus_labeled.tiered.jsonl",
           paths.REDDIT_DATA / "corpus.jsonl"]
# which L1 sound types are plausible for each region (drop mislabeled audio)
L1_OK = {
    "engine": {"ticking or clicking noise", "knocking or clunking noise",
               "humming or droning roar", "rattling noise",
               "normal smooth engine idle", "high-pitched whining noise"},
    "chassis": {"knocking or clunking noise", "grinding noise",
                "rattling noise", "squealing or squeaking noise",
                "humming or droning roar", "ticking or clicking noise"},
}
GATES = {  # mech_confirm floor, max clip seconds, apply L1 domain rule
    "strict": (0.5, 12, True),
    "medium": (0.4, 13, True),
    "loose":  (0.3, 15, False),
    "scale":  (0.35, 14, True),
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
                                   float(r.get("end", 0)) - float(r.get("start", 0)),
                                   r.get("l1"))
    return q


def feats(names):
    out = {}
    for n in names:
        fn = {"clap": "clap_embeddings.npz", "ast": "ast_embeddings.npz"}[n]
        z = np.load(DATA / fn, allow_pickle=True)
        X = z["X"] / (np.linalg.norm(z["X"], axis=1, keepdims=True) + 1e-9)
        out[n] = dict(zip(z["ids"], X))
    return out


def load(gate, feat_names):
    chan, qmap = channel_map(), quality_map()
    F = feats(feat_names)
    mech_min, dur_max, l1rule = GATES[gate]
    reg = lambda c: ("engine" if c in ENGINE
                     else "chassis" if c in CHASSIS else None)
    X, y, cre = [], [], []
    for l in open(DATA / "corpus_textmined_labels.jsonl"):
        o = json.loads(l)
        r = reg(o.get("tm_category"))
        hi = o.get("tm_explicit") and (o.get("tm_trust") == "timestamp"
                                       or o.get("platform") == "tiktok")
        if not r or not hi or any(o["eid"] not in F[n] for n in feat_names):
            continue
        mech, dur, l1 = qmap.get(o["eid"].split(":", 1)[1], (None, 0, None))
        if (mech or 0) < mech_min or dur > dur_max:
            continue
        if l1rule and l1 and l1 not in L1_OK[r]:
            continue
        X.append(np.concatenate([F[n][o["eid"]] for n in feat_names]))
        y.append(r)
        cre.append("yt:" + chan.get(o["video"], o["video"])
                   if o["platform"] == "youtube"
                   else o["platform"][:2] + ":" + o["video"])
    return np.array(X), np.array(y), np.array(cre)


def model(method):
    base = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=3000, class_weight="balanced"))
    return CalibratedClassifierCV(base, method=method, cv=3)


def cov_at(corr, conf, t):
    c = corr[np.argsort(-conf)]
    return next((k / len(c) for k in range(len(c), 0, -1)
                 if c[:k].mean() >= t), 0.0)


def oof_pred(X, y, cre, method):
    """Creator-grouped OOF probs: each clip scored by a model blind to its
    creator. Used both for CV and to find likely-mislabeled clips."""
    P = np.zeros(len(y))
    pred = np.empty(len(y), dtype=object)
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    for tr, te in skf.split(X, y, cre):
        clf = model(method).fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])
        cls = np.array(clf.classes_)
        pred[te] = cls[p.argmax(1)]
        P[te] = p.max(1)
    return pred, P


def clean_mask(X, y, cre, method, conf_drop):
    """Mark likely-mislabeled clips: OOF prediction wrong AND confident. These
    high-confidence errors are what cap cov@p90; most are bad labels."""
    if conf_drop >= 1:
        return np.ones(len(y), bool)
    pred, P = oof_pred(X, y, cre, method)
    return ~((pred != y) & (P >= conf_drop))


def evaluate(X, y, cre, method, conf_drop=1.0):
    # creator-grouped CV (clean training within each fold; test untouched)
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    covs = []
    for tr, te in skf.split(X, y, cre):
        keep = clean_mask(X[tr], y[tr], cre[tr], method, conf_drop)
        clf = model(method).fit(X[tr][keep], y[tr][keep])
        p = clf.predict_proba(X[te])
        corr = (np.array(clf.classes_)[p.argmax(1)] == y[te])
        covs.append(cov_at(corr, p.max(1), 0.90))
    # fixed creator hold-out, robust over seeds; clean only the training side
    hocov90, hocov80, hoacc, removed = [], [], [], 0
    for seed in range(4):
        h = np.array([(int(hashlib.md5((c + str(seed)).encode()).hexdigest(), 16)
                       % 4 == 0) for c in cre])
        keep = clean_mask(X[~h], y[~h], cre[~h], method, conf_drop)
        removed += int((~keep).sum())
        clf = model(method).fit(X[~h][keep], y[~h][keep])
        p = clf.predict_proba(X[h])
        cls = np.array(clf.classes_)
        corr = (cls[p.argmax(1)] == y[h])
        conf = p.max(1)
        hocov90.append(cov_at(corr, conf, .90))
        hocov80.append(cov_at(corr, conf, .80))
        hoacc.append(corr.mean())
    return {
        "n": len(y), "bal": dict(Counter(y)),
        "dropped": removed // 4,
        "cv_cov@p90": f"{np.mean(covs):.3f}±{np.std(covs):.3f}",
        "ho_acc": round(float(np.mean(hoacc)), 3),
        "ho_cov@p90": f"{np.mean(hocov90):.3f}±{np.std(hocov90):.3f}",
        "ho_cov@p80": f"{np.mean(hocov80):.3f}±{np.std(hocov80):.3f}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", default="strict", choices=list(GATES))
    ap.add_argument("--feat", default="clap")
    ap.add_argument("--method", default="isotonic", choices=["isotonic", "sigmoid"])
    ap.add_argument("--clean-conf", type=float, default=1.0,
                    help="drop training clips OOF-misclassified at >= this conf")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    feat_names = args.feat.split("+")

    X, y, cre = load(args.gate, feat_names)
    res = evaluate(X, y, cre, args.method, args.clean_conf)
    print(json.dumps({"gate": args.gate, "feat": args.feat, "method": args.method,
                      "clean": args.clean_conf, **res}))

    if args.save:
        keep = clean_mask(X, y, cre, args.method, args.clean_conf)
        clf = model(args.method).fit(X[keep], y[keep])
        joblib.dump({"model": clf, "classes": list(clf.classes_),
                     "feat": feat_names, "task": "engine_vs_running_gear"},
                    DATA / "triage_model.joblib")
        print(f"saved triage_model.joblib (feat={args.feat}, "
              f"n={int(keep.sum())}/{len(y)})")


if __name__ == "__main__":
    main()
