"""Verified-supervised ceiling: train AND test on clean labels.

The F1 diagnosis (cross-domain cause stuck at baseline) has two candidate
causes: weak SCRAPE LABELS or weak CLAP FEATURES. This separates them.

Split the verified car-diagnostics clips 50/50 by a hash of the filename
(deterministic, no recording-id available), train a head on one half's clean
labels, test on the other. Compare three regimes on the SAME verified test
half:
    A) scrape-trained   (weak labels)  -> our deployed setting
    B) verified-trained (clean labels)  -> ceiling with these features
    C) hybrid           (scrape + verified-train half)
If B >> A, weak labels are the bottleneck (collect/clean more). If B also low,
the feature space is the bottleneck (need a better backbone).

Usage: uv run training/eval/verified_supervised.py --emb clap_embeddings.npz
"""
import argparse
import hashlib
import json
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression

from cardiag import paths

DATA = paths.TRAIN_DATA

# car-diagnostics native fault classes (verified) -> scrape cause groups
CARDIAG_TO_CAUSE = {"brakes": "brakes", "belt": "belt",
                    "power_steering": "power_steering",
                    "accessories": "accessories",
                    "fuel_ignition": "fuel_ignition", "low_oil": "low_oil"}


def half(cid):
    return int(hashlib.md5(cid.encode()).hexdigest(), 16) % 2


def fit_eval(Xtr, ytr, Xte, yte):
    clf = LogisticRegression(max_iter=3000, class_weight="balanced",
                             C=1.0).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    top3 = np.mean([yi in clf.classes_[np.argsort(-pr)[:3]]
                    for yi, pr in zip(yte, clf.predict_proba(Xte))])
    return {"n_train": len(ytr), "n_test": len(yte),
            "top1": round(float(np.mean(pred == np.array(yte))), 3),
            "top3": round(float(top3), 3),
            "majority": round(max(Counter(yte).values()) / len(yte), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="clap_embeddings.npz")
    args = ap.parse_args()
    z = np.load(DATA / args.emb, allow_pickle=True)
    emb = {i: v / (np.linalg.norm(v) + 1e-9) for i, v in zip(z["ids"], z["X"])}

    ext = [json.loads(l) for l in open(DATA / "external_eval.jsonl")]
    train = [json.loads(l) for l in open(DATA / "train.jsonl")]

    # --- cause: car-diagnostics verified clips with a mapped cause ---
    cd = [r for r in ext if r["source"] == "ext:car-diagnostics"
          and r.get("cause") in CARDIAG_TO_CAUSE and r["id"] in emb]
    te = [r for r in cd if half(r["id"]) == 0]
    vtr = [r for r in cd if half(r["id"]) == 1]
    Xte = np.array([emb[r["id"]] for r in te])
    yte = [r["cause"] for r in te]

    # regime A: scrape-trained (overlapping classes only, for fair compare)
    classes = set(yte)
    str_rows = [r for r in train if r.get("kind") == "fault"
                and r.get("cause") in classes and r["id"] in emb
                and r.get("tier") in ("gold", "silver")]
    Xa = np.array([emb[r["id"]] for r in str_rows])
    ya = [r["cause"] for r in str_rows]
    # regime B: verified-trained
    Xb = np.array([emb[r["id"]] for r in vtr])
    yb = [r["cause"] for r in vtr]
    # regime C: hybrid
    Xc, yc = np.vstack([Xa, Xb]), ya + yb

    rep = {"emb": args.emb, "cause": {
        "classes": sorted(classes),
        "A_scrape_trained": fit_eval(Xa, ya, Xte, yte),
        "B_verified_trained": fit_eval(Xb, yb, Xte, yte),
        "C_hybrid": fit_eval(Xc, yc, Xte, yte)}}

    # --- kind: verified fault vs normal, same split ---
    fn = [r for r in ext if r["kind"] in ("fault", "normal") and r["id"] in emb]
    te = [r for r in fn if half(r["id"]) == 0]
    vtr = [r for r in fn if half(r["id"]) == 1]
    Xte = np.array([emb[r["id"]] for r in te])
    yte = [r["kind"] for r in te]
    Xb = np.array([emb[r["id"]] for r in vtr])
    yb = [r["kind"] for r in vtr]
    str_rows = [r for r in train if r["id"] in emb]
    Xa = np.array([emb[r["id"]] for r in str_rows])
    ya = [r["kind"] for r in str_rows]
    rep["kind"] = {
        "A_scrape_trained": fit_eval(Xa, ya, Xte, yte),
        "B_verified_trained": fit_eval(Xb, yb, Xte, yte)}

    json.dump(rep, open(DATA / "iterations" /
                        f"verified_supervised_{args.emb.split('_')[0]}.json",
                        "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
