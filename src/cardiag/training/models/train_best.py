"""Train the recommended clean-teacher model + honest held-out eval.

The iteration research (docs/iteration-research.md) showed clean verified labels
beat weak scrape labels decisively on the SAME features. This trains the
deployable model that recommendation implies: clean teachers on a grouped-train
slice of verified data, evaluated on a leakage-safe held-out verified slice that
training never sees.

Heads (all linear on frozen embeddings):
    kind  fault-vs-normal   (all verified, grouped by source+vehicle)
    knock knock-vs-normal   (Car-Engine, grouped by vehicle)
    cause 6-class           (car-diagnostics; clip-grouped, caveat in doc)

Held-out = 25% of groups per task (deterministic). Reports test metrics with
bootstrap CIs and saves heads for inference.

Usage:
    uv run training/models/train_best.py --emb clap_embeddings.npz
"""
import argparse
import hashlib
import json
import re
from collections import Counter

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from cardiag import paths

DATA = paths.TRAIN_DATA
RNG = np.random.default_rng(0)
CD = {"brakes", "belt", "power_steering", "accessories", "fuel_ignition",
      "low_oil"}


def held_out(group):
    """25% of groups -> test, deterministic by group hash."""
    return int(hashlib.md5(group.encode()).hexdigest(), 16) % 4 == 0


def ce_vehicle(r):
    n = r["id"].split(":", 2)[-1].lower()
    n = re.sub(r"(engine )?(knock\w*|tick\w*|click\w*|rattl\w*|crank\w*|sound|"
               r"noise|normal|idle).*", "", n)
    return "ce:" + re.sub(r"[^a-z0-9]+", " ", n).strip()[:25]


def src_vehicle(r):
    return r["source"] + ":" + re.sub(r"[^a-z0-9]", "",
                                      r["id"].split(":")[-1].lower())[:18]


def boot_ci(correct, b=2000):
    correct = np.asarray(correct, float)
    if not len(correct):
        return [None, None]
    m = [correct[RNG.integers(0, len(correct), len(correct))].mean()
         for _ in range(b)]
    return [round(float(np.percentile(m, 2.5)), 3),
            round(float(np.percentile(m, 97.5)), 3)]


def train_eval(rows, labelf, groupf, emb, name):
    rows = [r for r in rows if r["id"] in emb and labelf(r)]
    tr = [r for r in rows if not held_out(groupf(r))]
    te = [r for r in rows if held_out(groupf(r))]
    gtr, gte = {groupf(r) for r in tr}, {groupf(r) for r in te}
    assert not (gtr & gte), "group leakage!"
    Xtr = np.array([emb[r["id"]] for r in tr])
    ytr = [labelf(r) for r in tr]
    Xte = np.array([emb[r["id"]] for r in te])
    yte = [labelf(r) for r in te]
    clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xtr,
                                                                         ytr)
    pred = clf.predict(Xte)
    corr = [p == g for p, g in zip(pred, yte)]
    out = {"name": name, "n_train": len(tr), "n_test": len(te),
           "train_groups": len(gtr), "test_groups": len(gte),
           "acc": round(float(np.mean(corr)), 3), "acc_ci": boot_ci(corr),
           "majority": round(max(Counter(yte).values()) / len(yte), 3)}
    kn = [p == "knock" for p, g in zip(pred, yte) if g == "knock"]
    if kn:
        out["knock_recall"] = round(float(np.mean(kn)), 3)
    return clf, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="clap_embeddings.npz")
    args = ap.parse_args()
    z = np.load(DATA / args.emb, allow_pickle=True)
    emb = {i: v / (np.linalg.norm(v) + 1e-9) for i, v in zip(z["ids"], z["X"])}
    ext = [json.loads(l) for l in open(DATA / "external_eval.jsonl")]

    heads, report = {}, {"emb": args.emb}
    heads["kind"], report["kind"] = train_eval(
        [r for r in ext if r["kind"] in ("fault", "normal")],
        lambda r: r["kind"], src_vehicle, emb, "fault_vs_normal")
    heads["knock"], report["knock"] = train_eval(
        [r for r in ext if r["source"] == "ext:car-engine-sounds"
         and r.get("l1") in ("knock", "normal_idle")],
        lambda r: r["l1"], ce_vehicle, emb, "knock_vs_normal")
    heads["cause"], report["cause"] = train_eval(
        [r for r in ext if r["source"] == "ext:car-diagnostics"],
        lambda r: r["cause"] if r.get("cause") in CD else None,
        lambda r: r["id"], emb, "cause_6class")

    joblib.dump({"heads": heads, "emb": args.emb},
                DATA / f"best_model_{args.emb.split('_')[0]}.joblib")
    json.dump(report, open(DATA / "iterations" /
                           f"best_model_{args.emb.split('_')[0]}.json", "w"),
              indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
