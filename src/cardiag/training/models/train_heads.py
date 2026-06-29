"""Train linear heads on frozen CLAP embeddings: the first trained model.

What's honest about each head (inference is audio-only; labels are weak):
  cause : the REAL test. Labels came from text, features from audio, so this
          measures whether audio embeddings carry cause signal at all. The
          number to beat: zero-shot CLAP got 45.7% vs 42.4% baseline on
          verified external labels.
  kind  : fault/normal/nonauto. Mostly text-backed labels; useful for the app.
  l1    : scrape labels ARE CLAP outputs, so training on them with CLAP
          features is distillation; its only honest eval is the verified
          Car-Engine sound types (where zero-shot knock recall was 0%).

Tier ablation on the cause head (gold vs gold+silver vs +bronze) tells us
whether the weak tiers help or poison; cheap with linear heads.

Usage (after embed.py):
    uv run training/models/train_heads.py

Writes data/training/model_eval.json + heads.joblib
"""
import json
from collections import Counter

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from cardiag import paths

OUT = paths.TRAIN_DATA

# external causes that exist as scrape cause groups (low_oil doesn't)
EXT_CAUSE_OVERLAP = {"brakes", "belt", "power_steering", "accessories",
                     "fuel_ignition"}


def load_split(name, emb):
    rows = [json.loads(l) for l in open(OUT / f"{name}.jsonl")]
    return [r for r in rows if r["id"] in emb]


def xy(rows, emb, label_fn):
    pairs = [(r, label_fn(r)) for r in rows]
    pairs = [(r, y) for r, y in pairs if y is not None]
    X = np.array([emb[r["id"]] for r, _ in pairs])
    y = [y for _, y in pairs]
    return X, y


def fit(X, y, class_weight=None):
    # default (unbalanced) answers "beats majority baseline?"; balanced
    # answers "recalls rare faults?"; report both, they trade off.
    return LogisticRegression(max_iter=3000, class_weight=class_weight,
                              C=1.0).fit(X, y)


def topk_acc(clf, X, y, k):
    proba = clf.predict_proba(X)
    top = np.argsort(-proba, axis=1)[:, :k]
    classes = np.array(clf.classes_)
    return float(np.mean([yi in classes[t] for yi, t in zip(y, top)]))


def macro_recall(clf, X, y):
    pred = clf.predict(X)
    recalls = []
    for c in set(y):
        idx = [i for i, yi in enumerate(y) if yi == c]
        recalls.append(np.mean([pred[i] == c for i in idx]))
    return float(np.mean(recalls))


def eval_head(clf, X, y, ks=(1, 3)):
    if len(y) == 0:
        return None
    out = {"n": len(y)}
    for k in ks:
        out[f"top{k}"] = round(topk_acc(clf, X, y, k), 3)
    out["macro_recall"] = round(macro_recall(clf, X, y), 3)
    out["majority_baseline"] = round(max(Counter(y).values()) / len(y), 3)
    return out


def main():
    z = np.load(OUT / "clap_embeddings.npz", allow_pickle=True)
    # L2-normalize to match train_best.py + the inference path (classifier.py);
    # otherwise these heads train in a different feature space than deployment.
    emb = {i: v / (np.linalg.norm(v) + 1e-9) for i, v in zip(z["ids"], z["X"])}
    train = load_split("train", emb)
    val = load_split("val", emb)
    test = load_split("test", emb)
    ext = load_split("external_eval", emb)
    rep = {"n_embedded": len(emb)}

    # ---------------- cause head (the real test) ----------------
    cause_label = lambda r: (r.get("cause")
                             if r.get("kind") == "fault"
                             and r.get("cause") not in (None, "other")
                             else None)
    tiers_of = lambda rows, ts: [r for r in rows
                                 if r.get("tier") in ts or r["kind"] != "fault"]
    Xv, yv = xy(val, emb, cause_label)
    ablation = {}
    best = {"clf": None, "key": None, "top1": -1}
    for key, ts in [("gold", {"gold"}), ("gold+silver", {"gold", "silver"}),
                    ("gold+silver+bronze", {"gold", "silver", "bronze"})]:
        Xtr, ytr = xy(tiers_of(train, ts), emb, cause_label)
        if len(set(ytr)) < 2:
            continue
        # default weighting -> beat-baseline; balanced -> rare-class recall
        clf = fit(Xtr, ytr)
        clf_bal = fit(Xtr, ytr, class_weight="balanced")
        ablation[key] = {"n_train": len(ytr),
                         "val_default": eval_head(clf, Xv, yv),
                         "val_balanced": eval_head(clf_bal, Xv, yv)}
        if ablation[key]["val_default"] and \
                ablation[key]["val_default"]["top1"] > best["top1"]:
            best = {"clf": clf, "clf_bal": clf_bal, "key": key,
                    "top1": ablation[key]["val_default"]["top1"]}
    rep["cause_tier_ablation"] = ablation
    rep["cause_best_config"] = best["key"]

    Xte, yte = xy(test, emb, cause_label)
    rep["cause_test"] = {"default": eval_head(best["clf"], Xte, yte),
                         "balanced": eval_head(best["clf_bal"], Xte, yte)}
    # verified external eval, restricted to overlapping cause classes
    ext_cause = [r for r in ext if r.get("cause") in EXT_CAUSE_OVERLAP]
    Xe = np.array([emb[r["id"]] for r in ext_cause])
    ye = [r["cause"] for r in ext_cause]
    if len(ye):
        rep["cause_external_verified"] = {
            "default": eval_head(best["clf"], Xe, ye),
            "balanced": eval_head(best["clf_bal"], Xe, ye)}

    # ---------------- kind head ----------------
    kind_label = lambda r: r.get("kind")
    Xtr, ytr = xy(train, emb, kind_label)
    kind_clf = fit(Xtr, ytr)
    kind_bal = fit(Xtr, ytr, class_weight="balanced")
    rep["kind"] = {
        "val": eval_head(kind_clf, *xy(val, emb, kind_label), ks=(1,)),
        "test": eval_head(kind_clf, *xy(test, emb, kind_label), ks=(1,)),
    }
    # verified fault-vs-normal on external (zero-shot pipeline rule: 0.712).
    # external is ~50/50, so the majority-biased default under-performs the
    # balanced head here; report both.
    ext_fn = [r for r in ext if r["kind"] in ("fault", "normal")]
    Xe = np.array([emb[r["id"]] for r in ext_fn])
    ye = [r["kind"] for r in ext_fn]
    fvn = lambda c: round(float(np.mean(
        [(p == "normal") == (g == "normal")
         for p, g in zip(c.predict(Xe), ye)])), 3)
    rep["kind"]["external_fault_vs_normal_acc"] = {
        "default": fvn(kind_clf), "balanced": fvn(kind_bal),
        "zero_shot_pipeline_rule": 0.712}

    # ---------------- l1 head (distillation; honest eval = verified L1) ----
    l1_label = lambda r: r.get("l1")
    Xtr, ytr = xy(train, emb, l1_label)
    l1_clf = fit(Xtr, ytr)
    ext_l1 = [r for r in ext if r.get("l1")]
    Xe = np.array([emb[r["id"]] for r in ext_l1])
    ye = [r["l1"] for r in ext_l1]
    pred = l1_clf.predict(Xe)
    rep["l1_external_verified"] = {
        "n": len(ye),
        "acc": round(float(np.mean(pred == np.array(ye))), 3),
        "knock_recall": round(float(np.mean(
            [p == "knock" for p, g in zip(pred, ye) if g == "knock"])), 3)
        if any(g == "knock" for g in ye) else None,
    }

    joblib.dump({"cause": best["clf"], "cause_balanced": best["clf_bal"],
                 "kind": kind_clf, "l1": l1_clf}, OUT / "heads.joblib")
    json.dump(rep, open(OUT / "model_eval.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))
    print(f"\nwrote {OUT}/model_eval.json, heads.joblib")


if __name__ == "__main__":
    main()
