"""Rigorous, reproducible scorecard: the project's answer to "is it real, and did
this change actually help?"

Implements the evaluation protocol from the literature (see
docs/DEFENSE.md):

  * **By-video grouping** so clips from one recording never straddle train/test
    (Kaufman 2012 leakage; Roberts 2021). `StratifiedGroupKFold` keeps the 74%
    fault ratio balanced across folds.
  * **Imbalance-aware metrics** (balanced accuracy, macro-F1, MCC (Chicco &
    Jurman 2020), AUROC + AUPRC (Saito & Rehmsmeier 2015)), never raw accuracy,
    and the majority baseline is always printed beside the number.
  * **Calibration**: ECE (equal-frequency bins, Naeini 2015) + Brier, so a
    "90% confidence" claim is auditable.
  * **Significance**: a by-video label-permutation null (Ojala & Garriga 2010)
    answers "above chance?"; the Nadeau–Bengio (2003) corrected resampled t-test
    answers "better than the previous version, or just noise?".

Run:  python -m cardiag.training.eval.scorecard         # full scorecard -> docs/SCORECARD.md
It embeds the local corpus with CLAP once and caches to data/training/corpus_emb.npz.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cardiag import config, paths

L1_NORMAL = "normal smooth engine idle"
ENGINE = {"engine_internal", "low_oil", "fuel_ignition", "belt", "accessories",
          "alternator", "water_pump", "turbo", "exhaust", "ac_compressor", "fuel_pump"}
CHASSIS = {"wheel_bearing", "brakes", "cv_joint", "cv_axle", "suspension",
           "differential", "tires", "wheel_tire", "power_steering"}


# --------------------------------------------------------------- embeddings
def embed_corpus(cache: Path | None = None) -> dict:
    """Embed every corpus clip with the real CLAP path; cache to npz. Returns a
    dict of aligned arrays (X, ids, kind, src, l1, cause, video)."""
    cache = cache or (paths.TRAIN_DATA / "corpus_emb.npz")
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return {k: z[k] for k in z.files}
    import librosa

    from cardiag.audio.embed import embed_clip
    from cardiag.pipeline import build
    from cardiag.pipeline.build import _cause_of
    rows = build.load_corpus()
    X, ids, kind, src, l1, cause, video = [], [], [], [], [], [], []
    for i, r in enumerate(rows):
        try:
            y, _ = librosa.load(r["wav"], sr=config.SR_CLAP, mono=True)
        except Exception:
            continue
        if len(y) < config.SR_CLAP // 2:
            continue
        X.append(embed_clip(y))
        ids.append(r["clip_id"]); video.append(r.get("video", r["clip_id"]))
        kind.append(r.get("kind") or ""); l1.append(r.get("l1") or "")
        cause.append(_cause_of(r) or "")
        w = r["wav"]
        src.append("youtube" if "/youtube/" in w else "tiktok" if "/tiktok/" in w
                   else "reddit" if "/reddit/" in w else "?")
        if (i + 1) % 100 == 0:
            print(f"  embedded {i+1}/{len(rows)}", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    d = {"X": np.array(X), "ids": np.array(ids), "kind": np.array(kind),
         "src": np.array(src), "l1": np.array(l1), "cause": np.array(cause),
         "video": np.array(video)}
    np.savez(cache, **d)
    return d


# ------------------------------------------------------------------- heads
def lr_head():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=3000, class_weight="balanced",
                                            random_state=0))


# ----------------------------------------------------------------- metrics
def ece(p, y_pos, bins: int = 5) -> float:
    """Expected Calibration Error with equal-FREQUENCY bins (robust on small n)."""
    p = np.asarray(p, float); y_pos = np.asarray(y_pos, float)
    m = ~np.isnan(p); p, y_pos = p[m], y_pos[m]
    if len(p) == 0:
        return float("nan")
    order = np.argsort(p); p, y_pos = p[order], y_pos[order]
    e = 0.0
    for chunk_p, chunk_y in zip(np.array_split(p, bins), np.array_split(y_pos, bins)):
        if len(chunk_p):
            e += len(chunk_p) / len(p) * abs(chunk_y.mean() - chunk_p.mean())
    return float(e)


def _binary_metrics(y, pred, proba, pos) -> dict:
    out = {"bal_acc": balanced_accuracy_score(y, pred),
           "macro_f1": f1_score(y, pred, average="macro"),
           "mcc": matthews_corrcoef(y, pred)}
    yp = (np.asarray(y) == pos).astype(int)
    if proba is not None and len(set(yp)) == 2:
        out["auroc"] = roc_auc_score(yp, proba)
        out["auprc"] = average_precision_score(yp, proba)
        out["brier"] = brier_score_loss(yp, proba)
    return out


# ---------------------------------------------------------- grouped CV core
def grouped_cv(X, y, groups, make=lr_head, n_splits: int = 5, repeats: int = 5,
               shuffle_labels: bool = False, base_seed: int = 0):
    """Repeated StratifiedGroupKFold. Returns (per-fold metric dicts, OOF proba,
    OOF y, pos label). With shuffle_labels, permutes labels by GROUP (whole-video)
    to build the permutation null without breaking the grouping."""
    y = np.asarray(y); groups = np.asarray(groups); classes = sorted(set(y))
    pos = classes[-1] if len(classes) == 2 else None
    if shuffle_labels:                       # permute at the video level
        rng = np.random.default_rng(base_seed)
        gl = {}
        for g, yy in zip(groups, y):
            gl.setdefault(g, yy)
        gs = list(gl); perm = rng.permutation([gl[g] for g in gs])
        gmap = dict(zip(gs, perm)); y = np.array([gmap[g] for g in groups])
    folds = []
    n_per_class_groups = min(len(set(groups[y == c])) for c in classes)
    ns = max(2, min(n_splits, n_per_class_groups))
    for rep in range(repeats):
        sgkf = StratifiedGroupKFold(n_splits=ns, shuffle=True, random_state=base_seed + rep)
        oof = np.full(len(y), np.nan)
        for tr, te in sgkf.split(X, y, groups):
            assert not (set(groups[tr]) & set(groups[te])), "video leak across split!"
            if len(set(y[tr])) < 2:
                continue
            clf = make().fit(X[tr], y[tr]); pred = clf.predict(X[te])
            pr = None
            if pos is not None:
                pr = clf.predict_proba(X[te])[:, list(clf.classes_).index(pos)]
                oof[te] = pr
            folds.append(_binary_metrics(y[te], pred, pr, pos))
        last_oof = (oof, y, pos)
    return folds, last_oof


def agg(folds, keys=("bal_acc", "macro_f1", "mcc", "auroc", "auprc", "brier")):
    out = {}
    for k in keys:
        vals = [f[k] for f in folds if k in f and not math.isnan(f[k])]
        out[k] = (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), 0.0)
    return out


def topk_cv(X, y, groups, ks=(1, 2, 3, 4), make=lr_head, n_splits=5, repeats=5):
    """Repeated grouped-CV top-k accuracy: is the true label within the top-k
    ranked predictions? The honest metric for a head whose *product is a ranked
    shortlist* (cause): top-1 badly understates a useful top-3. Returns micro
    (per-clip) and macro (per-class) top-k plus the random-chance baseline."""
    X = np.asarray(X); y = np.asarray(y); groups = np.asarray(groups)
    cls = sorted(set(y))
    ns = max(2, min(n_splits, min(len(set(groups[y == c])) for c in cls)))
    micro = {k: [] for k in ks}; macro = {k: [] for k in ks}
    for rep in range(repeats):
        for tr, te in StratifiedGroupKFold(ns, shuffle=True, random_state=rep).split(X, y, groups):
            if len(set(y[tr])) < 2:
                continue
            clf = make().fit(X[tr], y[tr])
            P = clf.predict_proba(X[te]); C = np.array(clf.classes_)
            order = np.argsort(-P, axis=1)
            for k in ks:
                hit = np.array([y[te][j] in C[order[j][:k]] for j in range(len(te))])
                micro[k].append(float(hit.mean()))
                per: dict = {}
                for j in range(len(te)):
                    per.setdefault(y[te][j], []).append(hit[j])
                macro[k].append(float(np.mean([np.mean(v) for v in per.values()])))
    return {k: {"micro": (float(np.mean(micro[k])), float(np.std(micro[k]))),
                "macro": (float(np.mean(macro[k])), float(np.std(macro[k]))),
                "random": round(k / len(cls), 3)} for k in ks}


def permutation_p(X, y, groups, observed_balacc, n: int = 200, make=lr_head):
    null = []
    for i in range(n):
        folds, _ = grouped_cv(X, y, groups, make=make, repeats=1, base_seed=1000 + i,
                              shuffle_labels=True)
        null.append(agg(folds)["bal_acc"][0])
    null = np.array(null)
    return float((np.sum(null >= observed_balacc) + 1) / (n + 1)), float(null.mean()), float(null.std())


def corrected_t(diffs, n_train, n_test):
    """Nadeau-Bengio corrected resampled t-test p-value (two-sided) for a vector of
    per-fold score differences (new - old). Corrects the variance for the
    train-set overlap between folds."""
    from scipy import stats
    diffs = np.asarray(diffs, float); k = len(diffs)
    if k < 2 or diffs.std(ddof=1) == 0:
        return float("nan")
    mean = diffs.mean(); var = diffs.var(ddof=1)
    corr = (1.0 / k + n_test / max(1, n_train))
    t = mean / math.sqrt(var * corr)
    return float(2 * (1 - stats.t.cdf(abs(t), df=k - 1)))


# ------------------------------------------------------- task definitions
def tasks(d):
    KIND, SRC, L1, CAUSE = d["kind"], d["src"], d["l1"], d["cause"]
    knock = np.array(["knock" if "knock" in x else ("normal_idle" if x == L1_NORMAL else "")
                      for x in L1])
    tri = np.array(["engine" if c in ENGINE else "chassis" if c in CHASSIS else ""
                    for c in CAUSE])
    return {
        "knock (knock vs normal-idle)": knock != "",
        "kind | YouTube (confound-free)": ((KIND == "fault") | (KIND == "normal")) & (SRC == "youtube"),
        "kind | all sources": (KIND == "fault") | (KIND == "normal"),
        "triage (engine vs running-gear)": (KIND == "fault") & (tri != ""),
        "cause (part family)": CAUSE != "",
    }, {"knock (knock vs normal-idle)": knock,
        "kind | YouTube (confound-free)": KIND,
        "kind | all sources": KIND,
        "triage (engine vs running-gear)": tri,
        "cause (part family)": CAUSE}


# ------------------------------------------------------------------- run
def run(out_md: Path | None = None) -> dict:
    d = embed_corpus()
    X, VID, SRC = d["X"], d["video"], d["src"]
    masks, labelmaps = tasks(d)
    report = {"corpus": {"n_clips": len(X), "n_videos": len(set(VID)),
                         "by_source": {str(k): int(v) for k, v in Counter(SRC).items()},
                         "by_kind": {str(k): int(v) for k, v in Counter(d["kind"]).items()}}}
    lines = ["# Scorecard — measured, by-video cross-validated\n",
             "Generated by `python -m cardiag.training.eval.scorecard`. Every number is "
             "5×5 repeated StratifiedGroupKFold (grouped by video — no leakage), "
             "class-weighted. Literature ceiling for in-the-wild machine-sound "
             "classification is mid-70s–low-80s AUROC (DCASE Task 2); ~90%+ on a "
             "confounded split is a red flag, not a triumph.\n",
             f"**Corpus:** {len(X)} clips, {len(set(VID))} videos, "
             f"sources={report['corpus']['by_source']}, kind={report['corpus']['by_kind']}\n",
             "| Head | n | classes | balAcc | macroF1 | MCC | AUROC | AUPRC | ECE | majority |",
             "|---|--:|---|---|---|---|---|---|--:|--:|"]
    heads = {}
    for name, m in masks.items():
        y = labelmaps[name][m]
        keep = y != ""
        Xs, ys, gs = X[m][keep], y[keep], VID[m][keep]
        if len(set(ys)) < 2:
            continue
        folds, (oof, oofy, pos) = grouped_cv(Xs, ys, gs)
        a = agg(folds)
        maj = max(Counter(ys).values()) / len(ys)
        ec = ece(oof, (oofy == pos)) if pos is not None else float("nan")
        g = lambda k: ("—" if math.isnan(a[k][0]) else f"{a[k][0]:.3f}±{a[k][1]:.2f}")
        ecs = "—" if math.isnan(ec) else f"{ec:.3f}"
        binc = "/".join(f"{str(k)}:{v}" for k, v in sorted(Counter(ys).items()))
        lines.append(f"| {name} | {len(ys)} | {binc} | {g('bal_acc')} | {g('macro_f1')} | "
                     f"{g('mcc')} | {g('auroc')} | {g('auprc')} | {ecs} | {maj:.3f} |")
        heads[name] = {"n": len(ys), "balacc": a["bal_acc"], "auroc": a["auroc"],
                       "auprc": a["auprc"], "mcc": a["mcc"], "ece": ec, "majority": maj,
                       "classes": {k: int(v) for k, v in Counter(ys).items()}}
        if len(set(ys)) > 2:                 # ranked-shortlist head -> report top-k
            heads[name]["topk"] = topk_cv(Xs, ys, gs)
    report["heads"] = heads

    # top-k for the ranked-shortlist (multiclass) heads: the metric that matches
    # how the product is used: "is the right answer in the 3-4 causes we show?"
    multi = {n: h for n, h in heads.items() if "topk" in h}
    if multi:
        lines += ["", "## Ranked-shortlist accuracy (top-k) — for multi-class heads",
                  "", "`cause` returns a ranked list with confidence, not one answer. "
                  "The honest metric is whether the true cause is *in the shortlist* — "
                  "top-1 alone badly understates it.", "",
                  "| Head | top-1 | top-2 | **top-3** | top-4 | random top-3 |",
                  "|---|--:|--:|--:|--:|--:|"]
        for n, h in multi.items():
            tk = h["topk"]
            cell = lambda k: f"{tk[k]['micro'][0]:.3f}"
            lines.append(f"| {n} | {cell(1)} | {cell(2)} | **{cell(3)}** | {cell(4)} | "
                         f"{tk[3]['random']:.3f} |")

    # permutation test on the honest cut (is fault/normal real once source is fixed?)
    name = "kind | YouTube (confound-free)"
    if name in heads:
        m = masks[name]; y = labelmaps[name][m]; keep = y != ""
        Xs, ys, gs = X[m][keep], y[keep], VID[m][keep]
        obs = heads[name]["balacc"][0]
        p, nm, ns = permutation_p(Xs, ys, gs, obs, n=200)
        report["permutation_kind_youtube"] = {"observed_balacc": obs, "null_mean": nm,
                                              "null_std": ns, "p_value": p}
        lines += ["", f"**Permutation null — {name}:** observed balAcc {obs:.3f} vs "
                  f"label-shuffled null {nm:.3f}±{ns:.3f}, **p={p:.3f}** "
                  f"({'REAL signal above chance' if p < 0.05 else 'not distinguishable from chance'})."]

    # source confound, Stage 0: how predictable is the recording source?
    folds, _ = grouped_cv(X, SRC, VID)   # multiclass -> only balAcc/macroF1/mcc populate
    a = agg(folds, keys=("bal_acc", "macro_f1", "mcc"))
    report["source_confound"] = {"predict_source_balacc": a["bal_acc"], "mcc": a["mcc"]}
    lines += ["", "## Source confound (the dataset's structural risk)", "",
              f"All normals are YouTube; TikTok/Reddit are 100% fault. Predicting the "
              f"recording **source** from the embedding alone scores balAcc "
              f"{a['bal_acc'][0]:.3f}±{a['bal_acc'][1]:.2f} (MCC {a['mcc'][0]:.3f}) — "
              f"the shortcut is *available*. The honest fault/normal number is therefore "
              f"the **YouTube-only** row above (source held constant), not the all-sources "
              f"row, whose AUROC is confound-inflated."]

    repo_root = Path(__file__).resolve().parents[4]   # eval/training/cardiag/src/<root>
    out_md = out_md or (repo_root / "docs" / "SCORECARD.md")
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text("\n".join(lines) + "\n")
    (Path(out_md).parent / "scorecard.json").write_text(json.dumps(report, indent=2))
    print("\n".join(lines))
    print(f"\nwrote {out_md} + scorecard.json")
    return report


if __name__ == "__main__":
    run()
