"""Calibrated, human-facing CONFIDENCE for region triage + inference ablations.

The split_check finding: point accuracy under honest creator splits is weak
(held-out cov@p90 0.562, majority 0.785). The product answer is selective
prediction: the app reports a confidence band, and the HIGH band must be
empirically trustworthy. This measures that honestly:

  - OOF protocol: 5-fold creator-grouped CV; every clip's confidence comes
    from a model that never saw its creator. All tables below use OOF probs.
  - Ablations: CLAP vs AST vs concat vs +DSP, logistic vs small-MLP ensemble,
    sigmoid vs isotonic calibration. Picks the best by cov@p90, ECE tiebreak.
  - Reliability table: claimed confidence bin -> empirical accuracy (is 0.9
    really 0.9?), ECE.
  - Band spec: answer-iff-conf>=t table -> the HIGH/MEDIUM/LOW cutoffs the
    app should ship with, each with its historical accuracy.
  - Video-level: mean prob across a recording's clips (the app records ~20s,
    we predict per 5s window and aggregate): agreement across windows is
    itself confidence.

    uv run training/models/confidence.py
"""
import glob
import hashlib
import json
from collections import Counter, defaultdict

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cardiag import paths

DATA = paths.TRAIN_DATA
ENGINE = {"engine_internal", "valvetrain", "low_oil"}
CHASSIS = {"suspension", "driveline", "steering"}


def channel_map():
    m = {}
    for p in glob.glob(str(paths.YT_DATA / "meta" / "*" / "*.info.json")):
        try:
            d = json.load(open(p))
            m[d["id"]] = d.get("channel_id") or d.get("uploader_id") or d["id"]
        except Exception:
            pass
    return m


def feats():
    out = {}
    for name, fn in [("clap", "clap_embeddings.npz"),
                     ("ast", "ast_embeddings.npz"),
                     ("dsp2", "dsp_features2.npz")]:
        z = np.load(DATA / fn, allow_pickle=True)
        X = z["X"]
        if name != "dsp2":                       # embeddings: L2 normalize
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        out[name] = dict(zip(z["ids"], X))
    return out


def load_rows(F):
    chan = channel_map()
    reg = lambda c: ("engine" if c in ENGINE
                     else "chassis" if c in CHASSIS else None)
    rows = []
    for l in open(DATA / "corpus_textmined_clean.jsonl"):
        o = json.loads(l)
        r = reg(o["tm_category"])
        if not r or any(o["eid"] not in F[k] for k in F):
            continue                              # intersection -> fair ablation
        cre = ("yt:" + chan.get(o["video"], o["video"])
               if o["platform"] == "youtube"
               else o["platform"][:2] + ":" + o["video"])
        rows.append((o["eid"], r, o["video"], cre))
    return rows


def cov_at(correct, conf, t):
    c = correct[np.argsort(-conf)]
    for k in range(len(c), 0, -1):
        if c[:k].mean() >= t:
            return k / len(c)
    return 0.0


def ece(correct, conf, nbins=10):
    bins = np.linspace(0.5, 1.0, nbins + 1)
    e, n = 0.0, len(conf)
    for a, b in zip(bins, bins[1:]):
        m = (conf >= a) & (conf < b if b < 1 else conf <= b)
        if m.sum():
            e += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return e


def make_model(kind, method):
    if kind == "logistic":
        base = make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=3000, class_weight="balanced"))
        return [CalibratedClassifierCV(base, method=method, cv=3)]
    return [CalibratedClassifierCV(                      # mlp-ens5
        make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(256,), alpha=1e-3, max_iter=600,
            random_state=s)), method=method, cv=3) for s in range(5)]


def oof_probs(X, y, groups, kind, method):
    """Creator-grouped 5-fold OOF calibrated P(engine), P(chassis)."""
    classes = np.array(sorted(set(y)))
    P = np.zeros((len(y), len(classes)))
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    for tr, te in skf.split(X, y, groups):
        ps = []
        for m in make_model(kind, method):
            m.fit(X[tr], y[tr])
            p = m.predict_proba(X[te])
            ps.append(p[:, np.argsort(np.argsort(classes))] if
                      list(m.classes_) != list(classes) else p)
        P[te] = np.mean(ps, 0)
    return classes, P


def main():
    F = feats()
    rows = load_rows(F)
    eids = [r[0] for r in rows]
    y = np.array([r[1] for r in rows])
    vid = np.array([r[2] for r in rows])
    cre = np.array([r[3] for r in rows])
    print(f"region set (all-features intersection): {len(y)} clips | "
          f"{len(set(vid))} videos | {len(set(cre))} creators | "
          f"{dict(Counter(y))}\n")

    stack = lambda names: np.hstack([np.array([F[n][e] for e in eids])
                                     for n in names])
    CONFIGS = [
        ("clap            log/sig", ["clap"], "logistic", "sigmoid"),
        ("clap            log/iso", ["clap"], "logistic", "isotonic"),
        ("ast             log/sig", ["ast"], "logistic", "sigmoid"),
        ("clap+ast        log/sig", ["clap", "ast"], "logistic", "sigmoid"),
        ("clap+ast+dsp2   log/sig", ["clap", "ast", "dsp2"], "logistic", "sigmoid"),
        ("clap+ast        mlp5/sig", ["clap", "ast"], "mlp", "sigmoid"),
        ("clap+ast+dsp2   mlp5/sig", ["clap", "ast", "dsp2"], "mlp", "sigmoid"),
    ]
    results = {}
    for name, fs, kind, method in CONFIGS:
        classes, P = oof_probs(stack(fs), y, cre, kind, method)
        pred = classes[np.argmax(P, 1)]
        corr, conf = (pred == y), P.max(1)
        # video-level: average windows of the same recording
        vp = defaultdict(list)
        for v, p, t in zip(vid, P, y):
            vp[v].append((p, t))
        vcorr = np.array([classes[np.argmax(np.mean([p for p, _ in l], 0))]
                          == Counter(t for _, t in l).most_common(1)[0][0]
                          for l in vp.values()])
        results[name] = (classes, P, corr, conf)
        print(f"{name}: acc {corr.mean():.3f} | cov@p90 "
              f"{cov_at(corr, conf, .90):.3f} | cov@p95 "
              f"{cov_at(corr, conf, .95):.3f} | ECE {ece(corr, conf):.3f} | "
              f"video-acc {vcorr.mean():.3f} (n={len(vcorr)})")

    best = max(results, key=lambda k: (cov_at(results[k][2], results[k][3], .9),
                                       -ece(results[k][2], results[k][3])))
    classes, P, corr, conf = results[best]
    print(f"\nBEST: {best}")

    print("\nreliability (claimed -> empirical), OOF creator-grouped:")
    for a, b in [(.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, .95), (.95, 1.01)]:
        m = (conf >= a) & (conf < b)
        if m.sum():
            print(f"  claimed {a:.2f}-{min(b,1):.2f}: n={m.sum():4d}  "
                  f"empirical acc {corr[m].mean():.3f}")

    print("\nband spec (answer iff conf >= t):")
    print(f"  {'t':>5} {'answered':>9} {'acc':>6} {'engine-n':>9} {'engine-acc':>11}")
    for t in [.60, .70, .75, .80, .85, .90, .95]:
        m = conf >= t
        if not m.sum():
            continue
        em = m & (y == "engine")
        print(f"  {t:5.2f} {m.mean():8.1%} {corr[m].mean():6.3f} "
              f"{em.sum():9d} {corr[em].mean() if em.sum() else float('nan'):11.3f}")

    # fixed creator held-out, best config (same split as split_check)
    test_c = {c for c in set(cre)
              if int(hashlib.md5(c.encode()).hexdigest(), 16) % 4 == 0}
    tr = np.array([i for i in range(len(y)) if cre[i] not in test_c])
    te = np.array([i for i in range(len(y)) if cre[i] in test_c])
    fs, kind, method = next((f, k, m) for n, f, k, m in CONFIGS if n == best)
    X = stack(fs)
    ps = []
    for mdl in make_model(kind, method):
        mdl.fit(X[tr], y[tr])
        ps.append(mdl.predict_proba(X[te]))
    Pte = np.mean(ps, 0)
    ct = (classes[np.argmax(Pte, 1)] == y[te])
    print(f"\nFIXED held-out (creator 75/25), best config: acc {ct.mean():.3f}"
          f" | cov@p90 {cov_at(ct, Pte.max(1), .90):.3f}"
          f" | majority {max(Counter(y[te]).values())/len(te):.3f}")


if __name__ == "__main__":
    main()
