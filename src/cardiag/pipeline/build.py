"""End-to-end build: scrape -> clean -> label -> embed -> train -> model.

This is the loop a fresh clone runs. It is deliberately self-contained: it needs
only ``yt-dlp`` + CLAP (downloaded from Hugging Face on first use) — **no LLM and
no external datasets**. Labels come from what is knowable at scrape time:

  * ``kind`` (fault / normal) — which query set discovered the video.
  * ``l1`` sound-type and the mechanical/tool gating — CLAP zero-shot.
  * ``cause`` (part family) — keyword match on the video title.

That makes a weaker model than the research pipeline (which adds LLM fusion and
verified external anchors — see ``docs/``), but it runs from nothing and produces
a real, working classifier. That is the point: it teaches the loop by running it.

    cardiag scrape youtube       # discover -> download -> clean -> corpus.jsonl
    cardiag train                # corpus -> CLAP embeddings -> heads -> model
    cardiag diagnose clip.wav    # now works
    cardiag demo                 # all of the above, small, in one command
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from cardiag import config, paths
from cardiag.training.prep import causes

# Map a canonical cause family (underscored) to the coarse triage class
# (engine-internal vs running-gear). Cause labels are normalized to underscores.
_ENGINE = {"engine_internal", "low_oil", "fuel_ignition", "belt", "accessories",
           "alternator", "water_pump", "turbo", "exhaust", "ac_compressor",
           "fuel_pump"}
_CHASSIS = {"wheel_bearing", "brakes", "cv_joint", "suspension", "differential",
            "tires", "power_steering"}


# --------------------------------------------------------------------- scrape
def scrape_youtube(per_query: int = 3, max_videos: int = 40) -> int:
    """Discover fault+normal videos, download/clean each, write corpus.jsonl.

    Returns the number of labeled clips written.
    """
    import random

    from cardiag.audio.clap import Clap
    from cardiag.ingest.youtube import discover, pipeline

    paths.ensure_data_dirs()
    discover.main(per_query)
    work = json.loads((paths.YT_DATA / "worklist.json").read_text())
    # interleave fault/normal so a capped run keeps both classes
    random.Random(0).shuffle(work)
    work = work[:max_videos]

    clap = Clap()
    corpus = paths.YT_DATA / "corpus.jsonl"
    n_clips = 0
    with open(corpus, "w") as out:
        for i, w in enumerate(work):
            try:
                recs = pipeline.process_video(w["id"], w["title"], clap=clap)
            except Exception as e:  # a dead/blocked video must not kill the run
                print(f"  skip {w['id']}: {type(e).__name__}")
                continue
            for r in recs:
                if not r.get("file"):
                    continue
                idx = r["clip_id"].rsplit("_", 1)[-1]
                wav = paths.YT_DATA / "clips" / r["video"] / f"clip_{idx}.wav"
                if not wav.exists():
                    continue
                r["wav"] = str(wav)          # absolute, layout-independent
                r["kind"] = w["kind"]
                out.write(json.dumps(r) + "\n")
                n_clips += 1
            print(f"  [{i+1}/{len(work)}] {w['kind']:<6} {w['id']}  "
                  f"clips so far: {n_clips}", flush=True)
    print(f"\nscraped {n_clips} labeled clips -> {corpus}")
    return n_clips


def load_corpus() -> list[dict]:
    """Every labeled clip across platforms that still has its wav on disk."""
    rows: list[dict] = []
    for base in (paths.YT_DATA, paths.TT_DATA, paths.REDDIT_DATA):
        f = base / "corpus.jsonl"
        if not f.exists():
            continue
        for line in open(f):
            r = json.loads(line)
            wav = r.get("wav")
            if wav and Path(wav).exists():
                rows.append(r)
    return rows


# ---------------------------------------------------------------- label helpers
def _cause_of(row: dict):
    """Canonical cause family (underscored) from the title keyword candidates.

    The candidates are ``config.L2_KEYWORDS`` keys produced at scrape time.
    ``causes.canonical_cause`` handles the values it knows; anything else that is
    a recognized part family falls back to its underscored key.
    """
    for part in row.get("l2_candidates", []):
        c = causes.canonical_cause(part)
        if not c or c == "other":
            c = part if part in config.L2_KEYWORDS else None
        if c:
            return c.replace(" ", "_")   # one consistent underscored form
    return None


def _knock_of(row: dict):
    l1 = row.get("l1") or ""
    if "knock" in l1:
        return "knock"
    if l1 == config.L1_NORMAL:
        return "normal_idle"
    return None


def _triage_of(row: dict):
    if row.get("kind") != "fault":
        return None
    c = _cause_of(row)
    if c in _ENGINE:
        return "engine"
    if c in _CHASSIS:
        return "chassis"
    return None


# ----------------------------------------------------------------- train
def _fit(rows, labelf, embed, min_class: int):
    """Fit a calibrated linear head with a leakage-safe (group-by-video) report.

    Returns ``(estimator, report)``. Falls back to a constant ``DummyClassifier``
    when there isn't enough data for two classes, so the head always exists.
    """
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    data = [(embed[r["clip_id"]], labelf(r), r.get("video", r["clip_id"]))
            for r in rows if labelf(r) is not None and r["clip_id"] in embed]
    counts = Counter(lbl for _, lbl, _ in data)
    keep = {c for c, n in counts.items() if n >= min_class}
    data = [(x, lbl, g) for x, lbl, g in data if lbl in keep]
    labels = {lbl for _, lbl, _ in data}

    if len(labels) < 2:
        only = next(iter(labels), "unknown")
        X = np.array([x for x, _, _ in data]) if data else np.zeros((1, 512))
        y = [lbl for _, lbl, _ in data] or [only]
        clf = DummyClassifier(strategy="prior").fit(X, y)
        return clf, {"degenerate": True, "classes": sorted(labels), "n": len(data)}

    # group-held-out 25% by video id (no leakage)
    groups = sorted({g for _, _, g in data})
    test_g = set(groups[::4])
    tr = [(x, y) for x, y, g in data if g not in test_g]
    te = [(x, y) for x, y, g in data if g in test_g]
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=3000, class_weight="balanced"))
    clf.fit([x for x, _ in tr], [y for _, y in tr])
    report = {"classes": sorted(labels), "n_train": len(tr), "n_test": len(te)}
    if te:
        pred = clf.predict([x for x, _ in te])
        gold = [y for _, y in te]
        report["held_out_acc"] = round(float(np.mean([p == g for p, g in zip(pred, gold)])), 3)
        report["majority"] = round(max(Counter(gold).values()) / len(gold), 3)
    return clf, report


def train(min_class: int = 2) -> dict:
    """Embed every scraped clip with CLAP and train the heads + triage model."""
    import librosa

    from cardiag.audio.clap import Clap

    rows = load_corpus()
    if len(rows) < 8:
        raise SystemExit(
            f"only {len(rows)} clips in the corpus — run `cardiag scrape youtube` "
            f"first (try a larger --per-query / --max-videos).")

    print(f"embedding {len(rows)} clips with CLAP…", flush=True)
    clap = Clap()
    embed: dict[str, np.ndarray] = {}
    for i, r in enumerate(rows):
        y, _ = librosa.load(r["wav"], sr=config.SR_CLAP, mono=True)
        if len(y) < config.SR_CLAP // 2:
            continue
        embed[r["clip_id"]] = clap.embed([y])[0]
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    heads, report = {}, {}
    heads["kind"], report["kind"] = _fit(
        [r for r in rows if r.get("kind") in ("fault", "normal")],
        lambda r: r.get("kind"), embed, min_class)
    heads["knock"], report["knock"] = _fit(rows, _knock_of, embed, min_class)
    heads["cause"], report["cause"] = _fit(
        [r for r in rows if r.get("kind") == "fault"], _cause_of, embed, min_class)

    if {"fault", "normal"} - set(report["kind"].get("classes", [])):
        raise SystemExit(
            "could not train the fault/normal head — the scrape produced only one "
            "class. Scrape more videos (both fault and normal queries run).")

    paths.TRAIN_DATA.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump({"heads": heads, "emb": "clap"}, paths.MODEL_CLAP)

    triage_clf, report["triage"] = _fit(rows, _triage_of, embed, min_class)
    classes = list(getattr(triage_clf, "classes_",
                           report["triage"].get("classes", ["engine", "chassis"])))
    joblib.dump({"model": triage_clf, "classes": classes}, paths.MODEL_TRIAGE)

    (paths.TRAIN_DATA / "train_report.json").write_text(json.dumps(report, indent=2))
    print("\n=== training report ===")
    print(json.dumps(report, indent=2))
    print(f"\nsaved model -> {paths.MODEL_CLAP}")
    print(f"saved triage -> {paths.MODEL_TRIAGE}")
    return report


# ----------------------------------------------------------------- demo
def demo(per_query: int = 1, max_videos: int = 18) -> None:
    """The whole loop, small, in one command — for a fresh clone."""
    print("STEP 1/3  scrape + clean (YouTube)\n" + "-" * 40)
    n = scrape_youtube(per_query=per_query, max_videos=max_videos)
    if n < 8:
        raise SystemExit("scrape produced too few clips; try `cardiag demo` again "
                         "or `cardiag scrape youtube --per-query 2`.")
    print("\nSTEP 2/3  embed + train\n" + "-" * 40)
    train()
    print("\nSTEP 3/3  inference\n" + "-" * 40)
    from cardiag import Classifier
    clf = Classifier.load()
    # diagnose one of the clips we just scraped as a smoke check
    sample = load_corpus()[0]["wav"]
    print(json.dumps(clf.diagnose(sample).to_dict(), indent=1))
    print("\n✓ loop complete: scraped, cleaned, trained, and diagnosed from scratch.")
