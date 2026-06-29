"""End-to-end build: scrape -> clean -> label -> embed -> train -> model.

This is the loop a fresh clone runs, across all three sources (YouTube, Reddit,
TikTok). It is deliberately self-contained: it needs only ``yt-dlp`` + CLAP
(downloaded on first use): **no LLM and no external datasets**. TikTok adds a
stealth browser (patchright) for discovery. Labels come from what is knowable at
scrape time:

  * ``kind`` (fault / normal): YouTube's fault vs normal query sets; Reddit and
    TikTok are fault-dominant sources, so they contribute the fault class.
  * ``l1`` sound-type and mechanical/tool gating: CLAP zero-shot.
  * ``cause`` (part family): keyword match on the title / caption.

Every platform funnels through one labeling function (:func:`_label_audio`) so a
clip looks identical regardless of source. That weaker-but-honest supervision
trains a real, working model from nothing: it teaches the loop by running it.

    cardiag scrape youtube|reddit|tiktok   # -> data/<platform>/corpus.jsonl
    cardiag train                          # corpus -> CLAP -> heads -> model
    cardiag diagnose clip.wav
    cardiag demo                           # all three, small, in one command
"""
from __future__ import annotations

import json
import os
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from cardiag import config, paths
from cardiag.training.prep import causes

# canonical cause family (underscored) -> coarse triage class
_ENGINE = {"engine_internal", "low_oil", "fuel_ignition", "belt", "accessories",
           "alternator", "water_pump", "turbo", "exhaust", "ac_compressor",
           "fuel_pump", "valvetrain", "rod_knock"}
_CHASSIS = {"wheel_bearing", "brakes", "cv_joint", "cv_axle", "suspension",
            "differential", "tires", "wheel_tire", "power_steering"}

# canonical cause family -> "where in the car" region (6 zones). This is the
# headline output: a recording -> a ranked region shortlist. The OOS sanity check
# (top-3 ~0.75 on the verified set) shows this generalizes where knock does not.
_REGION = {
    "engine": {"engine_internal", "rod_knock", "valvetrain", "low_oil",
               "fuel_ignition", "fuel_pump"},
    "accessory": {"belt", "alternator", "water_pump", "ac_compressor", "accessories"},
    "exhaust": {"exhaust", "turbo"},
    "drivetrain": {"cv_axle", "cv_joint", "differential"},
    "suspension/steering": {"suspension", "power_steering"},
    "brakes/wheels": {"brakes", "wheel_bearing", "wheel_tire", "tires"},
}
_CAUSE_TO_REGION = {c: z for z, cs in _REGION.items() for c in cs}


# ============================================================ unified labeling
def _progress(seq, desc):
    """A rich progress bar when available; a plain iterator otherwise."""
    seq = list(seq)
    try:
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            TextColumn,
            TimeElapsedColumn,
        )
        with Progress(TextColumn("  [cyan]{task.description}"), BarColumn(),
                      MofNCompleteColumn(), TimeElapsedColumn(),
                      transient=True) as pr:
            for item in pr.track(seq, description=desc):
                yield item
    except Exception:
        for item in seq:
            yield item


def _clap():
    from cardiag.audio.clap import Clap
    return Clap()


def _require(tool: str, fix: str) -> None:
    """Fail fast with a clear message if an external tool is missing (so a fresh
    clone gets 'install yt-dlp', not a raw FileNotFoundError mid-scrape)."""
    import shutil
    if not shutil.which(tool):
        raise SystemExit(f"'{tool}' not found on PATH — {fix}")


def _label_audio(wav, vid: str, title: str, kind: str, out_base: Path, clap,
                 cause: str | None = None) -> list[dict]:
    """Segment ONE audio file (any length) into labeled mechanical-span clips.

    This is the single segmentation path for the whole project: isolate the
    non-speech mechanical spans with the same cascade inference uses
    (:func:`cardiag.audio.cascade.candidate_regions`), score them with CLAP, and
    write one short clip WAV per surviving span. A 10-minute recording becomes N
    short spans exactly like a scraped clip, so training and inference always see
    the same unit. ``cause`` overrides the title-keyword label when the source
    already knows the part (e.g. a curated dataset folder).
    """
    import librosa
    import soundfile as sf

    from cardiag.audio.cascade import candidate_regions
    from cardiag.ingest.youtube.pipeline import gate, l2_from_text

    try:
        y16, _ = librosa.load(str(wav), sr=config.SR_CHEAP, mono=True)
    except Exception:
        return []
    regions, speech_frac = candidate_regions(y16, return_speech_frac=True)
    if not regions:                                   # phone clips: fall back whole
        dur = len(y16) / config.SR_CHEAP
        if dur < config.MIN_REGION_S:
            return []
        regions = [(0.0, round(min(dur, 10.0), 2))]

    y48, sr = librosa.load(str(wav), sr=config.SR_CLAP, mono=True)
    clips = [y48[int(s * sr):int(e * sr)] for s, e in regions]
    clips = [c for c in clips if len(c) >= sr // 2]
    if not clips:
        return []
    conf = clap.score(clips, config.CONFIRM_KEEP + config.CONFIRM_DROP)
    fault = clap.score(clips, config.FAULT_PROMPTS + config.TOOL_PROMPTS)
    l1p = clap.score(clips, config.L1_PROMPTS)

    out_dir = out_base / "clips" / vid
    out_dir.mkdir(parents=True, exist_ok=True)
    l2 = [cause] if cause else l2_from_text(title)    # explicit cause wins over keywords
    shop = speech_frac >= config.SHOP_SPEECH_FRAC
    recs = []
    for i, (cp, fp, lp) in enumerate(zip(conf, fault, l1p)):
        l1, cf, margin, status, _ = gate(cp, fp, lp, shop_context=shop)
        if status == "reject":
            continue
        f = out_dir / f"clip_{i:02d}.wav"
        sf.write(f, clips[i], sr)
        rec = {"clip_id": f"{vid}_{i:02d}", "video": vid, "wav": str(f),
               "kind": kind, "l1": l1, "l1_conf": round(cf, 3),
               "status": status, "l2_candidates": l2, "title": title}
        if cause:
            rec["cause"] = cause
        recs.append(rec)
    if not recs:
        try:
            out_dir.rmdir()
        except OSError:
            pass
    return recs


def ingest_dir(audio_dir, kind: str, cause: str | None = None,
               source: str = "local") -> int:
    """Bring-your-own-audio: segment every clip in a folder through the SAME cascade
    as scraping, into the corpus. Any length is handled: long recordings become
    multiple short spans. Then ``cardiag train`` consumes them like any other clip.

    The single coherent path for non-scraped data: a cloner with a folder of, say,
    'wheel_bearing' recordings runs one command and they're segmented + labeled
    identically to everything else."""
    import glob
    audio_dir = Path(audio_dir)
    if not audio_dir.exists():
        raise SystemExit(f"no such folder: {audio_dir}")
    files = [p for ext in ("wav", "mp3", "m4a", "ogg", "flac", "aac", "webm", "mp4")
             for p in glob.glob(str(audio_dir / f"**/*.{ext}"), recursive=True)]
    if not files:
        raise SystemExit(f"no audio files under {audio_dir}")
    out_base = paths.DATA / source
    clap, recs = _clap(), []
    for i, p in enumerate(files):
        vid = f"{source}_{Path(p).stem}"[:60]
        recs += _label_audio(p, vid, "", kind, out_base, clap, cause=cause)
        if (i + 1) % 25 == 0:
            print(f"  [{source} {i+1}/{len(files)}] spans so far: {len(recs)}", flush=True)
    n = _write_corpus(recs, out_base)
    print(f"ingested {len(files)} file(s) -> {n} mechanical-span clips in "
          f"{out_base/'corpus.jsonl'}  (kind={kind}, cause={cause or 'from title'})")
    return n


def _write_corpus(records, out_base: Path) -> int:
    """Merge new records into corpus.jsonl, deduped by clip_id, written ATOMICALLY.

    Critical: a scrape that yields nothing (flaky network, dead videos, blocked
    platform) must NEVER wipe a previously-built corpus. We merge with whatever is
    already on disk and write via a temp file + os.replace, so an interrupted or
    zero-yield run can't truncate or corrupt the corpus. Re-running scrape is
    therefore additive, not destructive.
    """
    out_base.mkdir(parents=True, exist_ok=True)
    dest = out_base / "corpus.jsonl"
    merged: dict = {}
    if dest.exists():                                   # keep prior good data
        for line in open(dest):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            merged[r.get("clip_id") or len(merged)] = r
    added = 0
    for r in records:
        key = r.get("clip_id") or f"_new{len(merged)}"
        if key not in merged:
            added += 1
        merged[key] = r
    if not merged:                                      # nothing old, nothing new
        print(f"  no clips to write; left {dest} untouched")
        return 0
    if not records:
        print(f"  this run added 0 clips; kept {len(merged)} existing in {dest}")
    tmp = dest.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as fh:
        for r in merged.values():
            fh.write(json.dumps(r) + "\n")
    os.replace(tmp, dest)                               # atomic
    return len(merged)


# ==================================================================== scrapers
def scrape_youtube(per_query: int = 3, max_videos: int = 40) -> int:
    """Discover fault+normal videos, download/clean each, write corpus.jsonl."""
    import random

    from cardiag.ingest.youtube import discover
    from cardiag.ingest.youtube.pipeline import acquire

    _require("yt-dlp", "pip install -e '.[scrape]'")
    _require("ffmpeg", "install ffmpeg (brew install ffmpeg / apt install ffmpeg)")
    paths.ensure_data_dirs()
    discover.main(per_query)
    work = json.loads((paths.YT_DATA / "worklist.json").read_text())
    random.Random(0).shuffle(work)                    # keep both classes when capped
    work = work[:max_videos]

    clap, recs = _clap(), []
    for i, w in enumerate(work):
        try:
            wav = acquire(w["id"])
        except Exception as e:
            print(f"  skip {w['id']}: {type(e).__name__}")
            continue
        recs += _label_audio(wav, w["id"], w["title"], w["kind"], paths.YT_DATA, clap)
        Path(wav).unlink(missing_ok=True)             # raw audio is transient
        print(f"  [youtube {i+1}/{len(work)}] {w['kind']:<6} clips: {len(recs)}",
              flush=True)
    n = _write_corpus(recs, paths.YT_DATA)
    print(f"youtube: {n} labeled clips -> {paths.YT_DATA/'corpus.jsonl'}")
    return n


def scrape_reddit(pages: int = 2, max_posts: int = 60) -> int:
    """Scrape r/MechanicAdvice-style posts (yt-dlp audio), clean + label each."""
    from cardiag.ingest.reddit import scrape as reddit

    _require("yt-dlp", "pip install -e '.[scrape]'")
    _require("ffmpeg", "install ffmpeg (brew install ffmpeg / apt install ffmpeg)")
    paths.ensure_data_dirs()
    reddit.main(pages)                                # -> posts.jsonl + audio/*.wav
    posts_f = paths.REDDIT_DATA / "posts.jsonl"
    if not posts_f.exists():
        print("reddit: no posts scraped")
        return 0
    posts = [json.loads(ln) for ln in open(posts_f)][:max_posts]
    clap, recs = _clap(), []
    for i, p in enumerate(posts):
        wav = paths.REDDIT_DATA / "audio" / f"{p['fullname']}.wav"
        if not wav.exists():
            continue
        recs += _label_audio(wav, p["fullname"], p.get("title", ""), "fault",
                             paths.REDDIT_DATA, clap)
        if (i + 1) % 20 == 0:
            print(f"  [reddit {i+1}/{len(posts)}] clips: {len(recs)}", flush=True)
    n = _write_corpus(recs, paths.REDDIT_DATA)
    print(f"reddit: {n} labeled clips -> {paths.REDDIT_DATA/'corpus.jsonl'}")
    return n


def scrape_tiktok(max_videos: int = 30, n_queries: int = 8, kind: str = "fault") -> int:
    """Discover clips via the stealth browser, download + label each with ``kind``.

    ``kind="fault"`` (default) uses the problem queries; ``kind="normal"`` uses the
    healthy-engine queries; scrape both to give `cardiag train` fault AND normal
    clips from TikTok, which breaks the recording-source confound (docs/MODEL_CARD.md).

    Needs the stealth browser: `pip install -e .[scrape]` then
    `python -m camoufox fetch`. TikTok anti-bot may block headless runs.
    """
    from cardiag.ingest.tiktok.discover import NORMAL_QUERIES, PROBLEM_QUERIES
    if kind not in ("fault", "normal"):
        raise SystemExit("tiktok kind must be 'fault' or 'normal'")

    _require("yt-dlp", "pip install -e '.[scrape]'")
    _require("ffmpeg", "install ffmpeg (brew install ffmpeg / apt install ffmpeg)")
    paths.ensure_data_dirs()
    queries = (NORMAL_QUERIES if kind == "normal" else PROBLEM_QUERIES)[:n_queries]

    # Prefer Camoufox (stealth Firefox); fall back to patchright (stealth Chromium).
    discovered = False
    try:
        from cardiag.ingest.tiktok import discover_camoufox
        discover_camoufox.run(queries, target=20, headless=True)
        discovered = True
    except Exception as e:
        print(f"  camoufox discovery unavailable ({type(e).__name__}: {e}); "
              f"trying patchright")
    if not discovered:
        import asyncio
        try:
            from cardiag.ingest.tiktok import discover
            asyncio.run(discover.run(queries, target=20, headed=False))
        except Exception as e:
            raise SystemExit(f"tiktok discovery failed (need a stealth browser: "
                             f"`python -m camoufox fetch` or patchright): {e}")

    wl = paths.TT_DATA / "worklist.jsonl"
    if not wl.exists():
        raise SystemExit("tiktok discovery produced no worklist "
                         "(browser missing or anti-bot block).")
    work = [json.loads(ln) for ln in open(wl)][:max_videos]

    clap, recs = _clap(), []
    tmp = paths.TT_DATA / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    for i, w in enumerate(work):
        vid = w["id"]
        url = w.get("url") or f"https://www.tiktok.com/@{w.get('author','x')}/video/{vid}"
        mp4, wav = tmp / f"{vid}.mp4", tmp / f"{vid}.wav"
        try:
            subprocess.run(["yt-dlp", "--no-warnings", "-f", "b", "-o", str(mp4), "--", url],
                           check=True, capture_output=True, timeout=180)
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i",
                            str(mp4), "-ar", str(config.SR_CLAP), "-ac", "1",
                            str(wav), "-y"], check=True, timeout=120)
        except Exception:
            continue
        finally:
            mp4.unlink(missing_ok=True)
        recs += _label_audio(wav, vid, w.get("desc", ""), kind, paths.TT_DATA, clap)
        wav.unlink(missing_ok=True)
        print(f"  [tiktok {kind} {i+1}/{len(work)}] clips: {len(recs)}", flush=True)
    n = _write_corpus(recs, paths.TT_DATA)
    print(f"tiktok ({kind}): {n} labeled clips -> {paths.TT_DATA/'corpus.jsonl'}")
    return n


def scrape(platform: str, **kw) -> int:
    return {"youtube": scrape_youtube, "reddit": scrape_reddit,
            "tiktok": scrape_tiktok}[platform](**kw)


# ==================================================================== corpus
def load_corpus() -> list[dict]:
    """Every labeled clip across platforms that still has its wav on disk.

    Tolerant of a corrupt corpus: a malformed JSONL line or a row missing its
    required keys is skipped with a warning, never crashes the whole train.
    """
    rows: list[dict] = []
    skipped = 0
    for base in (paths.YT_DATA, paths.TT_DATA, paths.REDDIT_DATA):
        f = base / "corpus.jsonl"
        if not f.exists():
            continue
        for n, line in enumerate(open(f), 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not (r.get("wav") and r.get("clip_id")):   # required keys
                skipped += 1
                continue
            if Path(r["wav"]).exists():
                rows.append(r)
    if skipped:
        print(f"  load_corpus: skipped {skipped} malformed/incomplete row(s)")
    return rows


# ---------------------------------------------------------------- label helpers
def _cause_of(row: dict):
    """Canonical cause family (underscored) from the title keyword candidates."""
    for part in row.get("l2_candidates", []):
        c = causes.canonical_cause(part)
        if not c or c == "other":
            c = part if part in config.L2_KEYWORDS else None
        if c:
            return c.replace(" ", "_")
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


# ===================================================================== train
def _new_head():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=3000, class_weight="balanced",
                                            random_state=0))


def _source_of(r: dict) -> str:
    """Recording source (youtube/tiktok/reddit) from a clip's wav path."""
    w = r.get("wav", "") or ""
    for s in ("youtube", "tiktok", "reddit"):
        if f"/{s}/" in w:
            return s
    return r.get("source", "?")


def _self_confidence(X, y, groups):
    """Out-of-fold P(observed label) per clip: the confident-learning signal
    (Northcutt et al. 2021). A clip whose audio the model confidently assigns to a
    *different* class than its (weak, keyword-derived) label is a likely mislabel."""
    from sklearn.model_selection import StratifiedGroupKFold
    y = np.asarray(y)
    groups = np.asarray(groups)
    cls = sorted(set(y))
    ns = max(2, min(4, min(len(set(groups[y == c])) for c in cls)))
    sp = np.full(len(y), 1.0)
    for tr, te in StratifiedGroupKFold(ns, shuffle=True, random_state=0).split(X, y, groups):
        if len(set(y[tr])) < 2:
            continue
        clf = _new_head().fit(X[tr], y[tr])
        P = clf.predict_proba(X[te])
        C = list(clf.classes_)
        for j, i in enumerate(te):
            sp[i] = P[j][C.index(y[i])] if y[i] in C else 0.0
    return sp


def _prune_keep(X, y, groups, sources, frac: float):
    """Boolean keep-mask dropping the lowest-self-confidence `frac` of clips WITHIN
    each source (per the literature: global pruning would just re-discover the
    YouTube-vs-rest source split, not real mislabels). Never drops so aggressively
    that a class disappears."""
    if frac <= 0:
        return np.ones(len(y), bool)
    y = np.asarray(y)
    sources = np.asarray(sources)
    sp = _self_confidence(X, y, groups)
    keep = np.ones(len(y), bool)
    for s in set(sources):
        idx = np.where(sources == s)[0]
        k = int(len(idx) * frac)
        if k > 0:
            keep[idx[np.argsort(sp[idx])[:k]]] = False
    if len(set(y[keep])) < len(set(y)):       # don't let a class vanish
        return np.ones(len(y), bool)
    return keep


def _cv_report(X, y, groups, sources=None, prune_frac: float = 0.0,
               n_splits: int = 5, repeats: int = 5) -> dict:
    """Honest by-video performance: repeated StratifiedGroupKFold balanced accuracy
    (mean±std), not a single arbitrary split. Balanced accuracy because the corpus
    is class-skewed (raw accuracy vs majority misleads; see docs/MODEL_CARD.md).
    With ``prune_frac`` the confident-learning prune is applied WITHIN each train
    fold only (test stays untouched), so the estimate reflects the shipped pipeline."""
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedGroupKFold
    y = np.asarray(y)
    groups = np.asarray(groups)
    cls = sorted(set(y))
    ns = max(2, min(n_splits, min(len(set(groups[y == c])) for c in cls)))
    accs = []
    for r in range(repeats):
        sgkf = StratifiedGroupKFold(n_splits=ns, shuffle=True, random_state=r)
        for tr, te in sgkf.split(X, y, groups):
            if len(set(y[tr])) < 2:
                continue
            if prune_frac > 0 and sources is not None:
                k = _prune_keep(X[tr], y[tr], groups[tr], sources[tr], prune_frac)
                if len(set(y[tr][k])) >= 2:
                    tr = tr[k]
            clf = _new_head().fit(X[tr], y[tr])
            accs.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    maj = max(Counter(y).values()) / len(y)
    if not accs:                              # no usable fold (too few videos/classes)
        return {"cv_bal_acc": None, "cv_bal_acc_std": None, "cv_folds": 0,
                "majority_acc": round(float(maj), 3), "n_videos": len(set(groups)),
                "cv_unreliable": "too few videos/classes for a grouped CV estimate"}
    out = {"cv_bal_acc": round(float(np.mean(accs)), 3),
           "cv_bal_acc_std": round(float(np.std(accs)), 3),
           "cv_folds": len(accs), "majority_acc": round(float(maj), 3),
           "n_videos": len(set(groups))}
    if len(set(y)) == 2 and out["cv_bal_acc"] - 0.5 < 2 * out["cv_bal_acc_std"]:
        out["weak_signal"] = ("balanced accuracy is within ~2σ of chance (0.5) — "
                              "this head carries little signal; needs more/cleaner data")
    return out


def _fit_temperature(X, y, groups) -> float:
    """Fit a single temperature (Guo et al. 2017) on out-of-fold logits so the
    head's probabilities stop being over-confident. Binary heads only; preserves
    the decision boundary (argmax unchanged), only rescales confidence. Returns
    T (1.0 = no change). Critically de-overconfidences weak heads so diagnose()'s
    'fault p=0.9' means what it says."""
    from scipy.optimize import minimize_scalar
    from sklearn.model_selection import StratifiedGroupKFold
    y = np.asarray(y)
    groups = np.asarray(groups)
    cls = sorted(set(y))
    if len(cls) != 2:
        return 1.0
    pos = cls[-1]
    ns = max(2, min(5, min(len(set(groups[y == c])) for c in cls)))
    logits, gold = [], []
    for tr, te in StratifiedGroupKFold(ns, shuffle=True, random_state=0).split(X, y, groups):
        if len(set(y[tr])) < 2:
            continue
        clf = _new_head().fit(X[tr], y[tr])
        logits.append(clf.decision_function(X[te]))
        gold.append((y[te] == pos).astype(float))
    if not logits:
        return 1.0
    L = np.concatenate(logits)
    G = np.concatenate(gold)
    sig = lambda z: 1 / (1 + np.exp(-z))

    def nll(T):
        p = sig(L / T)
        return -np.mean(G * np.log(p + 1e-9) + (1 - G) * np.log(1 - p + 1e-9))
    return round(float(minimize_scalar(nll, bounds=(0.3, 10.0), method="bounded").x), 3)


def _fit(rows, labelf, embed, min_class: int, prune_noisy: float = 0.0):
    """Fit a linear head with an honest by-video CV report + temperature scaling.

    Measures with repeated grouped CV, then refits the shipped model on ALL the
    data (no held-out data is wasted), and fits a calibration temperature. With
    ``prune_noisy`` (0..1) the confident-learning prune drops that fraction of the
    lowest-self-confidence (likely-mislabeled) clips per source before the final
    fit, and the CV report reflects it. Falls back to a constant
    ``DummyClassifier`` when there isn't enough data for two classes, so the head
    always exists. Returns (clf, report, temperature)."""
    from sklearn.dummy import DummyClassifier

    def _real(lbl):                          # drop None / placeholder labels as a class
        return lbl is not None and str(lbl).strip().lower() not in ("", "none", "nan", "unknown")

    dim = len(next(iter(embed.values()))) if embed else 512
    data = [(embed[r["clip_id"]], labelf(r), r.get("video", r["clip_id"]), _source_of(r))
            for r in rows if _real(labelf(r)) and r["clip_id"] in embed]
    counts = Counter(lbl for _, lbl, _, _ in data)
    keep = {c for c, n in counts.items() if n >= min_class}
    data = [(x, lbl, g, s) for x, lbl, g, s in data if lbl in keep]
    labels = {lbl for _, lbl, _, _ in data}

    if len(labels) < 2:                     # not enough to learn anything
        X = np.array([x for x, _, _, _ in data]) if data else np.zeros((1, dim))
        y = [lbl for _, lbl, _, _ in data] or [next(iter(labels), "unknown")]
        return DummyClassifier(strategy="prior").fit(X, y), {
            "degenerate": True, "classes": sorted(labels), "n": len(data)}, 1.0

    X = np.array([x for x, _, _, _ in data])
    if X.ndim != 2 or not np.isfinite(X).all():
        raise SystemExit("embeddings are non-finite or ragged — check the CLAP embed step")
    y = np.array([lbl for _, lbl, _, _ in data])
    groups = np.array([g for _, _, g, _ in data])
    sources = np.array([s for _, _, _, s in data])
    report = {"classes": sorted(labels), "n": len(data), "degenerate": False,
              **_cv_report(X, y, groups, sources, prune_noisy)}
    temperature = _fit_temperature(X, y, groups)
    report["temperature"] = temperature
    keep_mask = _prune_keep(X, y, groups, sources, prune_noisy)
    if prune_noisy > 0:
        report["pruned_noisy"] = {"frac": prune_noisy, "dropped": int((~keep_mask).sum())}
    clf = _new_head().fit(X[keep_mask], y[keep_mask])   # ship a model on the kept data
    return clf, report, temperature


def _train_heads(rows, embed, min_class: int, cause_fn, prune_noisy: float = 0.0) -> dict:
    """Fit the three heads + triage from rows and a clip_id->embedding map, and
    save the model artifacts. Shared by the CLAP path and the offline fixtures
    path. ``cause_fn(row)`` yields the cause label. ``prune_noisy`` (0..1) enables
    confident-learning label pruning per head."""
    import joblib

    def _dump(obj, dest):                    # atomic: temp + os.replace
        tmp = Path(str(dest) + ".tmp")
        joblib.dump(obj, tmp)
        os.replace(tmp, dest)

    heads, report, temps = {}, {}, {}
    heads["kind"], report["kind"], temps["kind"] = _fit(
        [r for r in rows if r.get("kind") in ("fault", "normal")],
        lambda r: r.get("kind"), embed, min_class, prune_noisy)
    heads["knock"], report["knock"], temps["knock"] = _fit(
        rows, _knock_of, embed, min_class, prune_noisy)
    heads["cause"], report["cause"], temps["cause"] = _fit(
        [r for r in rows if r.get("kind") == "fault"], cause_fn, embed, min_class, prune_noisy)
    # "where in the car" region head (6 zones): the OOS-robust headline output.
    # Derived from the SAME cause_fn as the cause head (so it works for scraped
    # rows and for explicit-cause rows alike), then mapped to a coarse zone.
    def region_label(r):
        return _CAUSE_TO_REGION.get(cause_fn(r)) if r.get("kind") == "fault" else None
    heads["region"], report["region"], temps["region"] = _fit(
        [r for r in rows if r.get("kind") == "fault"], region_label, embed, min_class, prune_noisy)
    # KNOCK SPECIALIST (a coarse-to-fine cascade, à la hierarchical fault diagnosis):
    # "knock" is one acoustic label worn by ~24 different causes (suspension, rod
    # knock, wheel bearing, CV…). A region head trained ONLY on knock-sound clips
    # localizes the knock 1.8x better than the general head (measured: top-1 0.44 vs
    # 0.33), because it doesn't have to also separate non-knock sounds. diagnose()
    # SOFT-routes to it by the knock probability (gating, not a hard gate, so a
    # shaky Stage-1 detector degrades gracefully to the general region head).
    def _is_knock(r):
        return r.get("kind") == "fault" and "knock" in (r.get("l1") or "").lower()
    heads["knock_region"], report["knock_region"], temps["knock_region"] = _fit(
        [r for r in rows if _is_knock(r)], region_label, embed, min_class, prune_noisy)

    # refuse to ship a model where EVERY head is a constant (e.g. --min-class too
    # high, or a single-class corpus): that would be a silent garbage model.
    if all(report[h].get("degenerate") for h in ("kind", "knock", "cause")):
        raise SystemExit(
            "every head is degenerate — the corpus has too few clips per class "
            f"(min_class={min_class}). Scrape more (both fault and normal), or "
            "lower --min-class. No model was written.")
    if {"fault", "normal"} - set(report["kind"].get("classes", [])):
        print("  note: only one kind class present; the fault/normal head is a "
              "constant and diagnose() will return UNCERTAIN. Scrape YouTube (it "
              "runs normal queries) for both classes.")

    paths.TRAIN_DATA.mkdir(parents=True, exist_ok=True)
    _dump({"heads": heads, "emb": "clap", "temps": temps,
           "degenerate": {h: report[h].get("degenerate", False)
                          for h in ("kind", "knock", "cause", "region", "knock_region")}},
          paths.MODEL_CLAP)

    def triage_label(r):
        if r.get("kind") != "fault":
            return None
        c = cause_fn(r)
        return "engine" if c in _ENGINE else "chassis" if c in _CHASSIS else None

    triage_clf, report["triage"], triage_temp = _fit(
        rows, triage_label, embed, min_class, prune_noisy)
    classes = list(getattr(triage_clf, "classes_",
                           report["triage"].get("classes", ["engine", "chassis"])))
    _dump({"model": triage_clf, "classes": classes, "temperature": triage_temp},
          paths.MODEL_TRIAGE)

    (paths.TRAIN_DATA / "train_report.json").write_text(json.dumps(report, indent=2))
    print("\n=== training report ===")
    print(json.dumps(report, indent=2))
    print(f"\nsaved model -> {paths.MODEL_CLAP}")
    print(f"saved triage -> {paths.MODEL_TRIAGE}")
    return report


def train(min_class: int = 2, prune_noisy: float = 0.0) -> dict:
    """Embed every scraped clip with CLAP and train the heads + triage model.

    ``prune_noisy`` (0..1) enables confident-learning label pruning: drop that
    fraction of the lowest-self-confidence (likely-mislabeled) clips per source
    before fitting. Measured ~+0.05 balanced accuracy on the fault/triage heads on
    the reference corpus (see docs/MODEL_CARD.md); 0.15 is a reasonable value."""
    import librosa

    rows = load_corpus()
    if len(rows) < 8:
        raise SystemExit(
            f"only {len(rows)} clips in the corpus — run `cardiag scrape …` first "
            f"(try a larger --per-query / --max-videos).")

    print(f"embedding {len(rows)} clips with CLAP…", flush=True)
    # Each corpus clip is an isolated span, embedded via the SAME embed_clip()
    # inference uses: train/serve share the contract. A span longer than the CLAP
    # window is split into <=10 s windows (window_spans, kept per the A/B test): each
    # window becomes its own training row with the SAME label and video group, so
    # windows of one clip never split across a CV fold (no leakage). Inference pools
    # the matching per-window vectors in probability space.
    from cardiag.audio.embed import embed_clip, window_spans
    embed: dict[str, np.ndarray] = {}
    expanded: list[dict] = []
    for r in _progress(rows, "embedding clips"):
        y, _ = librosa.load(r["wav"], sr=config.SR_CLAP, mono=True)
        if len(y) < config.SR_CLAP // 2:
            continue
        wins = window_spans(y)
        for k, w in enumerate(wins):
            cid = r["clip_id"] if len(wins) == 1 else f'{r["clip_id"]}#w{k}'
            embed[cid] = embed_clip(w)
            expanded.append({**r, "clip_id": cid})
    return _train_heads(expanded, embed, min_class, _cause_of, prune_noisy)


FIXTURES = Path(__file__).resolve().parent.parent / "_fixtures"


def train_from_fixtures(min_class: int = 2) -> dict:
    """Train OFFLINE on the bundled fixture embeddings: no scrape, no network,
    no CLAP download. Lets a fresh clone produce a model in seconds to learn the
    flow before running the real scrape."""
    npz = FIXTURES / "embeddings.npz"
    if not npz.exists():
        raise SystemExit(f"no fixture embeddings at {npz} (rebuild with "
                         f"scripts/make_fixtures.py).")
    z = np.load(npz, allow_pickle=False)        # plain float/str arrays; no pickle
    def _clean(s):                              # "", "None", "nan" -> "" (= missing)
        s = str(s)
        return "" if s.lower() in ("none", "nan") else s
    rows = [{"clip_id": str(c), "video": str(v), "kind": _clean(k),
             "l1": _clean(l1v), "cause": _clean(ca)} for c, v, k, l1v, ca in
            zip(z["clip_id"], z["video"], z["kind"], z["l1"], z["cause"])]
    embed = {str(c): x for c, x in zip(z["clip_id"], z["X"])}
    print(f"training offline on {len(rows)} bundled fixture embeddings…")
    return _train_heads(rows, embed, min_class, lambda r: r.get("cause") or None)


# ===================================================================== demo
def demo(per_query: int = 1, max_videos: int = 12) -> None:
    """The whole loop from nothing, across all three sources, in one command."""
    print("STEP 1/3  scrape + clean (YouTube + Reddit + TikTok)\n" + "-" * 48)
    results = {}
    for name, fn in (("youtube", lambda: scrape_youtube(per_query, max_videos)),
                     ("reddit", lambda: scrape_reddit(2, max_videos)),
                     ("tiktok", lambda: scrape_tiktok(max_videos))):
        try:
            results[name] = fn()
        except Exception as e:                        # one source must not kill it
            print(f"  [{name}] skipped: {type(e).__name__}: {e}")
            results[name] = 0
    print(f"\nscraped: {results}")
    if sum(results.values()) < 8:
        raise SystemExit("too few clips scraped; check your network and retry.")

    print("\nSTEP 2/3  embed + train\n" + "-" * 48)
    train()

    print("\nSTEP 3/3  inference\n" + "-" * 48)
    from cardiag import Classifier
    clf = Classifier.load()
    print(json.dumps(clf.diagnose(load_corpus()[0]["wav"]).to_dict(), indent=1))
    print("\n✓ loop complete: scraped (3 sources), cleaned, trained, diagnosed "
          "from scratch.")
