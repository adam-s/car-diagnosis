"""End-to-end build: scrape -> clean -> label -> embed -> train -> model.

This is the loop a fresh clone runs, across all three sources (YouTube, Reddit,
TikTok). It is deliberately self-contained: it needs only ``yt-dlp`` + CLAP
(downloaded on first use) — **no LLM and no external datasets**. TikTok adds a
stealth browser (patchright) for discovery. Labels come from what is knowable at
scrape time:

  * ``kind`` (fault / normal) — YouTube's fault vs normal query sets; Reddit and
    TikTok are fault-dominant sources, so they contribute the fault class.
  * ``l1`` sound-type and mechanical/tool gating — CLAP zero-shot.
  * ``cause`` (part family) — keyword match on the title / caption.

Every platform funnels through one labeling function (:func:`_label_audio`) so a
clip looks identical regardless of source. That weaker-but-honest supervision
trains a real, working model from nothing — it teaches the loop by running it.

    cardiag scrape youtube|reddit|tiktok   # -> data/<platform>/corpus.jsonl
    cardiag train                          # corpus -> CLAP -> heads -> model
    cardiag diagnose clip.wav
    cardiag demo                           # all three, small, in one command
"""
from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from cardiag import config, paths
from cardiag.training.prep import causes

# canonical cause family (underscored) -> coarse triage class
_ENGINE = {"engine_internal", "low_oil", "fuel_ignition", "belt", "accessories",
           "alternator", "water_pump", "turbo", "exhaust", "ac_compressor",
           "fuel_pump"}
_CHASSIS = {"wheel_bearing", "brakes", "cv_joint", "suspension", "differential",
            "tires", "power_steering"}


# ============================================================ unified labeling
def _clap():
    from cardiag.audio.clap import Clap
    return Clap()


def _label_audio(wav, vid: str, title: str, kind: str, out_base: Path, clap) -> list[dict]:
    """Cascade + CLAP-gate one local audio file into labeled clip records.

    This is the YouTube corpus logic, reused for every platform: isolate the
    non-speech mechanical spans, score them with CLAP (mechanical-confirm,
    fault-vs-tool, L1 sound-type), keep the survivors, and write clip WAVs. Each
    record carries an absolute ``wav`` path plus the labels ``train`` consumes.
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
    l2 = l2_from_text(title)
    shop = speech_frac >= config.SHOP_SPEECH_FRAC
    recs = []
    for i, (cp, fp, lp) in enumerate(zip(conf, fault, l1p)):
        l1, cf, margin, status, _ = gate(cp, fp, lp, shop_context=shop)
        if status == "reject":
            continue
        f = out_dir / f"clip_{i:02d}.wav"
        sf.write(f, clips[i], sr)
        recs.append({"clip_id": f"{vid}_{i:02d}", "video": vid, "wav": str(f),
                     "kind": kind, "l1": l1, "l1_conf": round(cf, 3),
                     "status": status, "l2_candidates": l2, "title": title})
    if not recs:
        try:
            out_dir.rmdir()
        except OSError:
            pass
    return recs


def _write_corpus(records, out_base: Path) -> int:
    out_base.mkdir(parents=True, exist_ok=True)
    with open(out_base / "corpus.jsonl", "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return len(records)


# ==================================================================== scrapers
def scrape_youtube(per_query: int = 3, max_videos: int = 40) -> int:
    """Discover fault+normal videos, download/clean each, write corpus.jsonl."""
    import random

    from cardiag.ingest.youtube import discover
    from cardiag.ingest.youtube.pipeline import acquire

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

    paths.ensure_data_dirs()
    reddit.main(pages)                                # -> posts.jsonl + audio/*.wav
    posts_f = paths.REDDIT_DATA / "posts.jsonl"
    if not posts_f.exists():
        print("reddit: no posts scraped")
        return 0
    posts = [json.loads(l) for l in open(posts_f)][:max_posts]
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


def scrape_tiktok(max_videos: int = 30, n_queries: int = 8) -> int:
    """Discover fault clips via the stealth browser, download + label each.

    Needs the stealth browser: `pip install -e .[scrape]` then
    `patchright install chromium`. TikTok anti-bot may block headless runs.
    """
    from cardiag.ingest.tiktok.discover import PROBLEM_QUERIES

    paths.ensure_data_dirs()
    queries = PROBLEM_QUERIES[:n_queries]

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
    work = [json.loads(l) for l in open(wl)][:max_videos]

    clap, recs = _clap(), []
    tmp = paths.TT_DATA / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    for i, w in enumerate(work):
        vid = w["id"]
        url = w.get("url") or f"https://www.tiktok.com/@{w.get('author','x')}/video/{vid}"
        mp4, wav = tmp / f"{vid}.mp4", tmp / f"{vid}.wav"
        try:
            subprocess.run(["yt-dlp", "--no-warnings", "-f", "b", "-o", str(mp4), url],
                           check=True, capture_output=True, timeout=180)
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i",
                            str(mp4), "-ar", str(config.SR_CLAP), "-ac", "1",
                            str(wav), "-y"], check=True, timeout=120)
        except Exception:
            continue
        finally:
            mp4.unlink(missing_ok=True)
        recs += _label_audio(wav, vid, w.get("desc", ""), "fault", paths.TT_DATA, clap)
        wav.unlink(missing_ok=True)
        print(f"  [tiktok {i+1}/{len(work)}] clips: {len(recs)}", flush=True)
    n = _write_corpus(recs, paths.TT_DATA)
    print(f"tiktok: {n} labeled clips -> {paths.TT_DATA/'corpus.jsonl'}")
    return n


def scrape(platform: str, **kw) -> int:
    return {"youtube": scrape_youtube, "reddit": scrape_reddit,
            "tiktok": scrape_tiktok}[platform](**kw)


# ==================================================================== corpus
def load_corpus() -> list[dict]:
    """Every labeled clip across platforms that still has its wav on disk."""
    rows: list[dict] = []
    for base in (paths.YT_DATA, paths.TT_DATA, paths.REDDIT_DATA):
        f = base / "corpus.jsonl"
        if not f.exists():
            continue
        for line in open(f):
            r = json.loads(line)
            if r.get("wav") and Path(r["wav"]).exists():
                rows.append(r)
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
def _fit(rows, labelf, embed, min_class: int):
    """Fit a calibrated linear head with a leakage-safe (group-by-video) report.

    Falls back to a constant ``DummyClassifier`` when there isn't enough data for
    two classes, so the head always exists.
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
        X = np.array([x for x, _, _ in data]) if data else np.zeros((1, 512))
        y = [lbl for _, lbl, _ in data] or [next(iter(labels), "unknown")]
        return DummyClassifier(strategy="prior").fit(X, y), {
            "degenerate": True, "classes": sorted(labels), "n": len(data)}

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

    rows = load_corpus()
    if len(rows) < 8:
        raise SystemExit(
            f"only {len(rows)} clips in the corpus — run `cardiag scrape …` first "
            f"(try a larger --per-query / --max-videos).")

    print(f"embedding {len(rows)} clips with CLAP…", flush=True)
    clap = _clap()
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
        print("  note: only one kind class scraped; the fault/normal head is a "
              "constant. Scrape YouTube (it runs normal queries) for both classes.")

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
