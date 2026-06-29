"""TikTok labeling pipeline over the discovered worklist.

Per video (yt-dlp mp4, transient):
  cascade -> sound bites
  per bite:  overlay OCR (easyocr, banner-stripped)  = the CAUSE claim (text)
             CLAP L1 sound-type                       = the SOUND (audio)
  cross-check: a bite is GOLD when OCR gives a label AND CLAP's sound is
  consistent with that part's expected sound (two independent modalities).

Writes data/corpus.jsonl. Then haiku_normalize() maps raw OCR strings -> taxonomy.

    python -m cardiag.ingest.tiktok.batch       # process whole worklist
    python -m cardiag.ingest.tiktok.batch 30    # first 30 only
"""
import json
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from ocrmac import ocrmac

from cardiag import config, paths
from cardiag.audio.cascade import candidate_regions, cyclic_features, spectral_fingerprint
from cardiag.audio.clap import Clap

DATA = paths.TT_DATA
L1_PROMPTS = config.L1_PROMPTS
BANNER_FRAC = 0.6


def download(url, vid):
    mp4 = DATA / "tmp" / f"{vid}.mp4"
    if mp4.exists() and mp4.stat().st_size > 0:
        return mp4
    mp4.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(2):   # one retry, recovers transient TikTok throttle
        try:
            subprocess.run(["yt-dlp", "--no-warnings", "-f", "b", "-o", str(mp4), url],
                           check=True, capture_output=True, timeout=180)
            if mp4.exists() and mp4.stat().st_size > 0:
                return mp4
        except subprocess.CalledProcessError:
            if attempt == 0:
                continue
            raise
    return mp4


def frame_at(mp4, t, out):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(t),
                    "-i", str(mp4), "-frames:v", "1", "-q:v", "3",
                    "-vf", "scale=360:-1", str(out), "-y"], check=True)


def ocr_raw(img):
    """Apple Vision OCR (Neural Engine). Returns [(text, conf, height_frac)].
    height_frac = normalized text-box height (bigger = more prominent caption)."""
    out = []
    for txt, conf, box in ocrmac.OCR(str(img), recognition_level="accurate").recognize():
        out.append((txt.strip(), float(conf), float(box[3])))  # box = [x,y,w,h] normalized
    return out


def ocr_consensus(frame_results, banners):
    """Consensus over PRE-OCR'd frames (each = [(text,conf,h),...]). Label = the
    candidate string recurring across the most frames (temporal consensus kills
    transient narration; the persistent part caption wins). Tiebreak by size*conf."""
    from collections import defaultdict
    score = defaultdict(lambda: [0, 0.0, 0.0])   # key -> [frame_count, sum_score, best_conf]
    rep = {}
    for res in frame_results:
        seen = set()
        for t, conf, h in res:
            if conf < 0.4 or len(t) < 3 or any(t.lower() in b or b in t.lower() for b in banners):
                continue
            key = t.lower().replace(" ", "")
            rep.setdefault(key, t)
            if key not in seen:
                score[key][0] += 1; seen.add(key)
            score[key][1] += h * conf
            score[key][2] = max(score[key][2], conf)
    if not score:
        return None, 0.0
    best = max(score.items(), key=lambda kv: (kv[1][0], kv[1][1]))
    return rep[best[0]], round(best[1][2], 2)


def extract_wav(mp4):
    """ffmpeg-extract 48k mono wav (robust where librosa's mp4 backend fails)."""
    wav = mp4.with_suffix(".wav")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(mp4),
                    "-ac", "1", "-ar", "48000", str(wav), "-y"], check=True)
    return wav


def process(url, vid, desc, query, clap):
    mp4 = download(url, vid)
    wav = extract_wav(mp4)
    y48, sr = librosa.load(str(wav), sr=48000, mono=True)
    wav.unlink(missing_ok=True)
    y16 = librosa.resample(y48, orig_sr=48000, target_sr=16000)
    regions = candidate_regions(y16, sr=16000)
    if not regions:
        mp4.unlink(missing_ok=True)
        return []

    fdir = DATA / "frames" / vid
    fdir.mkdir(parents=True, exist_ok=True)
    # 3 frames per bite (35/55/75%, biased toward where reveal captions stabilize)
    bite_frames = []
    for i, (s, e) in enumerate(regions):
        fs = []
        for j, frac in enumerate((0.35, 0.55, 0.75)):
            f = fdir / f"b{i:02d}_{j}.jpg"
            frame_at(mp4, s + (e - s) * frac, f)
            fs.append(f)
        bite_frames.append(fs)

    # OCR every frame ONCE; reuse for both banner detection and bite consensus
    ocr_cache = {f: ocr_raw(f) for fs in bite_frames for f in fs}
    raw = [[t.lower() for t, c, h in ocr_cache[f] if c > 0.4] for f in ocr_cache]
    banners = {t for t, n in Counter(t for ts in raw for t in set(ts)).items()
               if n >= max(2, BANNER_FRAC * len(ocr_cache))}

    mp4.unlink(missing_ok=True)
    clips = [y48[int(s * sr):int(e * sr)] for s, e in regions]
    l1_probs = clap.score(clips, L1_PROMPTS)

    out_dir = DATA / "clips" / vid
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i, ((s, e), fs, lp) in enumerate(zip(regions, bite_frames, l1_probs)):
        label, conf = ocr_consensus([ocr_cache[f] for f in fs], banners)
        o = np.argsort(-lp)
        clip = out_dir / f"b{i:02d}.wav"
        y22 = librosa.resample(clips[i], orig_sr=sr, target_sr=22050)
        per, hz, reg = cyclic_features(y22, 22050)
        sf.write(clip, clips[i], sr)
        for f in fs:
            f.unlink(missing_ok=True)   # frames transient (OCR done)
        records.append({
            "clip_id": f"{vid}_{i:02d}", "video": vid, "platform": "tiktok",
            "start": round(s, 2), "end": round(e, 2), "query": query,
            "ocr_label": label, "ocr_conf": conf,            # CAUSE claim (text)
            "l1": L1_PROMPTS[o[0]][2:], "l1_conf": round(float(lp[o[0]]), 3),  # SOUND (audio)
            "l1_margin": round(float(lp[o[0]] - lp[o[1]]), 3),
            "cyclic": {"periodicity": round(per, 3), "pulse_hz": round(hz, 2), "regularity": round(reg, 3)},
            "spectral": spectral_fingerprint(y22, 22050),
            "file": str(Path("data") / clip.relative_to(paths.TT_DATA)), "desc": desc,
            "provenance": {"label_source": "overlay-ocr(applevision)+clap-soundtype",
                           "banners_stripped": sorted(banners)},
        })
    fdir.rmdir()   # frames all consumed
    return records


def main(limit=0, workers=5):
    wl = [json.loads(l) for l in open(paths.TT_DATA / "worklist.jsonl")]
    if limit:
        wl = wl[:limit]
    # "attempted" = every video we've tried (incl. failures/empties), so restarts
    # never reprocess them, not just the ones that produced bites.
    att = DATA / "attempted.txt"
    attempted = set(att.read_text().split()) if att.exists() else set()
    wl = [w for w in wl if w["id"] not in attempted]
    print(f"processing {len(wl)} TikTok videos on {workers} threads")
    clap = Clap()   # thread-safe (locked MPS); OCR runs on the Neural Engine
    led, stats = DATA / "corpus.jsonl", Counter()

    def work(w):
        return w, process(w["url"], w["id"], w.get("desc", ""), w.get("query", ""), clap)

    # Threads overlap: ffmpeg (subprocess), librosa/cascade (numpy), Apple Vision
    # OCR (Neural Engine) and CLAP (MPS) all release the GIL, so N videos pipeline
    # across CPU cores + both accelerators. CLAP/silero are lock-guarded.
    done_n = 0
    with open(led, "a") as f, open(att, "a") as af, \
            ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, w) for w in wl]
        for fut in as_completed(futs):
            done_n += 1
            try:
                w, recs = fut.result()
                for r in recs:
                    f.write(json.dumps(r) + "\n")
                    stats["bites"] += 1
                    stats["ocr_labeled"] += 1 if r["ocr_label"] else 0
                af.write(w["id"] + "\n")
                stats["ok"] += 1
            except Exception as e:
                stats["failed"] += 1
                print(f"  FAIL {type(e).__name__}: {str(e)[:60]}", flush=True)
            if done_n % 25 == 0:
                f.flush(); af.flush()
                print(f"  --- {done_n}/{len(wl)} | {stats['bites']} bites, "
                      f"{stats['ocr_labeled']} labeled, {stats['failed']} failed ---", flush=True)

    for leftover in (DATA / "tmp").glob("*"):
        leftover.unlink(missing_ok=True)
    fr = DATA / "frames"
    if fr.exists():
        for d in fr.glob("*"):
            for ff in d.glob("*"):
                ff.unlink(missing_ok=True)
            d.rmdir() if d.is_dir() else None
    print(f"\nDONE: {stats['ok']} ok, {stats['failed']} failed | "
          f"{stats['bites']} bites, {stats['ocr_labeled']} OCR-labeled "
          f"({100*stats['ocr_labeled']/max(1,stats['bites']):.0f}%)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
