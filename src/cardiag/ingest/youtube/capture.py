"""Per-video metadata capture (text only, no media download) for the YouTube
corpus. The payoff is CHAPTERS: uploader-authored timestamped fault labels
("Wheel Bearing 378-663s") present on ~40% of our videos, plus timestamped
transcripts, full descriptions, tags, and top comments. These join down to the
clip level (enrich.py) to replace the weak global-title L2 with local labels.

8-way parallel (network-bound), idempotent/resumable: skips ids whose
data/meta/<id>/<id>.info.json already exists. Safe to run alongside the TikTok
batch (different servers; uses the idle CPUs).

    uv run capture.py
"""
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from cardiag import paths

META = paths.YT_DATA / "meta"
WORKERS = 8


def video_ids():
    ids = set()
    for src in (paths.YT_DATA / "corpus.jsonl",):
        p = Path(src)
        if p.exists():
            ids |= {json.loads(l)["video"] for l in open(p)}
    # also any discovered-but-unprocessed
    wl = paths.YT_DATA / "worklist.json"
    if wl.exists():
        ids |= {w["id"] for w in json.loads(wl.read_text())}
    return sorted(ids)


def capture(vid):
    out = META / vid
    if (out / f"{vid}.info.json").exists():
        return vid, "skip"
    out.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["yt-dlp", "--skip-download", "--no-warnings", "--ignore-errors",
         "--write-info-json",
         "--write-subs", "--write-auto-subs", "--sub-langs", "en.*",
         "--sub-format", "json3/vtt",
         "--write-comments",
         "--extractor-args", "youtube:comment_sort=top;max_comments=20,20,0,0",
         "-o", f"{out}/%(id)s.%(ext)s",
         f"https://www.youtube.com/watch?v={vid}"],
        capture_output=True, text=True, timeout=120)
    return vid, ("ok" if (out / f"{vid}.info.json").exists() else "fail")


def main():
    ids = video_ids()
    print(f"capturing metadata for {len(ids)} YouTube videos ({WORKERS}-way)")
    from collections import Counter
    stats, chap = Counter(), 0
    failed = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(capture, v) for v in ids]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                vid, st = f.result()
            except Exception:
                stats["fail"] += 1; continue
            stats[st] += 1
            if st == "fail":
                failed.append(vid)
            if i % 50 == 0:
                print(f"  {i}/{len(ids)} | {dict(stats)}", flush=True)
    if failed:
        (META / "_failed.txt").write_text("\n".join(failed) + "\n")
    # quick chapter coverage tally
    for d in META.glob("*/"):
        ij = d / f"{d.name}.info.json"
        if ij.exists():
            try:
                if json.loads(ij.read_text()).get("chapters"):
                    chap += 1
            except (json.JSONDecodeError, OSError):
                pass
    print(f"\nDONE: {dict(stats)} | videos with chapters: {chap}")


if __name__ == "__main__":
    main()
