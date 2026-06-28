"""Batch the v2 pipeline over the worklist, using the whole machine:
network (yt-dlp prefetch pool) || CPU (cascade) || GPU (CLAP) run concurrently.

Appends one JSON line per clip to data/corpus.jsonl (the single ledger);
restart-safe (skips videos already in the ledger). Raw wavs are transient.

    uv run batch.py [n_videos]
"""
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from cardiag import paths
from cardiag.audio.clap import Clap

from .pipeline import acquire, process_video

LEDGER = paths.YT_DATA / "corpus.jsonl"
PREFETCH_WORKERS = 6   # parallel downloads; network-bound, doesn't fight CPU/GPU


def load_worklist(n):
    work = json.load(open(paths.YT_DATA / "worklist.json"))
    done = set()
    if LEDGER.exists():
        done = {json.loads(l)["video"] for l in open(LEDGER)}
    todo = [w for w in work if w["id"] not in done]
    return todo[:n] if n else todo


def main(n=0):
    todo = load_worklist(n)
    print(f"{len(todo)} videos to process "
          f"({sum(w['kind']=='fault' for w in todo)} fault, "
          f"{sum(w['kind']=='normal' for w in todo)} normal)\n")
    clap = Clap()
    pool = ThreadPoolExecutor(max_workers=PREFETCH_WORKERS)
    futures = {w["id"]: pool.submit(acquire, w["id"]) for w in todo}

    stats, t0 = Counter(), time.time()
    with open(LEDGER, "a") as ledger:
        for i, w in enumerate(todo, 1):
            vid = w["id"]
            try:
                futures[vid].result(timeout=600)         # wait for prefetch
                recs = process_video(vid, w["title"], clap=clap)
                for r in recs:
                    r["kind"] = w["kind"]                # fault-search vs normal-search
                    ledger.write(json.dumps(r) + "\n")
                    stats[r["status"]] += 1
                ledger.flush()
                stats["videos_ok"] += 1
            except Exception as e:
                stats["videos_failed"] += 1
                print(f"[{i}/{len(todo)}] {vid} FAILED: {type(e).__name__}: {str(e)[:80]}")
            if i % 10 == 0:
                el = time.time() - t0
                print(f"  --- {i}/{len(todo)}, {el/60:.1f}min ({el/i:.1f}s/video) ---")
    pool.shutdown(wait=False, cancel_futures=True)
    for f in (paths.YT_DATA / "tmp").glob("*"):          # clear any prefetch leftovers
        f.unlink(missing_ok=True)

    el = time.time() - t0
    print(f"\n=== DONE: {stats['videos_ok']} ok, {stats['videos_failed']} failed "
          f"in {el/60:.1f}min ===")
    print(f"clips: auto={stats['auto']} review={stats['review']} reject={stats['reject']}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
