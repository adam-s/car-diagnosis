"""Sharded parallel runner over batch.py's per-video pipeline.

The single-process run is latency-bound (~0.5 of 12 cores): serial ffmpeg /
OCR / MPS round-trips, not CPU saturation, so N worker processes over
disjoint worklist shards scale ~linearly. Carefully:

  - downloads stay globally gentle: 1 download thread + window of 4 per
    worker, so 4 workers ~= the single run's 4-way concurrency
  - each worker writes its OWN ledger shard (concurrent appends to
    corpus.jsonl from several processes can interleave partial lines);
    `merge` folds shards into the main ledger afterwards
  - shard assignment strides the RAW worklist (video i -> shard i mod N),
    THEN filters the done-set: stable across restarts, shards stay disjoint
  - done-set = corpus.jsonl + every shard file, so reruns never rework
  - workers never run batch.py's global tmp/ cleanup (other workers' mp4s
    live there); each cleans only its own leftovers

    python -m cardiag.ingest.tiktok.batch_shard worker 0 4   # worker 0 of 4
    python -m cardiag.ingest.tiktok.batch_shard merge        # after all workers exit
"""
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from cardiag import paths
from cardiag.audio.clap import Clap

from .batch import DATA, download, process


def done_videos():
    done = set()
    for led in [DATA / "corpus.jsonl", *DATA.glob("corpus.shard*.jsonl")]:
        if led.exists():
            for line in open(led):
                try:
                    done.add(json.loads(line)["video"])
                except json.JSONDecodeError:
                    pass    # partial last line from a killed worker
    return done


def worker(k, n):
    wl = [json.loads(l) for l in open(paths.TT_DATA / "worklist.jsonl")]
    done = done_videos()
    wl = [w for i, w in enumerate(wl) if i % n == k and w["id"] not in done]
    led = DATA / f"corpus.shard{k}.jsonl"
    print(f"[shard {k}/{n}] {len(wl)} videos -> {led.name}", flush=True)

    clap = Clap()
    stats = Counter()
    WINDOW = 4
    pool = ThreadPoolExecutor(max_workers=1)   # 1/worker keeps global TikTok pressure ~unchanged
    futures = {}
    for w in wl[:WINDOW]:
        futures[w["id"]] = pool.submit(download, w["url"], w["id"])
    with open(led, "a") as f:
        for i, w in enumerate(wl, 1):
            nxt = i - 1 + WINDOW
            if nxt < len(wl):
                nw = wl[nxt]
                futures[nw["id"]] = pool.submit(download, nw["url"], nw["id"])
            try:
                if w["id"] in futures:
                    futures[w["id"]].result(timeout=300)
                recs = process(w["url"], w["id"], w.get("desc", ""), w.get("query", ""), clap)
                for r in recs:
                    f.write(json.dumps(r) + "\n")
                    stats["bites"] += 1
                    if r["ocr_label"]:
                        stats["ocr_labeled"] += 1
                f.flush()
                stats["ok"] += 1
            except Exception as e:
                stats["failed"] += 1
                print(f"  [s{k} {i}] {w['id']} FAIL {type(e).__name__}: {str(e)[:60]}", flush=True)
            if i % 20 == 0:
                print(f"  --- s{k}: {i}/{len(wl)} | {stats['bites']} bites, "
                      f"{stats['ocr_labeled']} OCR-labeled ---", flush=True)
    pool.shutdown(wait=False, cancel_futures=True)
    for vid in futures:                          # own leftovers only, tmp/ is shared
        (DATA / "tmp" / f"{vid}.mp4").unlink(missing_ok=True)
    print(f"[shard {k}] DONE: {stats['ok']} ok, {stats['failed']} failed, "
          f"{stats['bites']} bites ({stats['ocr_labeled']} OCR-labeled)")


def merge():
    led = DATA / "corpus.jsonl"
    seen = set()
    if led.exists():
        for line in open(led):
            try:
                seen.add(json.loads(line)["clip_id"])
            except json.JSONDecodeError:
                pass
    moved = bad = 0
    with open(led, "a") as out:
        for sh in sorted(DATA.glob("corpus.shard*.jsonl")):
            for line in open(sh):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                if r["clip_id"] in seen:
                    continue
                out.write(json.dumps(r) + "\n")
                seen.add(r["clip_id"])
                moved += 1
            sh.unlink()
    for f in (DATA / "tmp").glob("*"):
        f.unlink(missing_ok=True)
    print(f"merged {moved} records into corpus.jsonl ({bad} malformed lines skipped)")


if __name__ == "__main__":
    if sys.argv[1] == "worker":
        worker(int(sys.argv[2]), int(sys.argv[3]))
    elif sys.argv[1] == "merge":
        merge()
