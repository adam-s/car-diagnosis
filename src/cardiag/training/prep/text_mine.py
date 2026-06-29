"""Mine the FULL video text (description + chapters + transcript + comments)
with Haiku to recover SPECIFIC, timestamped fault labels: the signal the
per-clip fusion threw away.

Why this is the right approach: the iteration research (docs/iteration-research.md)
found label quality, not the feature space, is the bottleneck. The corpus
already holds full transcripts (1,116 files), chapters (uploader-authored
timestamped fault labels, 34% of videos), and long descriptions, but fusion.py
only saw a per-clip transcript SLICE + one comment. A mechanic naming "that's a
rod knock, you're low on oil" anywhere in the video was invisible to it.

This extracts, per video, every distinct fault with subtype granularity, an
engine-internal/suspension/driveline/... category, an explicit-vs-inferred flag,
and time ranges (from chapters/transcript) so labels join to clips by timestamp.

Verifier-gated by design (per the overstamp lesson): `explicit` marks labels a
human actually stated; downstream we trust explicit ≫ inferred and cross-check
against audio. Haiku generates candidates; agreement makes them trustworthy.

    uv run training/prep/text_mine.py --videos yt:ID1,yt:ID2   # pilot
    uv run training/prep/text_mine.py --knock-videos --limit 40
"""
import argparse
import json
import os
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from cardiag import paths

DATA = paths.TRAIN_DATA
META = paths.YT_DATA / "meta"
CACHE = DATA / "text_mined.jsonl"
# Cost: default to LOCAL Qwen (ollama, $0) instead of Haiku. Override with
# LLM_BACKEND=claude (haiku) or LLM_BACKEND=ollama (default). 14b for label
# quality (the project's whole lesson is that label quality is the ceiling).
BACKEND = os.environ.get("LLM_BACKEND", "ollama")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b-instruct")
CLAUDE_MODEL = "claude-haiku-4-5"
WORKERS = 4 if BACKEND == "ollama" else 6   # ollama serves one model instance


def call_llm(prompt):
    if BACKEND == "claude":
        return subprocess.run(["claude", "-p", "--model", CLAUDE_MODEL, prompt],
                              capture_output=True, text=True,
                              timeout=240).stdout
    body = json.dumps({"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate",
                                 data=body,
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))["response"]

INSTR = (
    "You are an expert mechanic reading ALL the text from one car-repair video "
    "(title, description, uploader chapter markers, spoken transcript with "
    "timestamps, top comments). Identify every DISTINCT mechanical fault the "
    "video diagnoses or demonstrates. Be SPECIFIC about engine noises — "
    "distinguish rod/bearing knock, piston slap, lifter/valvetrain tick, "
    "low-oil knock, detonation/pre-ignition, timing-chain rattle — when the "
    "text supports it. For each fault return:\n"
    "  part: specific failing component, lowercase (e.g. 'rod bearing', "
    "'lifter', 'low engine oil', 'ball joint', 'cv joint', 'wheel bearing').\n"
    "  category: one of engine_internal | valvetrain | low_oil | "
    "fuel_ignition | suspension | driveline | brakes | exhaust | "
    "accessory_belt | steering | cooling | other.\n"
    "  explicit: true if a person SAYS it or text NAMES it; false if you "
    "inferred it from symptoms.\n"
    "  evidence: a short verbatim quote from the text that supports it.\n"
    "  time_ranges: list of [start_sec,end_sec] the fault is shown/discussed, "
    "from chapter markers or transcript timing; [] if the video never localizes "
    "it in time.\n"
    "Return ONLY a raw JSON object: {\"faults\":[{...}]}. No prose, no fences."
)


def parse_json3(path):
    """YouTube json3 transcript -> [(t_sec, text)]."""
    try:
        d = json.load(open(path))
    except Exception:
        return []
    out = []
    for e in d.get("events", []):
        t = e.get("tStartMs", 0) / 1000.0
        s = "".join(seg.get("utf8", "") for seg in e.get("segs", []))
        if s.strip():
            out.append((t, s.strip()))
    return out


def transcript_for(vid):
    # prefer human ('.en.json3') over auto ('.en-orig'/'en-*'); then any
    d = META / vid
    cands = (sorted(d.glob(f"{vid}.en.json3"))
             or sorted(d.glob(f"{vid}.en-*.json3"))
             or sorted(d.glob(f"{vid}*.json3")))
    if not cands:
        return ""
    segs = parse_json3(cands[0])
    # compress to ~one line per ~12s to keep tokens bounded
    lines, last = [], -99
    for t, s in segs:
        if t - last >= 12:
            lines.append(f"[{int(t)}s] {s}")
            last = t
        else:
            lines[-1] += " " + s if lines else None
    return "\n".join(l for l in lines if l)[:6000]


def video_text(vid):
    ij = META / vid / f"{vid}.info.json"
    if not ij.exists():
        return None
    d = json.load(open(ij))
    chapters = [f"[{int(c.get('start_time',0))}s] {c.get('title','')}"
                for c in (d.get("chapters") or [])]
    comments = [c.get("text", "")[:200]
                for c in (d.get("comments") or [])[:6]]
    return {
        "title": d.get("title", ""),
        "description": (d.get("description") or "")[:1500],
        "chapters": chapters,
        "tags": (d.get("tags") or [])[:25],
        "comments": comments,
        "transcript": transcript_for(vid),
    }


def build_prompt(vid):
    txt = video_text(vid)
    if not txt:
        return None
    bundle = {k: v for k, v in txt.items() if v}
    return INSTR + "\n\nVIDEO TEXT:\n" + json.dumps(bundle)[:14000]


def parse_faults(text):
    obj = json.loads(text[text.index("{"):text.rindex("}") + 1])
    return obj.get("faults", [])


def mine_one(vid):
    prompt = build_prompt(vid)
    if not prompt:
        return vid, None
    try:
        return vid, parse_faults(call_llm(prompt))
    except Exception:
        return vid, None


def knock_videos(limit):
    vids = []
    seen = set()
    for l in open(paths.YT_DATA / "corpus.enriched.tiered.jsonl"):
        r = json.loads(l)
        if (r.get("l1") == "knocking or clunking noise"
                and r.get("fused_kind") == "fault"
                and r.get("tier") in ("gold", "silver")
                and r["video"] not in seen):
            seen.add(r["video"])
            vids.append(r["video"])
    return vids[:limit]


def fault_videos(limit):
    """All videos that contributed >=1 fault clip: the full sub-label harvest.
    Ordered by fault-clip count so the richest videos mine first."""
    cnt = {}
    for l in open(paths.YT_DATA / "corpus.enriched.tiered.jsonl"):
        r = json.loads(l)
        if r.get("fused_kind") == "fault" and (META / r["video"]).exists():
            cnt[r["video"]] = cnt.get(r["video"], 0) + 1
    return [v for v, _ in sorted(cnt.items(), key=lambda kv: -kv[1])][:limit]


def worklist_fault_videos(limit):
    """Newly-scraped fault videos (from the worklist) with meta captured but
    not yet in the enriched corpus: the overnight-scrape mining target."""
    wl = paths.YT_DATA / "worklist.json"
    if not wl.exists():
        return []
    vids = []
    for w in json.load(open(wl)):
        if w.get("kind") == "fault" and (META / w["id"]).exists():
            vids.append(w["id"])
    return vids   # limit is applied AFTER the cache filter in main()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=None, help="comma yt:ID")
    ap.add_argument("--knock-videos", action="store_true")
    ap.add_argument("--all-fault-videos", action="store_true")
    ap.add_argument("--worklist-faults", action="store_true")
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    if args.videos:
        vids = [v.split(":", 1)[-1] for v in args.videos.split(",")]
    elif args.worklist_faults:
        vids = worklist_fault_videos(args.limit)
    elif args.all_fault_videos:
        vids = fault_videos(args.limit)
    else:
        vids = knock_videos(args.limit)

    cached = set()
    if CACHE.exists():
        cached = {json.loads(l)["video"] for l in open(CACHE)}
    vids = [v for v in vids if v not in cached][:args.limit]
    model = CLAUDE_MODEL if BACKEND == "claude" else OLLAMA_MODEL
    print(f"mining {len(vids)} videos with {model} ({WORKERS}-way)", flush=True)

    caf = open(CACHE, "a")
    ok = 0
    if BACKEND == "modal":
        # one GPU batch through Modal (vLLM), far faster than streaming
        from cardiag.pipeline.llm import run_batch
        items = [(v, build_prompt(v)) for v in vids]
        items = [(v, p) for v, p in items if p]
        results = run_batch(items, backend="modal")
        for vid, _ in items:
            try:
                faults = parse_faults(results.get(vid, ""))
            except Exception:
                continue
            caf.write(json.dumps({"video": vid, "platform": "youtube",
                                  "faults": faults}) + "\n")
            ok += 1
        caf.flush()
    else:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for fut in as_completed([ex.submit(mine_one, v) for v in vids]):
                vid, faults = fut.result()
                if faults is None:
                    continue
                caf.write(json.dumps({"video": vid, "platform": "youtube",
                                      "faults": faults}) + "\n")
                caf.flush()
                ok += 1
                if ok % 10 == 0:
                    print(f"  {ok}/{len(vids)} mined", flush=True)
    caf.close()
    print(f"done: {ok} videos mined -> {CACHE}")


if __name__ == "__main__":
    main()
