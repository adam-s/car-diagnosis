"""Extract the canonical fault diagnosis from each Reddit post, batched on Modal.

The post title + OP description + top community comments ARE the diagnosis
(crowd-sourced). This turns that free text into a structured label. Runs the
batch on Modal Qwen (or --backend ollama for local/free): no per-call Haiku.

    python -m cardiag.ingest.reddit.label_reddit --backend modal
    python -m cardiag.ingest.reddit.label_reddit --backend ollama   # local, free

Writes data/reddit/labels.jsonl: {fullname, part, category, confidence, has_sound}
"""
import argparse
import json

from cardiag import paths
from cardiag.pipeline.llm import parse_json, run_batch

POSTS = paths.REDDIT_DATA / "posts.jsonl"
LABELS = paths.REDDIT_DATA / "labels.jsonl"

CATS = ("engine_internal valvetrain low_oil fuel_ignition suspension driveline "
        "brakes exhaust accessory_belt steering cooling electrical other")

INSTR = (
    "You are an expert mechanic. A Reddit user posted a short video of a car "
    "noise/problem. Given the TITLE, the poster's DESCRIPTION, and the top "
    "community COMMENTS (with upvote scores — these are the crowd diagnosis), "
    "identify the single most likely failing part. Be specific about engine "
    "noises (rod/bearing knock, lifter/valvetrain tick, low oil, timing chain). "
    "Return ONLY raw JSON, no prose:\n"
    '{"part": "<specific component lowercase, or null if no clear diagnosis>", '
    f'"category": "<one of: {CATS}>", '
    '"confidence": <0..1, high ONLY if a well-upvoted comment names the part>, '
    '"has_sound": <true if the post is about an audible mechanical noise>}')


def build_prompt(p):
    parts = [f"TITLE: {p.get('title', '')}"]
    if p.get("selftext"):
        parts.append(f"DESCRIPTION: {p['selftext'][:400]}")
    coms = sorted(p.get("comments", []), key=lambda c: -c.get("score", 0))[:6]
    if coms:
        parts.append("COMMENTS:")
        for c in coms:
            parts.append(f"  [{c.get('score', 0)} upvotes] {c['text'][:300]}")
    return INSTR + "\n\n" + "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="modal",
                    choices=["modal", "ollama", "claude"])
    args = ap.parse_args()
    posts = [json.loads(l) for l in open(POSTS)]
    done = set()
    if LABELS.exists():
        done = {json.loads(l)["fullname"] for l in open(LABELS)}
    todo = [p for p in posts if p["fullname"] not in done]
    print(f"{len(todo)} posts to label via {args.backend} "
          f"({len(done)} cached)")
    if not todo:
        return

    items = [(p["fullname"], build_prompt(p)) for p in todo]
    results = run_batch(items, backend=args.backend)

    n_ok = 0
    with open(LABELS, "a") as fh:
        for p in todo:
            obj = parse_json(results.get(p["fullname"], "")) or {}
            rec = {"fullname": p["fullname"], "subreddit": p.get("subreddit"),
                   "part": obj.get("part"), "category": obj.get("category"),
                   "confidence": obj.get("confidence"),
                   "has_sound": obj.get("has_sound"),
                   "file": p["file"]}
            fh.write(json.dumps(rec) + "\n")
            if rec["part"]:
                n_ok += 1
    print(f"labeled {n_ok}/{len(todo)} with a part -> {LABELS}")


if __name__ == "__main__":
    main()
