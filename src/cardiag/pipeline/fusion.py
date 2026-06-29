"""Per-clip multi-signal cause FUSION: the payoff of all the metadata work.

Every clip carries several weak, independent signals. None is reliable alone;
their AGREEMENT is. This hands Haiku all of them per clip and gets back one
cause + a confidence that IS the agreement:

  audio : l1 sound-type + rhythm (cyclic pulse_hz / periodicity)
  text  : chapter_label, clip_transcript, ocr_label, title/desc, top_comment

-> fused_cause, fused_kind (fault|normal|nonauto), fused_confidence (0-1),
   fused_support (which signals agreed).  High confidence only when audio AND
   text corroborate -> that subset is the trustworthy gold.

Haiku calls run in PARALLEL (API-bound). Works on either platform's ledger.

    uv run pipeline/fusion.py data/youtube/corpus.enriched.jsonl
    uv run pipeline/fusion.py data/tiktok/corpus_labeled.jsonl
"""
import json
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

MODEL = "claude-sonnet-4-6"   # cause inference needs diagnostic reasoning (Haiku declines symptoms)
BATCH = 14
WORKERS = 10

INSTR = (
    "You are an expert mechanic diagnosing a car fault from ONE short audio clip. "
    "Each item gives AUDIO evidence (sound_type from an audio model; rhythm = pulse "
    "in Hz, steady or not) and TEXT evidence (any of: chapter = video section title, "
    "transcript = words spoken during the clip, overlay = on-screen caption, "
    "title/desc, comment = top viewer comment). For each item return:\n"
    "  cause: the single most likely failing automotive part, canonical lowercase "
    "(e.g. 'wheel bearing','cv joint','serpentine belt'). USE DIAGNOSTIC REASONING — "
    "a symptom like 'clicking on turns' implies cv joint, 'growl rising with speed' "
    "implies wheel bearing, even when no part is named. Return null ONLY if there is "
    "genuinely no fault signal (healthy/normal sound, or non-car content).\n"
    "  kind: 'fault' (a problem), 'normal' (healthy engine/exhaust note, enthusiast "
    "rev, cold start), or 'nonauto' (not a car / junk).\n"
    "  confidence: 0..1 — CRITICAL RULES: confidence >= 0.7 REQUIRES corroborating "
    "TEXT (chapter/transcript/overlay/title/comment) that names or strongly implies "
    "the part. The sound_type ALONE is only a weak guess: with no usable text, "
    "confidence MUST be <= 0.45. Treat overlay/title text that is a username, "
    "@handle, #hashtag, channel name, watermark, tool size, or unrelated word "
    "(e.g. 'cartipx','EAGLES','55%','12 mm','auzence') as NO text signal — ignore "
    "it. NEVER output a cause that contradicts clear overlay text naming a part.\n"
    "  support: list of signals that agree, from "
    "['audio','chapter','transcript','overlay','title','comment'].\n"
    "Reply with ONLY the raw JSON array and NOTHING else — no prose, no "
    "reasoning notes, no explanation, no markdown fences before or after. Output "
    "exactly: [{\"i\":int,\"cause\":str|null,\"kind\":str,\"confidence\":number,"
    "\"support\":[str]}]"
)


def bundle(i, r):
    c = r.get("cyclic") or {}
    rhythm = f"{c.get('pulse_hz','?')}Hz periodicity={c.get('periodicity','?')}"
    b = {"i": i, "sound_type": r.get("l1"), "rhythm": rhythm}
    for src, key in [("chapter", "chapter_label"), ("transcript", "clip_transcript"),
                     ("overlay", "ocr_label"), ("comment", None)]:
        v = r.get(key) if key else None
        if v:
            b[src] = (v[:200] if isinstance(v, str) else v)
    tc = r.get("top_comments")
    if tc:
        b["comment"] = tc[0][:160]
    txt = r.get("desc") or r.get("source_text")
    if txt:
        b["title"] = txt[:160]
    return b


def fuse_batch(items):
    prompt = INSTR + "\n" + json.dumps(items)
    try:
        out = subprocess.run(["claude", "-p", "--model", MODEL, prompt],
                             capture_output=True, text=True, timeout=150).stdout
        return {o["i"]: o for o in json.loads(out[out.index("["):out.rindex("]") + 1])}
    except (ValueError, json.JSONDecodeError, subprocess.TimeoutExpired):
        return {}


def main(path):
    recs = [json.loads(l) for l in open(path)]
    cache_path = path.replace(".jsonl", ".fusedcache.jsonl")
    # resume: clip_ids already fused (a 2h run must survive interruption)
    cache = {}
    if __import__("os").path.exists(cache_path):
        for l in open(cache_path):
            try:
                o = json.loads(l); cache[o["clip_id"]] = o
            except json.JSONDecodeError:
                pass

    # batch only the not-yet-fused clips; carry clip_id through for keying
    pending = [(i, recs[i]["clip_id"], bundle(i, recs[i]))
               for i in range(len(recs)) if recs[i]["clip_id"] not in cache]
    batches = [pending[k:k + BATCH] for k in range(0, len(pending), BATCH)]
    print(f"{len(recs)} clips, {len(cache)} cached, fusing {len(pending)} in "
          f"{len(batches)} batches, {WORKERS} parallel {MODEL}", flush=True)

    def run_batch(group):
        res = fuse_batch([b for (_, _, b) in group])
        # remap local-index results back to clip_ids
        return [(cid, res.get(li, {})) for (li, cid, _) in
                [(g[2]["i"], g[1], g) for g in group]]

    done = 0
    caf = open(cache_path, "a")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for fut in as_completed([ex.submit(run_batch, g) for g in batches]):
            for cid, o in fut.result():
                rec = {"clip_id": cid, "cause": o.get("cause"), "kind": o.get("kind"),
                       "confidence": o.get("confidence"), "support": o.get("support") or []}
                cache[cid] = rec
                caf.write(json.dumps(rec) + "\n")
            caf.flush()
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(batches)} batches ({len(cache)}/{len(recs)} clips)", flush=True)
    caf.close()

    nh = 0
    for r in recs:
        o = cache.get(r["clip_id"], {})
        r["fused_cause"] = o.get("cause")
        r["fused_kind"] = o.get("kind")
        r["fused_confidence"] = o.get("confidence")
        r["fused_support"] = o.get("support") or []
        if (o.get("confidence") or 0) >= 0.7 and o.get("cause"):
            nh += 1
    out_path = path.replace(".jsonl", ".fused.jsonl")
    open(out_path, "w").write("\n".join(json.dumps(r) for r in recs) + "\n")

    print(f"\n-> {out_path}")
    print(f"high-confidence (>=0.7) caused clips: {nh} ({100*nh/len(recs):.0f}%)")
    print(f"kind: {dict(Counter(r.get('fused_kind') for r in recs))}")
    print(f"top fused causes: {dict(Counter(r['fused_cause'] for r in recs if r.get('fused_cause')).most_common(15))}")
    sup = Counter(s for r in recs for s in (r.get('fused_support') or []))
    print(f"signal support tallies: {dict(sup)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/youtube/corpus.enriched.jsonl")
