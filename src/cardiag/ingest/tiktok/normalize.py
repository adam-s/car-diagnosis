"""Normalize OCR cause-claims into a canonical taxonomy AND cross-check them
against the CLAP sound-type: the two-independent-modalities gold gate.

For each labeled bite Haiku receives (raw OCR text, CLAP sound-type) and returns:
  - canonical_part : OCR text mapped to a clean part name (or null if not a part)
  - consistent     : does that part plausibly make that sound? (text vs audio)

Tiers:
  gold   = canonical_part set AND consistent  (OCR cause + acoustics agree)
  silver = canonical_part set, not consistent (claim present, audio disagrees -> review)
  bronze = no OCR, CLAP sound only            (sound-typed, uncaused)

    python -m cardiag.ingest.tiktok.normalize
"""
import json
import subprocess
from collections import Counter

from cardiag import paths

DATA = paths.TT_DATA
HAIKU = "claude-haiku-4-5"


def haiku_map(pairs, batch=25):
    """pairs: list of {i, ocr, desc, sound}. Returns {i: (canonical_part|None, consistent)}."""
    out = {}
    for k in range(0, len(pairs), batch):
        chunk = pairs[k:k + batch]
        prompt = (
            "Each item has OCR text from a video overlay, the video's description, and "
            "the sound-type our audio model gave that clip. For each: (1) map the OCR "
            "text to a canonical AUTOMOTIVE part name (lowercase, e.g. 'wheel bearing', "
            "'cv joint', 'serpentine belt') — but return null if it is NOT a car part, "
            "or if the description shows the video is not about a car (e.g. an appliance, "
            "power tool, or product ad); (2) judge if that part plausibly produces that "
            "sound-type (consistent: true/false). Reply ONLY a JSON array of "
            '{"i":int,"part":str|null,"consistent":bool}.\n' + json.dumps(chunk))
        try:
            res = subprocess.run(["claude", "-p", "--model", HAIKU, prompt],
                                 capture_output=True, text=True, timeout=150).stdout
            for o in json.loads(res[res.index("["):res.rindex("]") + 1]):
                out[o["i"]] = (o.get("part"), bool(o.get("consistent")))
        except (ValueError, json.JSONDecodeError, subprocess.TimeoutExpired):
            pass
        print(f"  normalized {min(k+batch, len(pairs))}/{len(pairs)}", flush=True)
    return out


GOLD_MARGIN_FLOOR = 0.10   # gold needs a confident SOUND anchor, not a coin-flip


def main():
    recs = [json.loads(l) for l in open(paths.TT_DATA / "corpus.jsonl")]
    labeled = [r for r in recs if r.get("ocr_label")]
    pairs = [{"i": i, "ocr": r["ocr_label"], "desc": (r.get("desc") or "")[:120],
              "sound": r["l1"]} for i, r in enumerate(labeled)]
    print(f"{len(recs)} bites, {len(labeled)} with OCR -> normalizing")
    mapping = haiku_map(pairs)

    tiers = Counter()
    for i, r in enumerate(labeled):
        part, consistent = mapping.get(i, (None, False))
        r["canonical_part"] = part
        r["cross_modal_consistent"] = consistent
        # gold = part named, sound-consistent, AND a confident sound anchor
        confident = r.get("l1_margin", 0) >= GOLD_MARGIN_FLOOR
        r["tier"] = ("gold" if (part and consistent and confident)
                     else ("silver" if part else "bronze"))
    for r in recs:
        if not r.get("ocr_label"):
            r["tier"] = "bronze"
        tiers[r.get("tier", "bronze")] += 1

    (paths.TT_DATA / "corpus_labeled.jsonl").write_text(
        "\n".join(json.dumps(r) for r in recs) + "\n")
    print(f"\ntiers: {dict(tiers)}")
    gold = [r for r in recs if r.get("tier") == "gold"]
    print(f"gold parts: {dict(Counter(r['canonical_part'] for r in gold).most_common(20))}")


if __name__ == "__main__":
    main()
