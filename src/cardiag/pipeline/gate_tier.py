"""Code-side trust tiering over a *.fused.jsonl.

Lesson from the Sonnet audit: the model's self-reported confidence is NOT
trustworthy: it stamps 0.8+ on sound-only guesses ("EAGLES"->ball joint 0.85).
So we DERIVE the tier in code from OBJECTIVE text corroboration, not the model's
number. Sonnet supplies the cause; code supplies the trust.

  gold   = fused_cause is a fault AND a VERIFIED-CLEAN text signal corroborates:
             - a chapter_label, OR
             - a content-bearing transcript (not just [Music]), OR
             - a Haiku-verified-real OCR part (canonical_part, null on junk), OR
             - a title/desc that names a part (l2_local / l2_candidates)
  silver = fused_cause + fault, but only the audio sound-type backs it (a guess)
  bronze = no cause, or kind normal/nonauto

This automatically rejects the junk-OCR overconfidence ("cartipx","55%") because
those have no clean text signal.

    uv run pipeline/gate_tier.py data/youtube/corpus.enriched.fused.jsonl
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

BRACKET = re.compile(r"\[[^\]]*\]")


def transcript_content(tx):
    if not tx:
        return False
    words = re.findall(r"[a-zA-Z]{3,}", BRACKET.sub("", tx))
    return len(words) >= 4


def has_clean_text(r):
    if r.get("chapter_label"):
        return True
    if transcript_content(r.get("clip_transcript")):
        return True
    if r.get("canonical_part"):          # Haiku-verified OCR part (null = junk)
        return True
    if r.get("l2_local") or r.get("l2_candidates"):
        return True
    return False


GOLD_CONF = 0.6   # gold needs the model to be confident the text SUPPORTS the cause,
                  # not just that some text exists (audit: "[Music]"/"YELLOW" leaked in)


def tier(r):
    if r.get("music_score", 0) >= 0.5:            # background music = unusable audio
        return "music"
    cause, kind = r.get("fused_cause"), r.get("fused_kind")
    if not cause or kind in ("normal", "nonauto"):
        return "bronze"
    conf = r.get("fused_confidence") or 0
    supports = set(r.get("fused_support") or [])
    text_backed = bool(supports - {"audio"})      # at least one TEXT signal agreed
    if has_clean_text(r) and conf >= GOLD_CONF and text_backed:
        return "gold"
    return "silver"


def main(path):
    recs = [json.loads(l) for l in open(path)]
    # attach music scores if present (sibling of the corpus dir)
    msc = Path(path).parent / "music_scores.json"
    if msc.exists():
        music = json.loads(msc.read_text())
        for r in recs:
            if r["clip_id"] in music:
                r["music_score"] = music[r["clip_id"]]["music"]
    tiers = Counter()
    for r in recs:
        r["tier"] = tier(r)
        tiers[r["tier"]] += 1
    out = path.replace(".fused.jsonl", ".tiered.jsonl")
    open(out, "w").write("\n".join(json.dumps(r) for r in recs) + "\n")

    gold = [r for r in recs if r["tier"] == "gold"]
    print(f"{len(recs)} clips -> {out}")
    print(f"tiers: {dict(tiers)}")
    print(f"gold causes: {dict(Counter(r['fused_cause'] for r in gold).most_common(20))}")
    # show that junk-OCR clips did NOT reach gold
    junk = [r for r in recs if r.get("ocr_label") and not r.get("canonical_part")
            and not r.get("chapter_label") and not transcript_content(r.get("clip_transcript"))]
    print(f"junk-OCR sound-only clips: {len(junk)}, of which gold: "
          f"{sum(1 for r in junk if r['tier']=='gold')} (should be 0)")


if __name__ == "__main__":
    main(sys.argv[1])
