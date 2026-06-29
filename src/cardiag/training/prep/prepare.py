"""Build training manifests from the scraped corpora + external anchor sets.

Selection, cleaning, canonicalization, and leakage-safe splits; clips are
referenced in place (repo-root-relative paths), never moved or copied.

Usage (from repo root):
    uv run training/prep/prepare.py

Outputs (data/training/, gitignored):
    train.jsonl / val.jsonl / test.jsonl   : scrape clips, group-split 80/10/10
    external_eval.jsonl                    : verified-label anchors (test-only)
    anchors_fewshot.jsonl                  : DB1 fine-fault exemplars
    stats.json                             : class/tier/split distributions

Rules baked in (see README.md for rationale):
    - groups: YT creator channel_id (via meta join), TT video id; split is a
      deterministic md5 hash of the group -> no clip from one creator/video
      ever crosses splits.
    - train keeps gold+silver+bronze (bronze flagged via `tier`); val/test
      keep only gold+silver. music tier and missing files are dropped.
    - external sets never enter train.
"""

import glob
import hashlib
import json
import wave
from collections import Counter
from pathlib import Path

from cardiag import paths
from cardiag.training.prep.causes import canonical_cause, canonical_l1

OUT = paths.TRAIN_DATA

SCRAPE = [
    # (platform, corpus path, clip base dir). The corpus path is resolved via
    # paths.resolve_clip(); `base` is the OLD-layout platform prefix that the
    # stored `path` field is built from (resolve_clip understands it later).
    ("youtube", "youtube/data/corpus.enriched.tiered.jsonl", "youtube"),
    ("tiktok", "tiktok/data/corpus_labeled.tiered.jsonl", "tiktok"),
]


def wav_duration(path):
    try:
        with wave.open(str(path)) as w:
            return round(w.getnframes() / w.getframerate(), 3)
    except Exception:
        return None


def yt_creator_map():
    """video id -> channel id, from yt-dlp info.json sidecars."""
    out = {}
    for p in glob.glob(str(paths.YT_DATA / "meta" / "*" / "*.info.json")):
        try:
            d = json.load(open(p))
            out[d["id"]] = (d.get("channel_id") or d.get("uploader_id")
                            or d.get("uploader") or d["id"])
        except Exception:
            pass
    return out


def split_of(group):
    """Deterministic 80/10/10 by group hash, stable across runs."""
    h = int(hashlib.md5(group.encode()).hexdigest(), 16) % 100
    return "train" if h < 80 else ("val" if h < 90 else "test")


def scrape_records():
    creators = yt_creator_map()
    for platform, corpus, base in SCRAPE:
        for line in open(paths.resolve_clip(corpus)):
            r = json.loads(line)
            if not r.get("file") or r.get("tier") == "music":
                continue
            kind = r.get("fused_kind")
            if kind not in ("fault", "normal", "nonauto"):
                continue
            # stored `path` stays in OLD repo-relative form (e.g.
            # "youtube/data/clips/...") so resolve_clip() can map it later.
            path = str(Path(base) / r["file"])
            if not paths.resolve_clip(path).exists():
                continue
            group = (f"yt:{creators.get(r['video'], r['video'])}"
                     if platform == "youtube" else f"tt:{r['video']}")
            split = split_of(group)
            tier = r.get("tier")
            # weak labels stay out of eval splits. Tier grades the *cause*,
            # so it only gates fault clips; normal/nonauto (always bronze by
            # construction) are gated on fusion confidence instead.
            if split != "train":
                if kind == "fault" and tier == "bronze":
                    continue
                if kind != "fault" and (r.get("fused_confidence") or 0) < 0.6:
                    continue
            yield {
                "id": f"{platform[:2]}:{r['clip_id']}",
                "source": platform,
                "path": path,
                "start": r.get("start"),
                "end": r.get("end"),
                "duration": wav_duration(paths.resolve_clip(path)),
                "kind": kind,
                "l1": canonical_l1(r.get("l1")),
                "cause_raw": r.get("fused_cause"),
                "cause": (canonical_cause(r.get("fused_cause"))
                          if kind == "fault" else None),
                "tier": tier,
                "confidence": r.get("fused_confidence"),
                "support": r.get("fused_support"),
                "group": group,
                "split": split,
            }


# ---- external verified-label anchors (never train) -------------------------

# Each external set has TWO roots: paths.* is the absolute new-layout dir we
# actually read files from; the *_REL string is the OLD repo-relative prefix
# that the stored `path` field must keep (resolve_clip() maps it back to disk).
CARDIAG = paths.CARDIAG_DS
CARDIAG_REL = "external-data/kaggle/car-diagnostics-dataset/car diagnostics dataset"
CARDIAG_LABELS = {
    # dir name -> (kind, cause group)
    "normal_engine_startup": ("normal", None),
    "normal_engine_idle": ("normal", None),
    "normal_brakes": ("normal", None),
    "worn_out_brakes": ("fault", "brakes"),
    "serpentine_belt": ("fault", "belt"),
    "power_steering": ("fault", "power_steering"),
    "low_oil": ("fault", "low_oil"),
    "dead_battery": ("fault", "accessories"),
    "bad_ignition": ("fault", "fuel_ignition"),
    # "combined" = multi-fault, ambiguous single label -> skipped
}

AIMECH = paths.AIMECH
AIMECH_REL = "external-data/repos/Ai_Mechanic/ML dataset/Audio"
AIMECH_LABELS = {
    "Normal engine": ("normal", None),
    "Engine Issue": ("fault", None),       # coarse: fault, cause unknown
    # "Non engine issue" excluded: semantics ambiguous (non-engine *car*
    # fault vs non-automotive?): measured mech_confirm 0.936 says these are
    # car sounds, so mapping them to nonauto would poison kind-level eval.
}

DB1 = paths.DB1
DB1_REL = ("external-data/repos/"
           "Sound-Based-Vehicle-Diagnostics-Emergency-Signal-Recognition/"
           "Datasets/DB1")

CARENGINE = paths.CARENGINE
CARENGINE_REL = "external-data/repos/Car-Engine-Sounds-Dataset/_audio"
# filename keyword -> verified L1 sound type (the only external set with
# sound-type ground truth; vehicle make/model also lives in the filename)
CARENGINE_L1 = [
    ("knock", "knock"), ("tick", "tick"), ("click", "tick"),
    ("rattl", "rattle"), ("squeal", "squeal"), ("grind", "grind"),
    ("hiss", "hiss"), ("whine", "whine"), ("hum", "hum"),
]

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4"}


def external_records():
    # car-diagnostics: state/fault dirs of verified clips
    for state_dir in sorted(CARDIAG.iterdir()):
        if not state_dir.is_dir():
            continue
        for fault_dir in sorted(state_dir.iterdir()):
            lab = CARDIAG_LABELS.get(fault_dir.name)
            if lab is None:
                continue  # e.g. "combined"
            kind, cause = lab
            for f in sorted(fault_dir.iterdir()):
                if f.suffix.lower() not in AUDIO_EXT:
                    continue
                yield {
                    "id": f"cardiag:{fault_dir.name}:{f.stem}",
                    "source": "ext:car-diagnostics",
                    "path": f"{CARDIAG_REL}/{f.relative_to(CARDIAG)}",
                    "duration": wav_duration(f),
                    "kind": kind,
                    "cause": cause,
                    "state": state_dir.name.replace(" state", ""),
                    "group": "ext:car-diagnostics",
                    "split": "external_eval",
                }
    # Car-Engine-Sounds: verified normal/abnormal per vehicle; abnormal
    # filenames name the sound (knocking/ticking/...) -> verified L1
    for cls_dir in sorted(CARENGINE.iterdir()):
        if not cls_dir.is_dir():
            continue
        kind = "normal" if cls_dir.name == "normal" else "fault"
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() not in AUDIO_EXT:
                continue
            name = f.stem.lower()
            l1 = next((v for k, v in CARENGINE_L1 if k in name), None)
            yield {
                "id": f"carengine:{cls_dir.name}:{f.stem}",
                "source": "ext:car-engine-sounds",
                "path": f"{CARENGINE_REL}/{f.relative_to(CARENGINE)}",
                "duration": wav_duration(f),
                "kind": kind,
                "cause": None,
                "l1": l1 if kind == "fault" else "normal_idle",
                "group": "ext:car-engine-sounds",
                "split": "external_eval",
            }
    # Ai_Mechanic: coarse 3-class
    for cls_dir in sorted(AIMECH.iterdir()):
        lab = AIMECH_LABELS.get(cls_dir.name)
        if lab is None or not cls_dir.is_dir():
            continue
        kind, cause = lab
        for f in sorted(cls_dir.rglob("*")):
            if f.suffix.lower() not in AUDIO_EXT:
                continue
            yield {
                "id": f"aimech:{cls_dir.name}:{f.stem}",
                "source": "ext:ai-mechanic",
                "path": f"{AIMECH_REL}/{f.relative_to(AIMECH)}",
                "duration": wav_duration(f),
                "kind": kind,
                "cause": cause,
                "group": "ext:ai-mechanic",
                "split": "external_eval",
            }


def db1_records():
    """~27 fine-fault exemplars: few-shot reference anchors, not eval."""
    for fault_dir in sorted(DB1.iterdir()):
        if not fault_dir.is_dir():
            continue
        for f in sorted(fault_dir.iterdir()):
            if f.suffix.lower() not in AUDIO_EXT:
                continue
            yield {
                "id": f"db1:{fault_dir.name}:{f.stem}",
                "source": "ext:db1",
                "path": f"{DB1_REL}/{f.relative_to(DB1)}",
                "kind": "fault",
                "cause": canonical_cause(fault_dir.name),
                "cause_raw": fault_dir.name,
                "group": "ext:db1",
                "split": "anchor",
            }


def main():
    OUT.mkdir(exist_ok=True)
    splits = {"train": [], "val": [], "test": []}
    for rec in scrape_records():
        splits[rec["split"]].append(rec)
    external = list(external_records())
    anchors = list(db1_records())

    for name, recs in [*splits.items(), ("external_eval", external),
                       ("anchors_fewshot", anchors)]:
        with open(OUT / f"{name}.jsonl", "w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")

    stats = {}
    for name, recs in [*splits.items(), ("external_eval", external)]:
        stats[name] = {
            "n": len(recs),
            "kind": dict(Counter(r["kind"] for r in recs)),
            "tier": dict(Counter(r.get("tier") for r in recs)),
            "l1": dict(Counter(r.get("l1") for r in recs).most_common()),
            "cause": dict(Counter(r["cause"] for r in recs
                                  if r.get("cause")).most_common()),
            "groups": len(set(r["group"] for r in recs)),
        }
    stats["anchors_fewshot"] = {"n": len(anchors)}
    json.dump(stats, open(OUT / "stats.json", "w"), indent=1)

    for name in [*splits, "external_eval"]:
        s = stats[name]
        print(f"{name:14} n={s['n']:5}  groups={s['groups']:4}  "
              f"kind={s['kind']}")
    print(f"anchors_fewshot n={len(anchors)}")
    print(f"\nwrote {OUT}/")


if __name__ == "__main__":
    main()
