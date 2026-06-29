"""Calibrate the pipeline's audio-side labeling against verified ground truth.

Runs the EXACT pipeline stack (CLAP model, prompts, gate() thresholds from
ingest/youtube/{config,audio} + pipeline.py) over external_eval.jsonl (clips with
trustworthy human labels the pipeline has never seen) and measures:

  1. gate behavior     : these are all real car sounds, so reject/review rates
                         are a direct false-discard estimate for the scrape
  2. fault vs normal   : can the audio side separate them, and at what threshold
  3. l1 plausibility   : does the predicted sound type match the sound family
                         the verified cause should make (brakes->squeal/grind…)
  4. zero-shot cause   : audio-only cause ceiling (we expect this to be weak;
                         that's WHY cause comes from text; here we measure it)
  5. confidence calibration : is l1_conf actually monotone in correctness

Usage (from repo root; scoring ~minutes on MPS, cached after first run):
    uv run training/eval/calibrate.py            # score + analyze
    uv run training/eval/calibrate.py --analyze  # re-analyze cached scores only
"""
import json
import sys
import time
from collections import Counter, defaultdict

import librosa
import numpy as np

from cardiag import (
    config,  # pipeline prompts/thresholds
    paths,
)
from cardiag.audio.clap import Clap  # pipeline CLAP wrapper
from cardiag.ingest.youtube.pipeline import gate  # pipeline decision rule
from cardiag.training.prep.causes import canonical_l1

OUT = paths.TRAIN_DATA
SCORES = OUT / "calibration_scores.jsonl"
REPORT = OUT / "calibration.json"

# Zero-shot cause probe over the verified car-diagnostics classes. This is
# the experiment the pipeline deliberately does NOT do (cause comes from
# text): calibration measures what audio alone buys.
CAUSE_PROMPTS = {
    "brakes": "worn brake pads squealing or grinding on a car",
    "belt": "a loose serpentine belt squealing in a car engine",
    "power_steering": "a failing power steering pump whining in a car",
    "low_oil": "a car engine ticking or knocking from low oil",
    "accessories": "a dead car battery clicking rapidly when trying to start",
    "fuel_ignition": "a car engine cranking and struggling to start",
    "normal": "a normal healthy car engine running smoothly",
}

# Which l1 sound families a verified cause should plausibly produce.
# Only causes with a well-defined acoustic signature participate in the
# plausibility metric (battery/ignition are start-event sounds, too fuzzy).
EXPECTED_L1 = {
    "brakes": {"squeal", "grind"},
    "belt": {"squeal"},
    "power_steering": {"whine", "hum", "squeal"},
    "low_oil": {"tick", "knock"},
}

MAX_S = 10.0  # CLAP/HTSAT window; longer files get a centered crop


def load_clip(path):
    dur = librosa.get_duration(path=str(path))
    off = max(0.0, (dur - MAX_S) / 2)
    y, _ = librosa.load(str(path), sr=config.SR_CLAP, mono=True,
                        offset=off, duration=MAX_S)
    return y if len(y) >= config.SR_CLAP // 2 else None  # <0.5s unusable


def score_all(recs):
    clap = Clap()
    cause_keys = list(CAUSE_PROMPTS)
    n_fail = 0
    t0 = time.time()
    with open(SCORES, "w") as fh:
        for i in range(0, len(recs), 96):
            chunk, clips = [], []
            for r in recs[i:i + 96]:
                try:
                    y = load_clip(paths.resolve_clip(r["path"]))
                except Exception:
                    y = None
                if y is None:
                    n_fail += 1
                    continue
                chunk.append(r)
                clips.append(y)
            if not clips:
                continue
            conf_p = clap.score(clips, config.CONFIRM_KEEP + config.CONFIRM_DROP)
            fault_p = clap.score(clips, config.FAULT_PROMPTS + config.TOOL_PROMPTS)
            l1_p = clap.score(clips, config.L1_PROMPTS)
            cause_p = clap.score(clips, list(CAUSE_PROMPTS.values()))
            for r, cp, fp, lp, zp in zip(chunk, conf_p, fault_p, l1_p, cause_p):
                # direct recordings, not repair vlogs -> no shop context
                l1, conf, margin, status, extras = gate(cp, fp, lp,
                                                        shop_context=False)
                fh.write(json.dumps({
                    "id": r["id"], "source": r["source"],
                    "gt_kind": r["kind"], "gt_cause": r.get("cause"),
                    "gt_l1": r.get("l1"), "state": r.get("state"),
                    "l1": canonical_l1(l1), "l1_conf": round(conf, 3),
                    "l1_margin": round(margin, 3), "status": status,
                    "normal_p": round(float(
                        lp[config.L1_PROMPTS.index("a " + config.L1_NORMAL)]), 3),
                    **extras,
                    "zs_cause": cause_keys[int(np.argmax(zp))],
                    "zs_cause_conf": round(float(np.max(zp)), 3),
                }) + "\n")
            done = min(i + 96, len(recs))
            print(f"  scored {done}/{len(recs)}  ({time.time()-t0:.0f}s)")
    if n_fail:
        print(f"  ! {n_fail} clips failed to load (skipped)")


def analyze():
    # join ground truth from the current manifest (not the cached scores) so
    # label fixes in prepare.py take effect without a full CLAP rescore;
    # scored clips no longer in the manifest are dropped.
    manifest = {r["id"]: r for r in
                (json.loads(l) for l in open(OUT / "external_eval.jsonl"))}
    rows = []
    for l in open(SCORES):
        r = json.loads(l)
        m = manifest.get(r["id"])
        if m is None:
            continue
        r.update(gt_kind=m["kind"], gt_cause=m.get("cause"),
                 gt_l1=m.get("l1"), state=m.get("state"))
        rows.append(r)
    cd = [r for r in rows if r["source"] == "ext:car-diagnostics"]
    am = [r for r in rows if r["source"] == "ext:ai-mechanic"]
    rep = {"n_scored": len(rows)}

    # 1. gate behavior on verified real car sounds (false-discard estimate)
    rep["gate"] = {
        "all": dict(Counter(r["status"] for r in rows)),
        "by_kind": {k: dict(Counter(r["status"] for r in rows
                                    if r["gt_kind"] == k))
                    for k in ("fault", "normal", "nonauto")},
    }

    # 2. fault vs normal (car-diagnostics only: both labels verified).
    # Pipeline rule: l1 == normal_idle. Also sweep normal_p for best split.
    fn = [r for r in cd if r["gt_kind"] in ("fault", "normal")]
    pipe_acc = np.mean([(r["l1"] == "normal_idle") == (r["gt_kind"] == "normal")
                        for r in fn])
    best = max(
        ((t, np.mean([(r["normal_p"] >= t) == (r["gt_kind"] == "normal")
                      for r in fn])) for t in np.arange(0.05, 0.95, 0.05)),
        key=lambda x: x[1])
    rep["fault_vs_normal"] = {
        "n": len(fn),
        "acc_pipeline_rule": round(float(pipe_acc), 3),
        "acc_best_normal_p_threshold": round(float(best[1]), 3),
        "best_threshold": round(float(best[0]), 2),
        "confusion_l1_by_kind": {
            k: dict(Counter(r["l1"] for r in fn
                            if r["gt_kind"] == k).most_common())
            for k in ("fault", "normal")},
    }

    # 3. l1 plausibility: predicted sound type within the verified cause's
    # expected sound family (acoustically well-defined causes only)
    plaus = {}
    for cause, fam in EXPECTED_L1.items():
        sub = [r for r in cd if r["gt_cause"] == cause]
        if sub:
            plaus[cause] = {
                "n": len(sub),
                "l1_in_family": round(
                    float(np.mean([r["l1"] in fam for r in sub])), 3),
                "l1_dist": dict(Counter(r["l1"] for r in sub).most_common()),
            }
    rep["l1_plausibility"] = plaus

    # 3b. true L1 accuracy where sound-type ground truth exists
    # (Car-Engine-Sounds: filename-verified knock/tick/... + normal)
    gl = [r for r in rows if r.get("gt_l1")]
    if gl:
        conf_l1 = defaultdict(Counter)
        for r in gl:
            conf_l1[r["gt_l1"]][r["l1"]] += 1
        rep["l1_accuracy_verified"] = {
            "n": len(gl),
            "acc": round(float(np.mean([r["l1"] == r["gt_l1"]
                                        for r in gl])), 3),
            "per_class_recall": {
                g: round(conf_l1[g][g] / sum(conf_l1[g].values()), 3)
                for g in sorted(conf_l1)},
            "confusion": {g: dict(c.most_common())
                          for g, c in conf_l1.items()},
        }

    # 4. zero-shot cause (the audio-only ceiling)
    probe = [r for r in cd if r["gt_cause"] in CAUSE_PROMPTS
             or r["gt_kind"] == "normal"]
    gt = [r["gt_cause"] or "normal" for r in probe]
    pred = [r["zs_cause"] for r in probe]
    conf_mat = defaultdict(Counter)
    for g, p in zip(gt, pred):
        conf_mat[g][p] += 1
    rep["zero_shot_cause"] = {
        "n": len(probe),
        "acc": round(float(np.mean([g == p for g, p in zip(gt, pred)])), 3),
        "majority_baseline": round(
            max(Counter(gt).values()) / max(1, len(gt)), 3),
        "per_class_recall": {
            g: round(conf_mat[g][g] / sum(conf_mat[g].values()), 3)
            for g in sorted(conf_mat)},
        "confusion": {g: dict(c.most_common()) for g, c in conf_mat.items()},
    }

    # 5. confidence calibration: l1_conf bins vs in-family correctness
    binned = defaultdict(list)
    for r in cd:
        fam = EXPECTED_L1.get(r["gt_cause"])
        if fam:
            binned[round(r["l1_conf"] // 0.1 * 0.1, 1)].append(r["l1"] in fam)
    rep["l1_conf_calibration"] = {
        f"{b:.1f}": {"n": len(v), "in_family": round(float(np.mean(v)), 3)}
        for b, v in sorted(binned.items())}

    # 6. ai-mechanic: nonauto separation via mech_confirm
    if am:
        nonauto = [r["mech_confirm"] for r in am if r["gt_kind"] == "nonauto"]
        engine = [r["mech_confirm"] for r in am if r["gt_kind"] != "nonauto"]
        rep["nonauto_separation"] = {
            "n_nonauto": len(nonauto), "n_engine": len(engine),
            "mech_confirm_median_nonauto": round(float(np.median(nonauto)), 3)
            if nonauto else None,
            "mech_confirm_median_engine": round(float(np.median(engine)), 3)
            if engine else None,
            "acc_at_pipeline_min": round(float(np.mean(
                [(m < config.MECH_CONFIRM_MIN) for m in nonauto]
                + [(m >= config.MECH_CONFIRM_MIN) for m in engine])), 3)
            if nonauto and engine else None,
        }

    json.dump(rep, open(REPORT, "w"), indent=1)
    print(json.dumps(rep, indent=1))
    print(f"\nwrote {REPORT}")


def main():
    if "--analyze" not in sys.argv:
        recs = [json.loads(l)
                for l in open(OUT / "external_eval.jsonl")]
        print(f"scoring {len(recs)} verified clips with pipeline CLAP stack…")
        score_all(recs)
    analyze()


if __name__ == "__main__":
    main()
