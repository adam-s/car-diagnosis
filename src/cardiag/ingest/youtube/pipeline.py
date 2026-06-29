"""Per-video corpus builder v2. All v1 lessons baked in:

  acquire (yt-dlp, transient wav)
    -> cascade: energy + Silero VAD + spectral      (CPU, ~free)
    -> CLAP confirm: mechanical vs music/speech/etc (GPU, survivors only)
    -> CLAP fault-vs-tool gate (margin-based; tools kept as 'shop_tool' negatives)
    -> CLAP L1 sound-type ('normal' needs a WIDE margin; CLAP defaults to it)
    -> cyclic + spectral features
    -> clip wav + ledger record (provenance on everything)

Statuses: auto (trusted), review (human queue), reject (metadata only, no wav).

    uv run pipeline.py <video_id> "<title>"
"""
import subprocess
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from cardiag import config, paths
from cardiag.audio.cascade import candidate_regions, cyclic_features, spectral_fingerprint
from cardiag.audio.clap import Clap

DATA = paths.YT_DATA


def acquire(vid):
    wav = DATA / "tmp" / f"{vid}.wav"
    if not wav.exists():
        wav.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["yt-dlp", "--no-warnings", "-f", "ba", "-x", "--audio-format", "wav",
             "--postprocessor-args", f"-ar {config.SR_CLAP} -ac 1",
             "-o", str(DATA / "tmp" / "%(id)s.%(ext)s"),
             "--", f"https://www.youtube.com/watch?v={vid}"],
            check=True, capture_output=True, timeout=600)
    return wav


def l2_from_text(text):
    tl = (text or "").lower()
    return sorted({part for part, kws in config.L2_KEYWORDS.items()
                   if any(k in tl for k in kws)})


def gate(conf_p, fault_p, l1_p, shop_context=True):
    """All v1 lessons in one place. Returns (l1, l1_conf, margin, status, extras).
    shop_context: tools only happen where a mechanic is working (speech-heavy
    video). In a no-narration compilation, "tool-like" sounds are usually
    electric car parts (pumps!) -> review, never auto shop_tool."""
    mech = float(conf_p[:len(config.CONFIRM_KEEP)].sum())
    fault_m = float(fault_p[:len(config.FAULT_PROMPTS)].sum())
    tool_m = float(fault_p[len(config.FAULT_PROMPTS):].sum())
    o = np.argsort(-l1_p)
    l1 = config.L1_PROMPTS[o[0]][2:]
    conf, margin = float(l1_p[o[0]]), float(l1_p[o[0]] - l1_p[o[1]])
    extras = {"mech_confirm": round(mech, 3), "fault_mass": round(fault_m, 3),
              "tool_mass": round(tool_m, 3)}

    if mech < config.MECH_REJECT_BELOW:
        return l1, conf, margin, "reject", extras
    if tool_m - fault_m > config.TOOL_MARGIN:                  # confident tool sound
        extras["tool_evidence"] = config.TOOL_PROMPTS[int(fault_p[len(config.FAULT_PROMPTS):].argmax())]
        if shop_context:
            return "shop_tool", conf, margin, "auto", extras
        return l1, conf, margin, "review", extras              # no shop -> likely a pump etc.
    if tool_m > fault_m:                                       # ambiguous tool-ish
        return l1, conf, margin, "review", extras
    if mech < config.MECH_CONFIRM_MIN:
        return l1, conf, margin, "review", extras
    need_margin = config.NORMAL_MARGIN if l1 == config.L1_NORMAL else config.L1_MARGIN_MIN
    if conf >= config.L1_CONF_MIN and margin >= need_margin:
        return l1, conf, margin, "auto", extras
    return l1, conf, margin, "review", extras


def process_video(vid, title="", clap=None, verbose=False):
    t0 = time.time()
    wav = acquire(vid)
    y16, _ = librosa.load(str(wav), sr=config.SR_CHEAP, mono=True)
    regions, speech_frac = candidate_regions(y16, return_speech_frac=True)
    shop_context = speech_frac >= config.SHOP_SPEECH_FRAC
    if not regions:
        wav.unlink(missing_ok=True)
        print(f"{vid}: no clean candidates ({time.time()-t0:.0f}s)")
        return []

    y48, sr = librosa.load(str(wav), sr=config.SR_CLAP, mono=True)
    wav.unlink(missing_ok=True)                     # raw audio is transient
    clips = [y48[int(s * sr):int(e * sr)] for s, e in regions]
    clap = clap or Clap()
    conf_probs = clap.score(clips, config.CONFIRM_KEEP + config.CONFIRM_DROP)
    fault_probs = clap.score(clips, config.FAULT_PROMPTS + config.TOOL_PROMPTS)
    l1_probs = clap.score(clips, config.L1_PROMPTS)

    out_dir = DATA / "clips" / vid
    out_dir.mkdir(parents=True, exist_ok=True)
    l2 = l2_from_text(title)
    records = []
    for i, ((s, e), cp, fp, lp) in enumerate(zip(regions, conf_probs, fault_probs, l1_probs)):
        l1, conf, margin, status, extras = gate(cp, fp, lp, shop_context=shop_context)
        y22 = librosa.resample(clips[i], orig_sr=sr, target_sr=22050)
        per, hz, reg = cyclic_features(y22, 22050)
        rec = {"clip_id": f"{vid}_{i:02d}", "video": vid, "start": s, "end": e,
               "l1": l1, "l1_conf": round(conf, 3), "l1_margin": round(margin, 3),
               **extras,
               "cyclic": {"periodicity": round(per, 3), "pulse_hz": round(hz, 2),
                          "regularity": round(reg, 3)},
               "spectral": spectral_fingerprint(y22, 22050),
               "l2_candidates": l2, "status": status, "version": 2,
               "video_speech_frac": speech_frac,
               "provenance": {"detector": "cascade(v2)+clap-gate",
                              "models": "silero-vad + CLAP/laion-htsat-unfused",
                              "title": title or None}}
        if status != "reject":
            f = out_dir / f"clip_{i:02d}.wav"
            sf.write(f, clips[i], sr)
            rec["file"] = str(Path("data") / f.relative_to(paths.YT_DATA))
        records.append(rec)
    if not any(r.get("file") for r in records):
        out_dir.rmdir()                              # no empty dirs

    n = {s: sum(r["status"] == s for r in records) for s in ("auto", "review", "reject")}
    n_tool = sum(r["l1"] == "shop_tool" for r in records)
    print(f"{vid}: {len(records)} cands -> {n['auto']} auto ({n_tool} tool), "
          f"{n['review']} review, {n['reject']} reject | {time.time()-t0:.0f}s | L2={l2 or '-'}")
    if verbose:
        for r in records:
            print(f"  [{r['start']:7.1f}->{r['end']:7.1f}] {r['status']:<7} {r['l1']:<30} "
                  f"conf {r['l1_conf']:.2f} m {r['l1_margin']:.2f} tool {r['tool_mass']:.2f}")
    return records


if __name__ == "__main__":
    process_video(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "", verbose=True)
