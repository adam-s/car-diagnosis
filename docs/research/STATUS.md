# Project status — car-fault sound corpus & diagnosis

*Snapshot of where the whole effort stands. Companion to
`youtube/README.md`, `tiktok/README.md`, `external-data/SOURCES.md`,
`.claude/reference/academic-research-2026.md`.*

## Goal
Build a corpus + model that names a car's mechanical fault from its sound.
Two-level target: **L1 = sound type** (grind/squeal/knock/whine/tick/…, decidable
from audio), **L2 = cause/part** (from text, not audio alone — fine cause is
acoustically undecidable, confirmed by us *and* the literature).

## What's built (Phase 1 — data collection: DONE)
A cheap-first, music-aware, multi-signal pipeline, mirrored across two platforms:

```text
discover → acquire → cascade(silero+energy) → CLAP gates → OCR/chapter/transcript
        → Sonnet fusion (multi-signal → cause+confidence) → code-side tiering
```

Per-platform scripts: `discover.py` (YT ytsearch / TT stealth-browser intercept),
`pipeline.py`/`batch.py`, `audio.py` (CLAP+cascade+cyclic), plus enrichment
(`capture.py` chapters/transcript, `enrich.py`, `normalize.py`), `fusion.py`
(Sonnet, multi-signal), `gate_tier.py` (code-side trust), `music_gate.py`.

## Corpus state (the deliverable)
| | clips | gold (trustworthy cause) | silver | bronze | music |
|---|---|---|---|---|---|
| YouTube | 9,021 | 1,607 | 3,195 | 4,219 | — |
| TikTok | 3,617 | 513 | 1,752 | 392 | 960 (excluded) |
| **Total** | **12,638** | **2,120** | 4,947 | 4,611 | 960 |

- **Gold = cause + ≥2 independent signals agree (text-backed) + Sonnet conf ≥0.6 +
  music-free.** Verified: 0 junk-OCR and 0 music in gold (hand-audited).
- Top gold causes: wheel bearing ~505, brakes ~440, belt ~264, power-steering
  ~159, CV joint ~142, water pump ~153.

## External verified-anchor set (~1,900 clips, in `external-data/`)
*Trustworthy* labels (vs our weak ones) for **calibration + held-out eval**:
malakragaie car-diagnostics (1,386, fault×state), Car-Engine-Sounds (216,
vehicle+fault), Ai_Mechanic (297), Sound-Based **DB1** (~30 fine fault exemplars).
Bench anchors downloading: IDMT, DCASE-2025, MIMII pump/fan (normal class).

## Key decisions / lessons (the "why")
- **Sound-type from audio, cause from text** — fine part-from-audio is expert-hard
  (knock vs tick) and partly non-acoustic; literature-confirmed.
- **Trust = code-side multi-signal agreement, NOT model confidence** — Sonnet
  overstamps confidence on junk; the gold tier is gated in code.
- **Music gate** — 26% of TikTok was background music; CLAP rejects it cleanly.
- **Cost**: Haiku for routine extraction; **Sonnet** for the diagnostic *inference*
  it does that Haiku declines; JSON-only prompt (~34% cheaper); Opus ≈ Sonnet cost
  but no quality gain (audited).
- **Cheap-first cascade**: free tiers (Silero VAD, energy) discard ~97% before CLAP.

## Next phase (Phase 2 — modeling, STARTED)
0. ✅ **Training manifests** (`training1/`): canonicalized labels (359 raw causes
   → 24 part-families, 10 L1 sound types), leakage-safe creator/video-group
   splits (train 7,077 / val 824 / test 907), external anchors as
   `external_eval` (1,140) + DB1 few-shot anchors. Clips referenced in place.
1. ✅ **Calibrate** (`training1/calibrate.py`, vs 1,356 verified clips):
   - zero-shot CLAP cause-from-audio: **45.7%** vs 42.4% majority baseline →
     the app cannot ship on zero-shot; the trained model must close the gap.
   - fault-vs-normal from audio: 71% (pipeline rule ≈ best threshold).
   - **knock recall 0%** on verified knocking engines (CLAP hears "normal
     idle") → scrape's *normal* class may hide faults; L1 knock/tick/idle
     boundary needs the trained model, not CLAP.
   - `l1_conf` is uncalibrated (0.7–0.8 bins *worse* than 0.4–0.6) — never
     use as a label weight without recalibration.
   - Gate false-discard on real fault clips: only ~3% hard-reject. ✔
2. ✅ **First model — CLAP embeddings + linear heads** (`embed.py`,
   `train_heads.py`): cause top1 0.329 / top3 0.593 in-domain (vs 0.189
   baseline); **silver labels load-bearing** (gold-only 0.253 < baseline,
   gold+silver 0.386); kind fault-vs-normal 0.742 verified (> 0.712 zero-shot);
   **L1 knock recall still 0%** — CLAP feature ceiling, large scrape→verified
   domain gap. ⇒ **Embed + train metric head on BEATs** next (literature:
   BEATs ≫ CLAP for the trained model, esp. unseen creators; LoRA fine-tune).
   CLAP stays for zero-shot labeling.
3. **Splits by creator/video, not clip** (anti-leakage); eval vs SNR + cross-domain.
4. **Few-shot retrieval** reference library seeded by gold + external anchors;
   abstain without multi-creator agreement.
5. **Anomaly-detection fallback** on the normal class; anomaly-score normalization.
6. **App** (ref: `external-data/repos`/autoear-ai product design): record → predict
   sound-type + top-k cause + nearest reference clips + risk/drive-or-not.

## Phase 2 update — honest splits + shippable confidence (2026-06-12)
The push toward a high-accuracy app forced the split question. Findings, in
order of how much they change the plan:

1. **"90%" was video-grouped CV variance, not a held-out number.** Re-measured
   region triage (engine vs chassis) on the 717-clip curated set
   (`training1/split_check.py`): CV-by-creator cov@p90 **0.80 ± 0.19** — the
   ±0.19 is the story. A single fixed creator-held-out test lands anywhere in
   that band. Report fixed-held-out + CV±std from now on, never bare CV.

2. **A `StandardScaler` before the logistic head is the cheapest real win.**
   Whitening the CLAP embedding per-dimension moves the *identical* fixed
   held-out from cov@p90 **0.562 → 0.800** (acc 0.808 vs 0.785 majority).
   Verified by ablation (scaler off/on × sigmoid/iso: 0.36/0.56 → 0.75/0.80).
   `predict.py`'s head should be refit with the scaler in-pipeline.

3. **Calibrated confidence works and is the product answer** (`confidence.py`,
   all numbers OOF creator-grouped). ECE 0.018; reliability holds:

   | model claims | empirical acc | n |   | predicts ENGINE @≥t | CHASSIS @≥t |
   |---|---|---|---|---|---|
   | 0.80–0.90 | 0.857 | 231 | t=0.70 | 16/16 = **1.00** | 0.89 |
   | 0.90–0.95 | 0.927 | 179 | t=0.90 | 7/7 = **1.00** | 0.96 |
   | 0.95–1.00 | **0.993** | 149 | t=0.95 | 6/6 = **1.00** | **0.99** |

   **Ship bands:** HIGH ≥0.95 (~99% right, fires ~21% of clips) · MEDIUM
   0.85–0.95 (~93%) · else ABSTAIN ("record closer, engine idling, no talk").
   The 99%-with-confidence goal is met *on the confident subset*; the model
   abstains elsewhere. It is never *wrong* saying engine (100% precision) but
   only dares on ~16/120 true-engine clips — engine **recall**, driven by the
   597/120 chassis/engine imbalance, is the remaining ceiling.

4. **More features do NOT help** (`confidence.py` ablation, full 717 w/ AST+DSP
   caches filled by `fill_features.py`): plain CLAP + scaler + isotonic beats
   CLAP+AST, +DSP2, and 5-MLP ensembles on coverage *and* calibration. MLPs win
   raw acc (0.879) but are overconfident (cov@p95 collapses to 0.06). Label
   quality + class balance remain the binding constraint — again.

5. **Audio-LLM second opinion (Qwen2-Audio-7B on Modal): NEGATIVE, dropped.**
   `modal_audio.py` + `eval_second_opinion.py`, 50 balanced clips, two prompts
   (naive + de-biased). Both **collapse to a constant** "region=engine,
   part=valvetrain, conf=0.8" on 50/50 clips. The free-text `heard` field
   varied per clip (it *is* hearing the audio: "fast metallic ticking" vs
   "rhythmic clunk"), but the structured region decision has an unshakeable
   engine prior — so it can neither give a second opinion nor verify labels
   cross-modally. Same lesson as CLAP knock-blindness: off-the-shelf audio
   models carry priors that don't fit this fine task. Revisit only with
   few-shot in-context or fine-tuning (no longer a cheap smoke test). Cost of
   finding this out: ~$1 and one evening.

6. **Apple MLSoundClassifier / SoundAnalysis baseline (2026-06-12): competitive
   on accuracy, much weaker on confidence.** Trained CreateML's on-device sound
   classifier (`build_cml.py` → `cml_train.swift` → `cml_score.swift`) on the
   *same* creator-grouped split, scored on the *same* 130 held-out clips as the
   CLAP head. Architecturally it's our recipe (frozen Apple audio embedding +
   trained head), so it's a fair backbone + the real on-device deployment path.

   | | Apple MLSoundClassifier | CLAP + scaler + isotonic |
   |---|---|---|
   | held-out accuracy | **0.831** | 0.808 |
   | engine recall | 0.36 (10/28) | 0.25 (7/28) |
   | chassis recall | 0.96 | 0.96 |
   | cov@p90 (best confidence signal) | 0.40 (window-vote) / 0.09 (raw) | **0.80** |
   | guessing? | no — predicts both | no — predicts both |
   | on-device | yes, native | needs Core ML conversion |

   Takeaways: (a) Apple's backbone is **as good or a hair better** on raw labels
   (3 more clips of 130, within noise) — confirms a *third* time that the
   backbone is not the ceiling. (b) **Neither guesses** (unlike Qwen2-Audio):
   both predict both classes, both nail chassis, both under-call engine — the
   597/120 imbalance is the shared ceiling. (c) **The decisive gap is
   confidence**: our calibrated head answers 80% of clips at 90% precision;
   CreateML's softmax answers 40% at best (window-vote) or 9% raw, and cov@p90
   is invariant to monotonic recalibration, so this is a *ranking* weakness
   calibration can't fix — CreateML's end-to-end confidence is simply a poor
   selective-prediction signal. For the "tell the human when it's sure" goal,
   CLAP+calibration wins decisively. Synthesis: Apple model = good on-device
   *label*; for trustworthy *confidence* keep our calibrated head (convertible
   to Core ML for on-device too). Reverses the earlier "drop MLSoundClassifier"
   — as an on-device baseline it's worth keeping, just not for confidence.

7. **Per-SYSTEM (multi-class) from audio is near-chance — thesis confirmed hard
   (2026-06-12).** `systems_eval.py` / top-k sweep, 3,698 high-trust clips, 10
   systems, creator-grouped OOF, best config: **top-1 0.310 (majority 0.279),
   top-3 0.613**, and **no confident subset exists** (max OOF confidence peaks
   at 0.56; cov@p60 = 0). The model collapses to the two catch-all buckets and
   cannot resolve the specific part. Top-3 recall by system:

   | works (coarse bucket) | top-3 | weak | top-3 | ~undetectable | top-3 |
   |---|---|---|---|---|---|
   | suspension | 0.99 | accessory_belt | 0.67 | exhaust | 0.16 |
   | driveline | 0.95 | valvetrain | 0.38 | fuel_ignition | 0.16 |
   |  |  | engine_internal | 0.34 | steering | **0.03** |
   |  |  | brakes | 0.25 | cooling | **0.02** |

   Read: from audio alone we can say "suspension-ish" or "driveline/trans-ish"
   (the catch-alls absorb everything), weakly flag belt/engine sounds, and
   essentially **cannot** identify steering, brakes, cooling, exhaust, or
   distinguish parts *within* suspension (bushing vs ball joint vs shock) or
   driveline (transmission vs CV vs u-joint). The catch-alls' high top-3 is
   absorption, not identification. This is the project's founding thesis —
   *sound type from audio, fine cause from text/context* — now quantified: the
   fine system is largely non-acoustic. Naming brakes-vs-steering-vs-trans needs
   user context (when/where the noise happens: braking? turning? over bumps? at
   speed?), which is exactly what the text pipeline mines. The honest product is
   audio (sound-type + engine/chassis region + confidence) as TRIAGE, narrowed
   to a part by a few context questions — not audio-only part naming.

8. **The confidence frontier — how fine can we narrow? (2026-06-12).**
   `systems_confidence.py` (drop class_weight=balanced — it buys minority recall
   at the cost of calibration), `granularity_frontier.py`, `binary_probes.py`.
   Calibration is honest at *every* granularity (ECE 0.02–0.05): the model
   reports low confidence rather than faking it. The question is where honest
   confidence stays *high enough to act on*:
   - **Fine part (20-way) and system (10-way): no.** Max OOF confidence ~0.50;
     no part/system clears a 50% bar; top-1 0.31 / 0.22. The model correctly
     knows it can't tell — it will never honestly say "80% brakes."
   - **Per-system INCLUDE / RULE-OUT (process of elimination): essentially no.**
     0% of clips can be confidently called any single system @85% precision
     (running-gear 4%); the one big rule-out (steering 93%) is ≈ its 94% base
     rate, i.e. trivial. Adding brakes/belt/exhaust/cooling/steering tangles
     everything — they don't separate from each other or from engine.
   - **Engine-internal vs running-gear (wheels/susp/driveline): YES, the one
     usable distinction.** Dropping the accessory systems: acc 0.76–0.79,
     **cov@p80 0.80–0.96** (confidently triages most clips at 80% precision),
     cov@p90 0.11–0.16 here — and **cov@p90 0.80 on the fully-curated 717**.
     The gap is curation depth (mech-filter only vs the full L1-domain gate).

   Product reading: the honest narrowing is a **2-bucket coarse triage —
   "engine-internal noise" vs "running-gear (wheel/suspension/driveline)
   noise" — with abstention**, not a full multi-system elimination. The app can
   truthfully show a calibrated "~65% running-gear" because the probability is
   honest (low ECE); it just can't resolve which brake/bushing/steering part.
   Lever: deeper curation + balanced engine data raises both the precision and
   the coverage of this coarse triage (the 717 already shows cov@p90 0.80).
   Saved calibrated artifacts: `conf_model_system.joblib`, `conf_model_part.joblib`
   (honest probabilities for display, even where confidence is low).

9. **RUNNABLE DEMO + iteration (2026-06-12).** `triage.py` (train/iterate),
   `demo.py` (audio file -> triage + confidence). The product is the validated
   coarse triage — engine-internal vs running-gear (wheel/susp/driveline) — with
   a calibrated confidence band. Iterated honestly (4-seed creator hold-out,
   metric = cov@p90):
   - baseline (strict gate, CLAP, isotonic): cov@p90 0.793±0.129, cov@p80 1.0
   - swept gates × {CLAP, CLAP+AST} × calibration: CLAP wins, cov@p80=1.0 every
     config; CLAP+AST stabilizes CV but on far fewer (AST-limited) clips.
   - **iterative self-cleaning** (drop training clips OOF-misclassified ≥0.85 —
     high-confidence errors are mostly mislabels): **cov@p90 0.793 → 0.816,
     variance 0.129 → 0.090**, dropping 24/816. sigmoid + other thresholds all
     worse. Saved `triage_model.joblib` (786 clips after cleaning).

   **Held-out (creator-grouped OOF) per-band reliability — the deliverable:**

   | band | % of clips | accuracy |
   |---|---|---|
   | HIGH (≥90%) | 49% | 90% |
   | MEDIUM (80–90%) | 23% | 84% |
   | LOW (65–80%) | 18% | 82% |
   | ABSTAIN (<65%) | 10% | 57% |

   Confidently answers **72% of clips at 88% accuracy**; the abstain bucket is
   near coin-flip (57%) — exactly the clips it should refuse. The confidence is
   calibrated and trustworthy, which was the goal. `demo.py` shows engine vs
   running-gear + HIGH/MEDIUM/LOW/ABSTAIN + a next-step, and explicitly refuses
   to name the exact part. (Verified live: a ball-joint clip → running-gear
   HIGH 100% ✓; a real engine clip mis-leaned running-gear but at LOW 75% —
   the engine-recall gap surfaces as honest low confidence, not a confident
   error.) This is POC-ready.

### Ranked levers that survive (best expected value first)
- **Grow the engine-internal minority** (scrape/curate). Pure data; directly
  lifts engine recall and shrinks the ±0.19 variance. Highest value, no Modal.
- **Refit `predict.py` with the scaler + ship the confidence bands above.**
  Local, done-able now, turns the validated numbers into the app.
- **Multi-window aggregation** (already light-TTA in `predict.py`): predict per
  5s window, average; agreement across windows is itself a confidence signal.
- **Bigger TEXT LLM on Modal for label extraction** (Qwen-32B/72B-AWQ) — text
  mining is where Qwen *works*; modest expected gain, ~$1–2 to test.
- *Not worth it:* more feature stacking (measured — no), audio-LLM zero-shot
  (measured — no), MLSoundClassifier (dropped), backbone fine-tune (too few
  curated clips until the minority grows).

## Open items
- Scale scrape 3–5× to lift per-cause counts, **prioritizing engine-internal**
  (rod/main bearing, lifter/valvetrain, low-oil) to fix the 597/120 imbalance.
- Demucs to *recover* music-buried clips (whole-clip input).
- The richest dataset (232-vehicle, arXiv 2403.11037) is proprietary — email authors.
