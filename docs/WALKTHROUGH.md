# Walkthrough: one clip, every stage

This traces a single recording through the whole pipeline with real numbers, so
you can see exactly what happens between "an audio file" and "a diagnosis." Every
command here works on a fresh clone.

> **Want to *see* it instead of read it?** Run `cardiag inspect <clip.wav>` — it
> renders this same story as an HTML page with spectrograms, the CLAP scores, and
> before/after audio you can play. The screenshots below come from it.

## 0. Get set up (90 seconds)

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[scrape,web,dev,viz]"
python -m camoufox fetch          # stealth Firefox for TikTok/Reddit
cardiag doctor                    # confirms everything's ready
```

`cardiag doctor` checks every dependency and prints a fix for anything missing —
you should see all ✓.

## 1. Get a model without waiting (the fast path)

You need a model to diagnose. The fastest way to *see the loop close* is to train
on the bundled sample embeddings — **offline, ~1.5 seconds, no scrape**:

```bash
cardiag train --fixtures
```

This trains four classifiers (`kind`, `knock`, `cause`, `triage`) on 74 bundled
CLAP embeddings and writes `data/training/best_model_clap.joblib`. It exists so a
beginner can reach a working `diagnose` immediately; the real model comes from
scraping (step 5).

## 2. What "clean" does to a clip

A real recording is mostly *not* the fault sound — it's talking, music, silence,
road noise. The cascade isolates the mechanical part:

```bash
cardiag clean some_clip.wav
```

```jsonc
{
  "total_seconds": 19.3,
  "kept_seconds": 2.3,          // only 2.3s of 19.3s was the actual noise
  "speech_fraction": 0.0,
  "music_probability": 0.0,
  "is_music": false,
  "segments": [                 // the spans it kept
    {"start": 13.0, "end": 14.3, "duration": 1.3},
    {"start": 14.8, "end": 15.8, "duration": 1.0}
  ]
}
```

What ran, in order (cheap → expensive, so 97% is discarded before the costly model):

| Stage | Removes | How |
|---|---|---|
| energy / RMS gate | silence | librosa RMS |
| **Silero VAD** | talking / narration | a tiny speech model |
| spectral flatness | wind / hiss / static | librosa |
| **CLAP music gate** | background music | CLAP zero-shot score |

The exact same `clean()` runs at training time and at inference — that's why an
uploaded clip is treated identically to a training clip.

## 3. Why it got that label

After cleaning, CLAP scores the isolated audio against each sound-type prompt:

```
grinding noise          ███████████████  67%
rattling noise          ███              11%
squealing/squeaking     █                 6%
high-pitched whine                        4%
```

The top sound-type becomes the clip's `l1` label. Note what's **not** here: the
*cause* (wheel bearing vs CV joint). Audio alone can't reliably tell those apart —
that label comes from the video's **title/caption text**, not the waveform. That
honesty is the whole point of the project.

## 4. Diagnose

```bash
cardiag diagnose some_clip.wav
```

```
Verdict: FAULT  (fault p=0.95)
Most likely cause (audio is suggestive, not definitive):
    39% ███████          low_oil
    17% ███              power_steering
    17% ███              belt
Isolated 2 mechanical span(s), 2.3s total.
```

`diagnose` re-runs `clean()`, embeds the isolated audio with CLAP, runs the three
heads, and reports a verdict + a *ranked, probabilistic* cause — never a single
confident answer. For the coarse-but-calibrated call, use `cardiag triage`.

## 5. Now do it for real (build your own corpus)

The fixtures model is a teaching stand-in. To build a real one, scrape across all
three sources (Camoufox handles TikTok + Reddit; yt-dlp handles YouTube):

```bash
cardiag scrape youtube --per-query 5 --max-videos 200
cardiag scrape reddit  --pages 3
cardiag scrape tiktok  --max-videos 80
cardiag train
```

Each `scrape` discovers videos, downloads them, runs the §2 cascade, labels the
survivors (`kind` from the query set, `l1` from CLAP, `cause` from title text),
and appends to `data/<platform>/corpus.jsonl`. `train` embeds every clip and
retrains the heads. Then `cardiag inspect --sample 12 -o report.html` lets you
audit what you collected — listen to clips, see the labels, judge them yourself.

## The whole loop, one command

```bash
cardiag demo        # scrape (3 sources) → clean → train → diagnose, end to end
```

That's the entire project in one invocation — and every step above is just a
piece of it you can run on its own.
