"""Visual-review harness for the streaming UI — renders the page across a set of
representative clips and saves screenshots so the look of the data visualization
can be judged (and iterated on) qualitatively, not just asserted to "work".

It builds inputs that exercise the interesting viz states — a narrated clip (so
the speech-removal overlay fires), real fault/normal clips (real spectrograms +
calibrated verdicts), and a near-silent clip (the empty state) — starts the
server with the shipped model, drives it in a headless browser, and writes a
full-page screenshot per clip plus close-ups of the waveform and spectrogram.

    python scripts/ui_review.py --out /tmp/ui_review            # uses models/ + data/ clips

Requires the [web] + [dev] extras (fastapi, uvicorn, playwright) and, for the
narrated clip, macOS `say` (optional; skipped if absent).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SR = 48000


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _load(path, sr=SR):
    import librosa
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


def _real_clip(kind: str, l1_prefix: str) -> Path | None:
    for src in ("youtube", "tiktok", "reddit"):
        f = ROOT / "data" / src / "corpus.jsonl"
        if not f.exists():
            continue
        for ln in open(f):
            r = json.loads(ln)
            if (r.get("kind") == kind and (r.get("l1") or "").startswith(l1_prefix)
                    and os.path.exists(r.get("wav", ""))):
                return Path(r["wav"])
    return None


def build_clips(outdir: Path) -> list[tuple[str, Path]]:
    """Create the representative inputs. Returns [(label, wav_path), …]."""
    outdir.mkdir(parents=True, exist_ok=True)
    clips: list[tuple[str, Path]] = []

    # 1) narrated clip: TTS speech + a real knock -> speech removed (red) + kept (green)
    knock = _real_clip("fault", "knocking")
    if knock and shutil.which("say"):
        aiff = outdir / "_say.aiff"
        subprocess.run(["say", "-o", str(aiff),
                        "Here's what a bad engine knock sounds like when it's idling."],
                       check=False)
        try:
            speech = _load(aiff)
            mech = _load(knock)
            gap = np.zeros(int(0.3 * SR), np.float32)
            y = np.concatenate([speech * 0.6, gap, mech])
            p = outdir / "1_narrated_knock.wav"
            sf.write(p, y, SR)
            clips.append(("narrated_knock", p))
        except Exception as e:
            print(f"  (narrated clip skipped: {e})")

    # 2) music + mechanical: a synthesized melody (CLAP flags it ~0.98 as music) then
    #    a real mechanical clip -> the music gate drops the musical span (purple)
    grind = _real_clip("fault", "grinding")
    if grind is not None:
        sr = SR
        notes = [261, 330, 392, 523, 392, 330, 440, 349]

        def _tone(f, d=0.28):
            t = np.linspace(0, d, int(sr * d), endpoint=False)
            wav = np.sin(2*np.pi*f*t) + 0.5*np.sin(2*np.pi*2*f*t) + 0.3*np.sin(2*np.pi*3*f*t)
            return (wav * np.hanning(len(t))).astype(np.float32)
        mel = np.concatenate([0.3 * _tone(n) for n in notes])
        beat = 0.6 + 0.4 * np.sign(np.sin(2*np.pi*2*np.linspace(0, len(mel)/sr, len(mel))))
        music = (mel * beat).astype(np.float32)
        # >0.5s gap so the cascade keeps music + mechanical as SEPARATE spans
        # (one dropped as music, one kept) rather than merging them into one
        y = np.concatenate([music, np.zeros(int(0.9*sr), np.float32), _load(grind)])
        p = outdir / "2_music_then_grind.wav"
        sf.write(p, y, SR)
        clips.append(("music_drop", p))

    # 3) multi-span: three different real sounds with gaps -> the cascade breaks the
    #    clip into several isolated spans, each numbered and diagnosed
    parts = [c for c in (_real_clip("fault", "knocking"), _real_clip("fault", "squealing"),
                         _real_clip("fault", "grinding")) if c is not None]
    if len(parts) >= 2:
        gap = np.zeros(int(0.8 * SR), np.float32)        # >0.5s so spans stay separate
        segs = []
        for c in parts[:3]:
            segs += [_load(c)[:int(1.5 * SR)], gap]
        sf.write(outdir / "3_multi_span.wav", np.concatenate(segs), SR)
        clips.append(("multi_span", outdir / "3_multi_span.wav"))

    # 4-5) real single clips: a normal idle and a squeal — real spectrograms + verdicts
    for label, kind, l1 in [("normal_idle", "normal", "normal smooth"),
                            ("squeal_fault", "fault", "squealing")]:
        c = _real_clip(kind, l1)
        if c:
            p = outdir / f"{len(clips)+2}_{label}.wav"
            sf.write(p, _load(c), SR)
            clips.append((label, p))

    # 5) near-silent -> the empty-state narration ("no clean span survived")
    p = outdir / "9_quiet.wav"
    sf.write(p, (1e-4 * np.random.default_rng(0).standard_normal(SR * 3)).astype(np.float32), SR)
    clips.append(("quiet_empty", p))
    return clips


def review(clips, base_url: str, outdir: Path) -> None:
    from playwright.sync_api import sync_playwright
    shots = []
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        for label, wav in clips:
            pg = b.new_page(viewport={"width": 1120, "height": 1600}, device_scale_factor=2)
            pg.goto(base_url, wait_until="networkidle")
            pg.set_input_files("#file", str(wav))
            try:
                pg.wait_for_selector("#diagnosis", state="visible", timeout=60000)
                pg.wait_for_function(
                    "document.querySelector('#verdict')&&document.querySelector('#verdict').textContent!=='—'",
                    timeout=60000)
            except Exception:
                pass
            pg.wait_for_timeout(1500)               # let bars/overlays settle
            full = outdir / f"review_{label}.png"
            pg.screenshot(path=str(full), full_page=True)
            shots.append((label, full))
            # close-up of the time-aligned audio panel (waveform + spectrogram)
            el = pg.query_selector("#viz-card")
            if el and el.is_visible():
                el.screenshot(path=str(outdir / f"closeup_{label}_audio.png"))
            # linked brushing: hover the 'isolated' log entry -> spans light up across views
            try:
                ent = pg.query_selector(".entry.brushable")
                if ent:
                    ent.hover()
                    pg.wait_for_timeout(400)
                    ncols = pg.eval_on_selector_all("#panel .brush-col", "els=>els.length")
                    if el and ncols:
                        el.screenshot(path=str(outdir / f"brush_{label}.png"))
                        print(f"    brushing: {ncols} span column(s) highlighted")
            except Exception as e:
                print(f"    brush check failed: {e}")
            # exercise playback if controls appeared: click play, capture the moving playhead
            ctrl = pg.query_selector("#controls")
            if ctrl and ctrl.is_visible():
                try:
                    pg.click("#play")
                    pg.wait_for_timeout(900)
                    if el:
                        el.screenshot(path=str(outdir / f"playing_{label}.png"))
                    playing = pg.eval_on_selector("#play", "b=>b.classList.contains('playing')")
                    print(f"    playback: controls shown, playing={playing}")
                except Exception as e:
                    print(f"    playback check failed: {e}")
            # diagnosis grounding: hover the evidence line -> mechanical spans light up above
            try:
                evd = pg.query_selector("#evidence")
                if evd and evd.is_visible():
                    evd.hover()
                    pg.wait_for_timeout(350)
                    pg.screenshot(path=str(outdir / f"ground_{label}.png"), full_page=True)
            except Exception:
                pass
            verdict = pg.text_content("#verdict") or "?"
            band = pg.text_content("#band") or ""
            print(f"  {label:16} verdict={verdict:9} triage[{band}] -> {full.name}")
            pg.close()
        b.close()
    print(f"\n{len(shots)} full-page screenshots in {outdir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/tmp/ui_review")
    ap.add_argument("--model", default=str(ROOT / "models"))
    ap.add_argument("--url", default=None, help="review a running server instead of starting one")
    a = ap.parse_args()
    outdir = Path(a.out)
    clips = build_clips(outdir)
    print(f"built {len(clips)} review clips")

    proc = None
    url = a.url
    if not url:
        port = _free_port()
        url = f"http://127.0.0.1:{port}"
        env = {**os.environ, "CARDIAG_MODEL": str(Path(a.model) / "best_model_clap.joblib"),
               "CARDIAG_TRIAGE": str(Path(a.model) / "triage_model.joblib")}
        proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "cardiag.web.app:app",
                                 "--port", str(port), "--log-level", "warning"], env=env)
        for _ in range(40):
            try:
                import urllib.request
                urllib.request.urlopen(url + "/health", timeout=1)
                break
            except Exception:
                time.sleep(0.5)
    try:
        review(clips, url, outdir)
    finally:
        if proc:
            proc.terminate()


if __name__ == "__main__":
    main()
