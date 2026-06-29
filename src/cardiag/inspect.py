"""``cardiag inspect``: render the pipeline as a self-contained HTML report.

For each clip it shows, side by side, *what the pipeline did and why*:
  * the original waveform with the kept mechanical spans highlighted,
  * a mel-spectrogram of the original and of the isolated audio,
  * the cleaning cascade's decisions (how much speech / static / music it
    dropped, how many spans it kept),
  * the CLAP zero-shot score for each sound-type prompt (so you see *why* it was
    labeled), and the music-gate scores,
  * audio players for the original and the isolated signal, to listen to the
    before/after.

Everything (images + audio) is inlined as base64, so the output is one HTML file
you can open in a browser or hand to someone. This is the feature that turns the
black box into something a beginner can see and hear.

    cardiag inspect clip1.wav clip2.wav -o report.html
    cardiag inspect --sample 8 -o report.html      # sample from data/*/clips
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np

from cardiag import config
from cardiag.audio.clean import clean


def _b64_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=90)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _b64_wav(y: np.ndarray, sr: int) -> str:
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


def _waveform_png(y, sr, segments):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 1.6))
    t = np.arange(len(y)) / sr
    ax.plot(t, y, lw=0.4, color="#58a6ff")
    for s in segments:
        ax.axvspan(s.start, s.end, color="#3fb950", alpha=0.25)
    ax.set_xlim(0, max(t[-1], 0.1))
    ax.set_yticks([])
    ax.set_xlabel("seconds (green = kept mechanical span)")
    fig.patch.set_facecolor("white")
    return _b64_png(fig)


def _melspec_png(y, sr, title):
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 1.8))
    if len(y) < 512:
        ax.text(0.5, 0.5, "(empty)", ha="center")
        ax.axis("off")
        return _b64_png(fig)
    m = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    librosa.display.specshow(librosa.power_to_db(m, ref=np.max), sr=sr,
                             x_axis="time", y_axis="mel", ax=ax, cmap="magma")
    ax.set_title(title, fontsize=9)
    return _b64_png(fig)


def _clap_scores(isolated, sr):
    """Top CLAP sound-type scores + music-gate scores for the isolated audio."""
    from cardiag.audio.clap import Clap
    from cardiag.audio.clean import _MUSIC_PROMPTS
    clap = Clap()
    l1 = clap.score(isolated, config.L1_PROMPTS, sr=sr).mean(0)
    music = clap.score(isolated, _MUSIC_PROMPTS, sr=sr).mean(0)
    l1_top = sorted(zip([p[2:] for p in config.L1_PROMPTS], l1),
                    key=lambda kv: -kv[1])[:4]
    return l1_top, list(zip(["music", "mechanical", "speech", "silence"], music))


def _bars(pairs) -> str:
    out = []
    for name, p in pairs:
        pct = int(float(p) * 100)
        out.append(
            f'<div class="bar-row"><span class="bl">{name}</span>'
            f'<span class="track"><span class="bar" style="width:{pct}%"></span></span>'
            f'<span class="bv">{pct}%</span></div>')
    return "".join(out)


def _clip_card(path, with_clap=True) -> str:
    import warnings
    from pathlib import Path

    import librosa
    if not Path(path).exists():
        raise FileNotFoundError(f"no such audio file: {path}")
    try:                                       # turn library decode errors into the
        with warnings.catch_warnings():        # friendly ValueError the CLI already catches
            warnings.simplefilter("ignore")
            y, sr = librosa.load(str(path), sr=config.SR_CLAP, mono=True)
    except Exception as exc:
        raise ValueError(f"could not read audio from {path} — is it a valid audio "
                         f"file? ({type(exc).__name__})") from None
    res = clean(str(path), music_gate=with_clap)
    iso = res.merged_audio()

    wave = _waveform_png(y, sr, res.segments)
    spec_full = _melspec_png(y, sr, "original (full)")
    spec_iso = _melspec_png(iso, sr, "isolated mechanical audio") if iso.size else ""

    clap_html = ""
    if with_clap and res.isolated:
        l1_top, music = _clap_scores(res.isolated, res.sr)
        clap_html = (
            f'<div class="col"><h4>CLAP sound-type (why this label)</h4>{_bars(l1_top)}'
            f'<h4>music gate</h4>{_bars(music)}</div>')

    decisions = (
        f"total {res.total_seconds:.1f}s · kept {res.kept_seconds:.1f}s in "
        f"{len(res.segments)} span(s) · dropped speech≈"
        f"{res.speech_fraction*100:.0f}% · music score {res.music_probability:.2f}"
        f"{' · MUSIC (dropped)' if res.is_music else ''}"
        f"{' · nothing isolated → whole-clip fallback' if res.is_empty else ''}")

    audio_iso = (f'<div>isolated:<br><audio controls src="{_b64_wav(iso, sr)}"></audio></div>'
                 if iso.size else "<div>isolated: (none)</div>")
    return f"""
    <div class="card">
      <div class="hd">{Path(path).name}</div>
      <div class="dec">{decisions}</div>
      <div class="row">
        <div class="col">
          <img src="{wave}"><img src="{spec_full}">{f'<img src="{spec_iso}">' if spec_iso else ''}
          <div class="players">
            <div>original:<br><audio controls src="{_b64_wav(y, sr)}"></audio></div>
            {audio_iso}
          </div>
        </div>
        {clap_html}
      </div>
    </div>"""


def sample_clips(n: int) -> list[Path]:
    from cardiag import paths
    found: list[Path] = []
    for base in (paths.YT_DATA, paths.TT_DATA, paths.REDDIT_DATA):
        d = base / "clips"
        if d.exists():
            found += sorted(d.rglob("*.wav"))
    import random
    random.Random(0).shuffle(found)
    return found[:n]


def report(files, out_path="report.html", with_clap: bool = True) -> Path:
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError as e:
        raise SystemExit("cardiag inspect needs matplotlib: pip install -e "
                         f"'.[viz]'  ({e})")
    files = [Path(f) for f in files]
    if not files:
        raise SystemExit("no clips to inspect (pass files, or --sample N after a "
                         "scrape).")
    cards = "".join(_clip_card(f, with_clap=with_clap) for f in files)
    html = _PAGE.replace("{{CARDS}}", cards).replace("{{N}}", str(len(files)))
    out = Path(out_path)
    out.write_text(html)
    print(f"wrote inspection report for {len(files)} clip(s) -> {out.resolve()}")
    return out


def gallery(rows=None, out_path="gallery.html", limit: int = 120) -> Path:
    """A compact, audio-playable grid of the scraped corpus grouped by sound-type,
    so you can listen to clips and judge the labels yourself."""
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError as e:
        raise SystemExit(f"gallery needs matplotlib: pip install -e '.[viz]'  ({e})")
    import librosa

    from cardiag import config
    if rows is None:
        from cardiag.pipeline import build
        rows = build.load_corpus()
    if not rows:
        raise SystemExit("empty corpus — run a scrape first (cardiag scrape …).")
    rows = rows[:limit]

    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(r.get("l1", "?"), []).append(r)

    sections = []
    for l1 in sorted(groups):
        cards = []
        for r in groups[l1]:
            y, sr = librosa.load(r["wav"], sr=config.SR_CLAP, mono=True)
            cause = ", ".join(r.get("l2_candidates", [])) or "—"
            cards.append(
                f'<div class="gc"><img src="{_melspec_png(y, sr, "")}">'
                f'<div class="gl"><b>{r.get("kind","")}</b> · {cause}</div>'
                f'<audio controls src="{_b64_wav(y, sr)}"></audio></div>')
        sections.append(f'<h2>{l1} <span class="cnt">{len(groups[l1])}</span></h2>'
                        f'<div class="grid">{"".join(cards)}</div>')
    html = _GALLERY.replace("{{SECTIONS}}", "".join(sections)).replace(
        "{{N}}", str(len(rows)))
    out = Path(out_path)
    out.write_text(html)
    print(f"wrote gallery of {len(rows)} clips ({len(groups)} sound-types) "
          f"-> {out.resolve()}")
    return out


_GALLERY = """<!doctype html><html><head><meta charset="utf-8">
<title>cardiag — corpus gallery</title><style>
 body{background:#0d1117;color:#e6edf3;font:14px system-ui,sans-serif;margin:0;padding:24px}
 h1{font-size:22px} h2{font-size:15px;color:#58a6ff;margin:22px 0 8px;border-bottom:1px solid #30363d;padding-bottom:4px}
 .cnt{color:#8b949e;font-weight:400} .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}
 .gc{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:8px}
 .gc img{width:100%;border-radius:4px;background:#fff} .gl{font-size:12px;color:#8b949e;margin:4px 0}
 audio{width:100%;height:30px}
</style></head><body>
 <h1>cardiag — corpus gallery ({{N}} clips)</h1>
 <p style="color:#8b949e">Clips grouped by CLAP sound-type. Each shows its kind + cause-from-text labels.
 Play them and judge the labels yourself.</p>
 {{SECTIONS}}
</body></html>"""


_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>cardiag — pipeline inspection</title><style>
 body{background:#0d1117;color:#e6edf3;font:14px system-ui,sans-serif;margin:0;padding:24px}
 h1{font-size:22px} h4{margin:10px 0 4px;color:#8b949e;font-size:12px;text-transform:uppercase}
 .card{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:16px;margin:16px 0}
 .hd{font-weight:700;font-size:16px} .dec{color:#8b949e;font-size:12px;margin:4px 0 10px}
 .row{display:flex;gap:18px;flex-wrap:wrap} .col{flex:1;min-width:320px}
 img{max-width:100%;display:block;border-radius:6px;margin:4px 0;background:#fff}
 audio{width:280px;height:32px} .players{display:flex;gap:16px;margin-top:8px;flex-wrap:wrap}
 .bar-row{display:flex;align-items:center;gap:8px;margin:3px 0}
 .bl{width:90px;color:#8b949e;font-size:12px} .bv{width:36px;text-align:right;font-variant-numeric:tabular-nums}
 .track{flex:1;height:9px;background:#21262d;border-radius:5px;overflow:hidden}
 .bar{display:block;height:9px;background:#58a6ff}
</style></head><body>
 <h1>cardiag — pipeline inspection ({{N}} clips)</h1>
 <p style="color:#8b949e">Green spans = mechanical audio the cascade kept. Spectrograms show
 original vs isolated. CLAP bars show why each label was chosen. Press play to hear before/after.</p>
 {{CARDS}}
</body></html>"""
