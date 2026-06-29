"""``cardiag`` command-line interface: a thin shell over the public API.

    cardiag diagnose clip.wav            # full model: verdict + knock + causes
    cardiag triage   clip.wav            # calibrated engine-vs-running-gear
    cardiag clean    clip.wav            # isolate mechanical sound (no model)
    cardiag serve                        # local web upload app
    cardiag scrape   youtube|tiktok      # (reddit deprecated, too noisy)
    cardiag train

Every command imports its heavy dependencies lazily, so `cardiag --help` and the
scraping commands don't pay for torch/CLAP.
"""
from __future__ import annotations

import json
from contextlib import contextmanager

import typer

app = typer.Typer(
    add_completion=False,
    help="Diagnose a car's mechanical fault from the sound it makes.\n\n"
         "New here? Run [bold]cardiag doctor[/bold] — it checks your setup and "
         "points you to the next step.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


def _nudge(*lines: str) -> None:
    """Print a subtle 'what next' hint: the gentle hand-off to the next step.

    Goes to stderr so it never pollutes machine-readable stdout (e.g. `--json`)."""
    from rich.console import Console
    c = Console(stderr=True)
    for ln in lines:
        c.print(f"  [dim]→ {ln}[/dim]")


@contextmanager
def _friendly_errors():
    """Turn expected failures (missing/unreadable audio, missing/invalid model)
    into a clean one-line message + exit code 1, never a scary traceback."""
    from rich.console import Console
    try:
        yield
    except (FileNotFoundError, ValueError, IsADirectoryError, OSError) as e:
        Console(stderr=True).print(f"[yellow]{e}[/yellow]")
        raise typer.Exit(1) from None


def _print_diagnosis(d) -> None:
    from rich.console import Console
    c = Console()
    c.print(f"\n  [bold]{d.file}[/bold]")
    c.print("  " + "─" * 52)
    color = {"fault": "red", "normal": "green", "uncertain": "yellow"}[d.verdict.value]
    c.print(f"  Verdict: [bold {color}]{d.verdict.value.upper()}[/bold {color}]  "
            f"(fault p={d.fault_probability:.2f})")
    if d.regions:
        c.print("  Where in the car (right zone in top-3 ≈75% out-of-sample):")
        for r in d.regions:
            bar = "█" * int(r.p * 20)
            c.print(f"    {r.p:4.0%} [green]{bar:<20}[/green] {r.zone}")
    c.print("  Most likely part — ranked shortlist (suggestive, not definitive):")
    for cause in d.causes:
        bar = "█" * int(cause.p * 20)
        c.print(f"    {cause.p:4.0%} [cyan]{bar:<20}[/cyan] {cause.part:14} {cause.note}")
    if d.engine_knock_probability >= 0.5:
        c.print(f"  [yellow]engine-knock hint p={d.engine_knock_probability:.2f}[/yellow] "
                f"[dim](in-distribution only — did not generalize out-of-sample)[/dim]")
    if d.segments:
        kept = sum(s.duration for s in d.segments)
        c.print(f"  Isolated {len(d.segments)} mechanical span(s), {kept:.1f}s total.")
    c.print("  " + "─" * 52)
    c.print(f"  [dim]{d.note}[/dim]\n")


@app.command()
def diagnose(
    audio: str = typer.Argument(..., help="Path to an audio file."),
    model: str | None = typer.Option(None, help="Path to best_model_clap.joblib."),
    no_clean: bool = typer.Option(False, "--no-clean", help="Skip the cleaning cascade."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
):
    """Diagnose a recording with the full model (fault / knock / cause)."""
    from cardiag import Classifier
    with _friendly_errors():
        clf = Classifier.load(model)
        result = clf.diagnose(audio, clean_audio=not no_clean)
    if as_json:
        print(json.dumps(result.to_dict(), indent=1))
    else:
        _print_diagnosis(result)
        _nudge(f"see *why* (spectrograms + scores):  cardiag inspect {audio} -o report.html",
               f"coarse calibrated call:  cardiag triage {audio}")


@app.command()
def triage(
    audio: str = typer.Argument(..., help="Path to an audio file."),
    model: str | None = typer.Option(None, help="Path to triage_model.joblib."),
    as_json: bool = typer.Option(False, "--json"),
):
    """Coarse engine-vs-running-gear call with a calibrated confidence band."""
    from cardiag import TriageClassifier
    with _friendly_errors():
        result = TriageClassifier.load(model).triage(audio)
    if as_json:
        print(json.dumps(result.to_dict(), indent=1))
        return
    print(f"\n  {result.file}\n  " + "─" * 56)
    if result.band.value == "abstain":
        print(f"  CAN'T TELL from this audio "
              f"({result.probabilities}).")
    else:
        print(f"  {result.label}")
        print(f"  Confidence: {result.band.value.upper()} ({result.confidence:.0%}) "
              f"— {result.band_gloss}")
        print(f"  Next: {result.next_step}")
    print("  " + "─" * 56 + "\n")
    _nudge(f"full diagnosis with causes:  cardiag diagnose {audio}")


@app.command()
def clean(
    audio: str = typer.Argument(..., help="Path to an audio file."),
    no_music_gate: bool = typer.Option(False, "--no-music-gate"),
    out: str | None = typer.Option(None, help="Write isolated audio to this WAV."),
):
    """Isolate the mechanical sound (remove music / voice / static). No model."""
    import sys

    from cardiag import clean as clean_fn
    with _friendly_errors():
        res = clean_fn(audio, music_gate=not no_music_gate)
    print(json.dumps(res.to_dict(), indent=1))      # stdout: machine-readable
    if out and not res.is_empty:
        import soundfile as sf
        sf.write(out, res.merged_audio(), res.sr)
        print(f"wrote {res.kept_seconds}s of isolated audio -> {out}", file=sys.stderr)
    _nudge(f"diagnose what was isolated:  cardiag diagnose {audio}",
           f"see it visually:  cardiag inspect {audio} -o report.html")


@app.command()
def inspect(
    audio: list[str] = typer.Argument(None, help="Clip(s) to inspect."),
    sample: int = typer.Option(0, min=0, max=200,
                               help="Instead, sample N clips from data/*/clips."),
    out: str = typer.Option("report.html", "-o", "--out", help="Output HTML file."),
    no_clap: bool = typer.Option(False, "--no-clap", help="Skip CLAP scores (faster)."),
):
    """Render a self-contained HTML report showing what the pipeline does to each
    clip: isolated spans, spectrograms, CLAP scores, and before/after audio."""
    from cardiag import inspect as inspect_mod
    files = list(audio or [])
    if sample:
        files += [str(p) for p in inspect_mod.sample_clips(sample)]
    with _friendly_errors():
        p = inspect_mod.report(files, out_path=out, with_clap=not no_clap)
    _nudge(f"open it in your browser:  open {p}   (file://{p})")


@app.command()
def gallery(
    out: str = typer.Option("gallery.html", "-o", "--out"),
    limit: int = typer.Option(120, min=1, max=2000, help="Max clips to include."),
):
    """Render an audio-playable grid of the scraped corpus grouped by sound-type,
    so you can listen to clips and judge the labels yourself."""
    from cardiag import inspect as inspect_mod
    with _friendly_errors():
        p = inspect_mod.gallery(out_path=out, limit=limit)
    _nudge(f"open it in your browser:  open {p}   (file://{p})")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    model: str | None = typer.Option(None),
):
    """Launch the local web upload app (binds 127.0.0.1; no auth).

    Pass --model to use a pre-trained model: either a directory containing
    best_model_clap.joblib + triage_model.joblib (e.g. the shipped `models/`), or
    the best_model_clap.joblib file itself. Without it, the app uses the model in
    data/training/ from your own `cardiag train` (or none)."""
    import os
    from pathlib import Path
    if model:
        p = Path(model)
        clap = p / "best_model_clap.joblib" if p.is_dir() else p
        triage = clap.parent / "triage_model.joblib"
        if not clap.exists():
            from rich.console import Console
            Console(stderr=True).print(f"[red]no model at {clap}[/red]")
            raise typer.Exit(code=1)
        from cardiag import Classifier
        try:                                   # fail fast at launch, not mid-request
            Classifier.load(str(clap))
        except (ValueError, FileNotFoundError) as e:
            from rich.console import Console
            Console(stderr=True).print(f"[red]{e}[/red]")
            raise typer.Exit(code=1) from None
        os.environ["CARDIAG_MODEL"] = str(clap)
        if triage.exists():
            os.environ["CARDIAG_TRIAGE"] = str(triage)
    if host not in ("127.0.0.1", "localhost", "::1"):
        from rich.console import Console
        Console(stderr=True).print(
            f"[bold red]⚠ exposing cardiag on {host} with NO authentication[/bold red] "
            f"— anyone who can reach this host can upload files and run inference.")
    try:
        import uvicorn
    except ImportError:
        raise typer.Exit(code=_missing("cardiag serve", "web")) from None
    uvicorn.run("cardiag.web.app:app", host=host, port=port)


def _missing(what: str, extra: str) -> int:
    from rich.console import Console
    Console(stderr=True).print(
        f"[yellow]{what} needs the [{extra}] extra: "
        f"pip install -e '.[{extra}]'[/yellow]")
    return 1


@app.command()
def scrape(
    platform: str = typer.Argument("youtube", help="youtube | tiktok"),
    per_query: int = typer.Option(3, min=1, help="(youtube) videos per search query."),
    max_videos: int = typer.Option(40, min=1, help="(youtube/tiktok) cap on videos."),
    pages: int = typer.Option(2, min=1, help="(reddit, deprecated) pages per feed."),
    normal: bool = typer.Option(
        False, "--normal",
        help="(tiktok) scrape HEALTHY-engine clips labeled 'normal' instead of "
             "fault — run this AND the default fault pass to give train both "
             "classes and break the source confound (docs/MODEL_CARD.md)."),
):
    """Discover, download, and clean sound clips into a labeled corpus.

    YouTube and TikTok funnel through one cleaning + CLAP-labeling path and write
    data/<platform>/corpus.jsonl ready for `cardiag train` (no LLM, no external
    data). YouTube runs both fault and normal queries; TikTok adds fault clips by
    default, or normal clips with --normal. TikTok needs the stealth browser
    (`python -m camoufox fetch`).

    Reddit is DEPRECATED as a training source: posts are long, uncurated, and
    off-target, which adds label noise without improving accuracy. The real-world
    input is a *targeted* clip (a user recording the symptom), so we train on the
    cleaner YouTube/TikTok signal and let the cleaning cascade handle messy input
    at inference time.
    """
    from cardiag.pipeline import build
    if platform == "youtube":
        build.scrape_youtube(per_query=per_query, max_videos=max_videos)
    elif platform == "reddit":
        from rich.console import Console
        Console(stderr=True).print(
            "[yellow]reddit is deprecated as a training source (uncurated/noisy — "
            "it adds label noise without helping). Use youtube/tiktok. Proceeding "
            "anyway since you asked explicitly…[/yellow]")
        build.scrape_reddit(pages=pages, max_posts=max_videos)
    elif platform == "tiktok":
        build.scrape_tiktok(max_videos=max_videos, kind="normal" if normal else "fault")
    else:
        raise typer.BadParameter("platform must be youtube or tiktok "
                                 "(reddit is deprecated)")
    _nudge("audit what you collected:  cardiag gallery -o gallery.html",
           "scrape the other sources too, then:  cardiag train")


@app.command()
def ingest(
    audio_dir: str = typer.Argument(..., help="Folder of audio files (any length)."),
    kind: str = typer.Option(..., help="fault | normal — the label for these clips."),
    cause: str | None = typer.Option(None, help="Part/fault family, e.g. wheel_bearing."),
    source: str = typer.Option("local", help="Corpus bucket name (data/<source>/)."),
):
    """Bring your own audio. Segment a folder of recordings (ANY length) through the
    SAME cascade as scraping (long recordings become multiple short mechanical
    spans) and add them to the corpus, ready for `cardiag train`. One coherent
    path: your clips are processed identically to scraped clips and to inference."""
    if kind not in ("fault", "normal"):
        raise typer.BadParameter("kind must be 'fault' or 'normal'")
    from cardiag.pipeline import build
    with _friendly_errors():
        build.ingest_dir(audio_dir, kind=kind, cause=cause, source=source)
    _nudge("see what was isolated:  cardiag gallery -o gallery.html",
           "train on it:  cardiag train")


@app.command()
def train(
    min_class: int = typer.Option(2, help="Min samples per class to keep."),
    fixtures: bool = typer.Option(False, "--fixtures",
                                  help="Train OFFLINE on bundled fixture "
                                       "embeddings (no scrape, no CLAP download)."),
    prune_noisy: float = typer.Option(
        0.0, "--prune-noisy", min=0.0, max=0.5,
        help="Confident-learning label cleaning: drop this fraction (e.g. 0.15) of "
             "the likely-mislabeled clips per source before fitting. Measured "
             "~+0.05 balanced accuracy on the fault/triage heads (docs/MODEL_CARD.md)."),
):
    """Embed the scraped corpus with CLAP and train the fault/knock/cause +
    triage models into data/training/. Use --fixtures to train instantly offline
    on the bundled sample, to learn the flow before a real scrape."""
    from cardiag.pipeline import build
    if fixtures:
        build.train_from_fixtures(min_class=min_class)
        _nudge("now diagnose a clip:  cardiag diagnose <clip.wav>",
               "ready for a real model? scrape:  cardiag scrape youtube")
    else:
        build.train(min_class=min_class, prune_noisy=prune_noisy)
        _nudge("diagnose a clip:  cardiag diagnose <clip.wav>",
               "audit the corpus:  cardiag gallery -o gallery.html")


@app.command()
def start():
    """First-run onboarding in one command: check the environment, train a quick
    offline model if you don't have one, and tell you exactly what to do next."""
    from rich.console import Console

    from cardiag import doctor as doc
    from cardiag import paths
    from cardiag.pipeline import build
    c = Console()
    c.print("\n[bold]Welcome to cardiag.[/bold] Getting you set up…\n")
    c.print("[bold]1/2[/bold]  environment check")
    if doc.run():
        c.print("[yellow]Fix the ✗ items above, then run [bold]cardiag start[/bold] "
                "again.[/yellow]")
        raise typer.Exit(1)
    c.print("[bold]2/3[/bold]  a model to diagnose with")
    if paths.resolve_clap().exists():
        where = ("your trained model" if paths.MODEL_CLAP.exists()
                 else "the shipped pre-trained model")
        c.print(f"  [green]✓[/green] using {where}")
    else:
        c.print("  training a quick model from bundled fixtures (offline, ~2s)…")
        build.train_from_fixtures()

    c.print("[bold]3/3[/bold]  diagnosing the bundled demo clip (a synthetic engine knock)")
    try:
        from cardiag import Classifier
        d = Classifier.load().diagnose(str(paths.DEMO_CLIP)).to_dict()
        c.print(f"  [green]✓[/green] verdict [bold]{d['verdict']}[/bold] · "
                f"engine-knock p={d['engine_knock_probability']:.2f} — the loop works end to end.")
    except Exception as e:
        c.print(f"  [yellow]could not run the demo clip ({type(e).__name__})[/yellow]")

    c.print("\n[green]You're ready.[/green] Try, in order:")
    c.print(f"  [bold]cardiag diagnose {paths.DEMO_CLIP.name}[/bold]   (or any clip of your own)")
    c.print("  [bold]cardiag serve --model models[/bold]   the live web app — paste a link or "
            "drop a clip")
    c.print("  [bold]cardiag demo[/bold]                   the whole loop, scraping for real\n")


@app.command()
def demo(
    per_query: int = typer.Option(1, min=1, help="Videos per query (keep small)."),
    max_videos: int = typer.Option(18, min=1, help="Cap on videos processed."),
):
    """The whole loop from nothing: scrape -> clean -> train -> diagnose.

    For a fresh clone with no data and no model. Takes a few minutes (downloads
    a handful of clips + the CLAP weights on first run)."""
    from cardiag.pipeline import build
    build.demo(per_query=per_query, max_videos=max_videos)
    _nudge("see/hear what happened:  cardiag inspect --sample 6 -o report.html",
           "browse your corpus:  cardiag gallery -o gallery.html",
           "scale up for a better model:  cardiag scrape youtube --per-query 5 --max-videos 200")


@app.command()
def doctor():
    """Preflight: check the environment and print fixes for anything missing."""
    from cardiag import doctor as doc
    raise typer.Exit(code=1 if doc.run() else 0)


@app.command()
def version():
    """Print the installed version."""
    import cardiag
    print(cardiag.__version__)


if __name__ == "__main__":
    app()
