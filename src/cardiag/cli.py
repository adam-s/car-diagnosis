"""``cardiag`` command-line interface — a thin shell over the public API.

    cardiag diagnose clip.wav            # full model: verdict + knock + causes
    cardiag triage   clip.wav            # calibrated engine-vs-running-gear
    cardiag clean    clip.wav            # isolate mechanical sound (no model)
    cardiag serve                        # local web upload app
    cardiag scrape   youtube|tiktok|reddit
    cardiag train

Every command imports its heavy dependencies lazily, so `cardiag --help` and the
scraping commands don't pay for torch/CLAP.
"""
from __future__ import annotations

import json
from typing import Optional

import typer

app = typer.Typer(
    add_completion=False,
    help="Diagnose a car's mechanical fault from the sound it makes.",
    no_args_is_help=True,
)


def _print_diagnosis(d) -> None:
    from rich.console import Console
    c = Console()
    c.print(f"\n  [bold]{d.file}[/bold]")
    c.print("  " + "─" * 52)
    color = {"fault": "red", "normal": "green", "uncertain": "yellow"}[d.verdict.value]
    c.print(f"  Verdict: [bold {color}]{d.verdict.value.upper()}[/bold {color}]  "
            f"(fault p={d.fault_probability:.2f})")
    if d.engine_knock_probability >= 0.5:
        c.print(f"  [bold red]⚠ ENGINE KNOCK likely "
                f"(p={d.engine_knock_probability:.2f})[/bold red] — check oil")
    c.print("  Most likely cause (audio is suggestive, not definitive):")
    for cause in d.causes:
        bar = "█" * int(cause.p * 20)
        c.print(f"    {cause.p:4.0%} [cyan]{bar:<20}[/cyan] {cause.part:14} {cause.note}")
    if d.segments:
        kept = sum(s.duration for s in d.segments)
        c.print(f"  Isolated {len(d.segments)} mechanical span(s), {kept:.1f}s total.")
    c.print("  " + "─" * 52)
    c.print(f"  [dim]{d.note}[/dim]\n")


@app.command()
def diagnose(
    audio: str = typer.Argument(..., help="Path to an audio file."),
    model: Optional[str] = typer.Option(None, help="Path to best_model_clap.joblib."),
    no_clean: bool = typer.Option(False, "--no-clean", help="Skip the cleaning cascade."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
):
    """Diagnose a recording with the full model (fault / knock / cause)."""
    from cardiag import Classifier
    clf = Classifier.load(model)
    result = clf.diagnose(audio, clean_audio=not no_clean)
    if as_json:
        print(json.dumps(result.to_dict(), indent=1))
    else:
        _print_diagnosis(result)


@app.command()
def triage(
    audio: str = typer.Argument(..., help="Path to an audio file."),
    model: Optional[str] = typer.Option(None, help="Path to triage_model.joblib."),
    as_json: bool = typer.Option(False, "--json"),
):
    """Coarse engine-vs-running-gear call with a calibrated confidence band."""
    from cardiag import TriageClassifier
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


@app.command()
def clean(
    audio: str = typer.Argument(..., help="Path to an audio file."),
    no_music_gate: bool = typer.Option(False, "--no-music-gate"),
    out: Optional[str] = typer.Option(None, help="Write isolated audio to this WAV."),
):
    """Isolate the mechanical sound (remove music / voice / static). No model."""
    from cardiag import clean as clean_fn
    res = clean_fn(audio, music_gate=not no_music_gate)
    print(json.dumps(res.to_dict(), indent=1))
    if out and not res.is_empty:
        import soundfile as sf
        sf.write(out, res.merged_audio(), res.sr)
        print(f"wrote {res.kept_seconds}s of isolated audio -> {out}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    model: Optional[str] = typer.Option(None),
):
    """Launch the local web upload app."""
    import os
    if model:
        os.environ["CARDIAG_MODEL"] = model
    import uvicorn
    uvicorn.run("cardiag.web.app:app", host=host, port=port)


@app.command()
def scrape(
    platform: str = typer.Argument("youtube", help="youtube | tiktok | reddit"),
    per_query: int = typer.Option(3, help="(youtube) videos per search query."),
    max_videos: int = typer.Option(40, help="(youtube) cap on videos processed."),
    pages: int = typer.Option(12, help="(reddit) pages per feed."),
):
    """Discover, download, and clean fault-sound clips into a labeled corpus.

    YouTube is the self-contained reference path (no LLM, no external data):
    it discovers fault+normal videos, runs the cleaning cascade + CLAP, and
    writes data/youtube/corpus.jsonl ready for `cardiag train`.
    """
    if platform == "youtube":
        from cardiag.pipeline import build
        build.scrape_youtube(per_query=per_query, max_videos=max_videos)
    elif platform == "tiktok":
        from cardiag.ingest.tiktok import discover
        discover.main()
    elif platform == "reddit":
        from cardiag.ingest.reddit import scrape as rscrape
        rscrape.main(pages)
    else:
        raise typer.BadParameter("platform must be youtube, tiktok, or reddit")


@app.command()
def train(min_class: int = typer.Option(2, help="Min samples per class to keep.")):
    """Embed the scraped corpus with CLAP and train the fault/knock/cause +
    triage models into data/training/."""
    from cardiag.pipeline import build
    build.train(min_class=min_class)


@app.command()
def demo(
    per_query: int = typer.Option(1, help="Videos per query (keep small)."),
    max_videos: int = typer.Option(18, help="Cap on videos processed."),
):
    """The whole loop from nothing: scrape -> clean -> train -> diagnose.

    For a fresh clone with no data and no model. Takes a few minutes (downloads
    a handful of clips + the CLAP weights on first run)."""
    from cardiag.pipeline import build
    build.demo(per_query=per_query, max_videos=max_videos)


@app.command()
def version():
    """Print the installed version."""
    import cardiag
    print(cardiag.__version__)


if __name__ == "__main__":
    app()
