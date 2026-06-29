"""``cardiag doctor``: preflight checks with fix-it instructions.

A novice should never meet a raw stack trace. This probes everything the pipeline
needs (Python, ffmpeg, yt-dlp, CLAP weights, Camoufox + its Firefox, the core ML
deps, disk, a trained model) and prints, for anything missing, the exact command
to fix it.

    cardiag doctor
"""
from __future__ import annotations

import shutil
import sys
from importlib import import_module

OK, WARN, BAD = "ok", "warn", "bad"


def _check_python():
    v = sys.version_info
    if (v.major, v.minor) == (3, 11):
        return OK, f"Python {v.major}.{v.minor}.{v.micro}", ""
    return BAD, f"Python {v.major}.{v.minor} (need 3.11)", \
        "create a 3.11 venv:  uv venv --python 3.11 && source .venv/bin/activate"


def _check_cmd(name, fix):
    path = shutil.which(name)
    return (OK, path, "") if path else (BAD, "not found", fix)


def _check_import(mod, extra):
    try:
        import_module(mod)
        return OK, "installed", ""
    except Exception:
        fix = f"pip install -e '.[{extra}]'" if extra else "pip install -e ."
        return BAD, "missing", fix


def _check_web():
    try:
        import_module("fastapi")
        import_module("uvicorn")
        return OK, "installed", ""
    except Exception:
        return WARN, "not installed", "needed for `cardiag serve`: pip install -e '.[web]'"


def _check_clap_cache():
    from pathlib import Path
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    hit = list(hub.glob("*clap-htsat-unfused*")) if hub.exists() else []
    if hit:
        return OK, "CLAP weights cached", ""
    return WARN, "CLAP not cached", \
        "downloads ~2GB from Hugging Face on first diagnose/train (then cached)"


def _check_camoufox():
    try:
        import camoufox  # noqa: F401
    except Exception:
        return WARN, "camoufox not installed", \
            "needed for TikTok/Reddit scraping:  pip install -e '.[scrape]'"
    # is the Firefox fetched?
    try:
        from camoufox.pkgman import installed_verstr
        installed_verstr()
        return OK, "camoufox + Firefox ready", ""
    except Exception:
        return WARN, "camoufox installed, Firefox not fetched", \
            "python -m camoufox fetch"


def _check_playwright():
    from importlib.metadata import PackageNotFoundError, version
    try:
        ver = version("playwright")
    except PackageNotFoundError:
        return WARN, "playwright missing", "pip install -e '.[scrape]'"
    if ver.startswith("1.51"):
        return OK, f"playwright {ver}", ""
    return WARN, f"playwright {ver} (Camoufox needs 1.51.0)", \
        "pip install 'playwright==1.51.0'"


def _check_model():
    from cardiag import paths
    if paths.MODEL_CLAP.exists():
        return OK, f"model at {paths.MODEL_CLAP}", ""
    shipped = paths.SHIPPED_DIR / "best_model_clap.joblib"
    if shipped.exists():
        return OK, "using the shipped pre-trained model in models/", ""
    return WARN, "no trained model yet", \
        "train one fast:  cardiag train --fixtures   (offline), or  cardiag demo"


def _check_disk():
    import shutil as sh

    from cardiag import paths
    base = paths.DATA if paths.DATA.exists() else paths.REPO_ROOT
    free_gb = sh.disk_usage(base).free / 1e9
    if free_gb > 5:
        return OK, f"{free_gb:.0f} GB free", ""
    return WARN, f"only {free_gb:.1f} GB free", "scraping + CLAP need a few GB"


CHECKS = [
    ("python", _check_python),
    ("ffmpeg", lambda: _check_cmd(
        "ffmpeg", "install ffmpeg (brew install ffmpeg / apt install ffmpeg)")),
    ("yt-dlp", lambda: _check_cmd("yt-dlp", "pip install -e '.[scrape]'")),
    ("torch/transformers", lambda: _check_import("torch", "")),
    ("librosa", lambda: _check_import("librosa", "")),
    ("scikit-learn", lambda: _check_import("sklearn", "")),
    ("CLAP weights", _check_clap_cache),
    ("camoufox (scraping)", _check_camoufox),
    ("playwright pin", _check_playwright),
    ("matplotlib (inspect)", lambda: _check_import("matplotlib", "viz")),
    ("fastapi (serve)", _check_web),
    ("trained model", _check_model),
    ("disk space", _check_disk),
]


def run() -> int:
    """Run all checks, print a report, return the number of hard failures."""
    from rich.console import Console
    c = Console()
    glyph = {OK: "[green]✓[/green]", WARN: "[yellow]·[/yellow]", BAD: "[red]✗[/red]"}
    c.print("\n[bold]cardiag doctor[/bold] — environment preflight\n")
    bad = 0
    for name, fn in CHECKS:
        try:
            status, detail, fix = fn()
        except Exception as e:
            status, detail, fix = BAD, f"check errored: {type(e).__name__}", ""
        bad += status == BAD
        c.print(f"  {glyph[status]} [bold]{name:22}[/bold] {detail}")
        if fix and status != OK:
            c.print(f"      [dim]→ {fix}[/dim]")
    if bad:
        c.print(f"\n[red]{bad} blocking issue(s).[/red] Fix the ✗ items above, "
                f"then re-run [bold]cardiag doctor[/bold].\n")
    else:
        c.print("\n[green]All set.[/green] Next steps, in order:")
        c.print("  [dim]1.[/dim] [bold]cardiag train --fixtures[/bold]      "
                "[dim]# a model in ~2s, offline[/dim]")
        c.print("  [dim]2.[/dim] [bold]cardiag inspect <clip.wav>[/bold]    "
                "[dim]# see + hear what the pipeline does[/dim]")
        c.print("  [dim]3.[/dim] [bold]cardiag demo[/bold]                  "
                "[dim]# the whole loop, scraping for real[/dim]\n")
    return bad


if __name__ == "__main__":
    raise SystemExit(run())
