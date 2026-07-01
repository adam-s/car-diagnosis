"""Instrumented pipeline: stream what happens to a clip, stage by stage.

``explain()`` runs the same cascade the corpus and inference use, but yields an
event after each stage so a UI can animate the cleaning live: the waveform, the
energy gate, Silero VAD speech, the spectral-flatness static test, the surviving
mechanical spans, the CLAP music gate, a log-mel spectrogram, and finally the
calibrated diagnosis. The authoritative regions still come from the tested
:func:`cardiag.audio.cascade.candidate_regions`; the per-stage overlays are
computed alongside purely for visualization.

Events are ``(name, payload)`` tuples; the web layer serializes them as SSE.
"""
from __future__ import annotations

import logging
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from cardiag import config

# CLAP music-gate prompts + threshold (same as cardiag.audio.clean).
_MUSIC_PROMPTS = ["music or a song with a beat",
                  "a mechanical car noise or engine sound",
                  "a person talking", "silence or ambient noise"]
MUSIC_THRESH = 0.5


# ---------------------------------------------------------------- URL intake
# Only these media platforms may be fetched server-side. Anything else (internal
# hosts, cloud metadata (169.254.169.254), file://, localhost) is refused BEFORE
# yt-dlp ever sees the URL. This is the SSRF guard for the link-paste feature.
_ALLOWED_HOSTS = ("youtube.com", "youtu.be", "youtube-nocookie.com",
                  "tiktok.com", "reddit.com", "redd.it", "v.redd.it")
MAX_DL_BYTES = 60 * 1024 * 1024
MAX_DL_SECONDS = 900                      # refuse clips longer than 15 min


def platform_of(url: str) -> str:
    u = url.lower()
    if "tiktok" in u:
        return "tiktok"
    if "redd" in u:
        return "reddit"
    if "youtu" in u:
        return "youtube"
    return "link"


def _validate_url(url: str) -> None:
    """Reject anything that isn't a public http(s) link to an allowlisted media
    host whose name resolves only to public IPs. Defends the URL-fetch feature
    against SSRF (internal services, metadata endpoints, file://, DNS rebinding)."""
    import ipaddress
    import socket
    from urllib.parse import urlparse
    u = urlparse(url.strip())
    if u.scheme not in ("http", "https"):
        raise ValueError("only http(s) links are supported")
    host = (u.hostname or "").lower()
    if not host or not any(host == h or host.endswith("." + h) for h in _ALLOWED_HOSTS):
        raise ValueError("only YouTube, TikTok, and Reddit links are supported")
    try:
        addrs = {res[4][0] for res in socket.getaddrinfo(host, None)}
    except socket.gaierror:
        raise ValueError("could not resolve that link's host") from None
    for a in addrs:
        ip = ipaddress.ip_address(a)
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                or ip.is_multicast or ip.is_unspecified):
            raise ValueError("that link resolves to a non-public address") from None


def acquire_url(url: str, dest_dir: Path, timeout: int = 180) -> tuple[Path, str]:
    """Download audio from a YouTube/TikTok/Reddit URL to a wav via yt-dlp. The URL
    is SSRF-validated first. Returns (wav_path, title). Raises ValueError."""
    import shutil
    _validate_url(url)
    if not shutil.which("yt-dlp"):
        raise ValueError("yt-dlp is not installed (pip install -e '.[scrape]')")
    out = dest_dir / "clip.%(ext)s"
    title = ""
    try:                                       # best-effort title for the header
        title = subprocess.run(
            ["yt-dlp", "--no-warnings", "--no-playlist", "--get-title", "--", url],
            capture_output=True, text=True, timeout=45).stdout.strip()[:140]
    except Exception:
        pass
    try:
        subprocess.run(
            ["yt-dlp", "--no-warnings", "--no-playlist",
             "--max-filesize", str(MAX_DL_BYTES),
             "--match-filter", f"duration <? {MAX_DL_SECONDS}",   # '<?' passes if duration unknown
             "-f", "ba/b", "-x", "--audio-format", "wav",
             "--postprocessor-args", f"-ar {config.SR_CLAP} -ac 1",
             "-o", str(out), "--", url],
            check=True, capture_output=True, timeout=timeout)
    except subprocess.CalledProcessError as e:  # never echo yt-dlp stderr to the client (SSRF exfil channel)
        # ...but do log it server-side so a local operator can see the real reason.
        logging.getLogger("cardiag.web").warning(
            "yt-dlp failed for %s:\n%s", url,
            (e.stderr or b"").decode("utf-8", "replace").strip())
        raise ValueError("could not download audio from this link (too long, private, "
                         "or unavailable)") from None
    except subprocess.TimeoutExpired:
        raise ValueError("download timed out — try a shorter clip or upload the file") from None
    wavs = list(dest_dir.glob("clip.wav")) or list(dest_dir.glob("clip.*"))
    if not wavs:
        raise ValueError("yt-dlp produced no audio for this link")
    return wavs[0], title


# ------------------------------------------------------------- helpers
def _peaks(y: np.ndarray, cols: int = 900) -> list[list[float]]:
    """Min/max envelope of the waveform downsampled to ``cols`` columns."""
    n = len(y)
    if n == 0:
        return []
    step = max(1, n // cols)
    out = []
    for i in range(0, n, step):
        seg = y[i:i + step]
        if len(seg):
            out.append([round(float(seg.min()), 4), round(float(seg.max()), 4)])
    return out


def _mel(y: np.ndarray, sr: int, t_bins: int = 240, f_bins: int = 64) -> dict:
    """Log-mel spectrogram, downsampled + normalized 0..1 for a heatmap."""
    import librosa
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=f_bins, n_fft=2048,
                                       hop_length=512, fmax=sr // 2)
    S = librosa.power_to_db(S + 1e-10, ref=np.max)        # dB, peak at 0
    if S.shape[1] > t_bins:                                # downsample time
        idx = np.linspace(0, S.shape[1] - 1, t_bins).astype(int)
        S = S[:, idx]
    lo, hi = float(S.min()), float(S.max())
    Sn = (S - lo) / (hi - lo + 1e-9)                       # 0..1, low->high energy
    return {"mel": np.round(Sn, 3).tolist(), "f": S.shape[0], "t": S.shape[1]}


def _flatness(seg: np.ndarray) -> float:
    import librosa
    return float(np.mean(librosa.feature.spectral_flatness(y=seg)))


# ------------------------------------------------------------------ explain
def explain(path, *, source: str = "upload", title: str = "") -> Iterator[tuple[str, dict]]:
    """Yield ``(event, payload)`` tuples narrating the pipeline on ``path``."""
    import librosa

    from cardiag.audio.cascade import _STEP, candidate_regions

    t0 = time.time()
    try:
        y16, _ = librosa.load(str(path), sr=config.SR_CHEAP, mono=True)
        yhi, _ = librosa.load(str(path), sr=config.SR_CLAP, mono=True)
    except Exception as exc:
        yield "error", {"message": f"could not read audio ({type(exc).__name__})"}
        return
    y16 = np.nan_to_num(y16)
    yhi = np.nan_to_num(yhi)
    dur = len(y16) / config.SR_CHEAP
    if dur < 0.2:
        yield "error", {"message": "clip is too short to analyze"}
        return

    yield "meta", {"source": source, "title": title, "duration": round(dur, 2),
                   "sr": config.SR_CLAP}
    yield "waveform", {"peaks": _peaks(y16), "duration": round(dur, 2)}

    # --- stage 1: energy gate -------------------------------------------------
    hop = int(_STEP * config.SR_CHEAP)
    rms = librosa.feature.rms(y=y16, frame_length=hop * 2, hop_length=hop)[0]
    thr = max(0.005, float(np.percentile(rms, 20)) * 1.5)
    loud = rms > thr
    env = (rms / (rms.max() + 1e-9))
    keepcols = np.linspace(0, len(env) - 1, min(len(env), 900)).astype(int)
    yield "stage", {
        "id": "energy", "label": "Energy gate",
        "narration": f"RMS energy over {len(rms)} frames; {int(loud.mean()*100)}% "
                     f"clear the loudness floor (the quiet rest is dropped).",
        "envelope": np.round(env[keepcols], 3).tolist(),
        "loud_frac": round(float(loud.mean()), 3),
    }

    # --- stage 2: Silero VAD (speech removal) ---------------------------------
    try:
        from silero_vad import get_speech_timestamps

        from cardiag.audio.cascade import _vad
        speech = get_speech_timestamps(__import__("torch").from_numpy(y16), _vad(),
                                       sampling_rate=config.SR_CHEAP, return_seconds=True)
    except Exception:
        speech = []
    sp_spans = [[round(t["start"], 2), round(t["end"], 2)] for t in speech]
    sp_secs = sum(e - s for s, e in sp_spans)
    yield "stage", {
        "id": "vad", "label": "Speech removal (Silero VAD)",
        "narration": (f"Silero VAD found {len(sp_spans)} speech span(s) "
                      f"({sp_secs:.1f}s) — narration is removed so only mechanical "
                      f"sound reaches the model." if sp_spans
                      else "No speech detected — nothing to remove."),
        "speech": sp_spans,
    }

    # --- stage 3: candidate mechanical regions (authoritative) ----------------
    regions, speech_frac = candidate_regions(y16, return_speech_frac=True)

    def _flat_of(s, e):
        return round(_flatness(yhi[int(s * config.SR_CLAP):int(e * config.SR_CLAP)]), 3)
    flat = [{"s": s, "e": e, "flatness": _flat_of(s, e)} for s, e in regions]
    yield "stage", {
        "id": "regions", "label": "Mechanical spans isolated",
        "narration": (f"After energy + speech + spectral-flatness filtering, "
                      f"{len(regions)} candidate mechanical span(s) survive "
                      f"({sum(e-s for s,e in regions):.1f}s of the {dur:.1f}s clip)."
                      if regions else
                      "No clean mechanical span survived — the whole clip is used."),
        "regions": [[s, e] for s, e in regions], "flatness": flat,
        "speech_frac": speech_frac,
    }

    # --- spectrogram ----------------------------------------------------------
    try:
        yield "spectrogram", _mel(yhi, config.SR_CLAP)
    except Exception:
        pass

    # --- stage 4: CLAP music gate --------------------------------------------
    isolated = [yhi[int(s*config.SR_CLAP):int(e*config.SR_CLAP)] for s, e in regions]
    isolated = [c for c in isolated if len(c) >= config.SR_CLAP // 2]
    music_prob = 0.0
    if isolated:
        try:
            from cardiag.audio.clap import Clap
            scores = Clap().score(isolated, _MUSIC_PROMPTS, sr=config.SR_CLAP)
            spans = []
            for (s, e), row in zip(regions, scores):
                mp = float(row[0])
                spans.append({"s": s, "e": e, "music": round(mp, 3),
                              "kept": mp < MUSIC_THRESH})
            music_prob = float(scores[:, 0].max())
            ndrop = sum(1 for sp in spans if not sp["kept"])
            yield "stage", {
                "id": "music", "label": "CLAP music gate",
                "narration": (f"CLAP scores each span against music vs mechanical "
                              f"prompts; {ndrop} musical span(s) dropped "
                              f"(peak music score {music_prob:.2f})." if ndrop else
                              f"No span looks like music (peak score {music_prob:.2f}) "
                              f"— all kept."),
                "spans": spans, "music_prob": round(music_prob, 3),
            }
        except Exception as exc:
            yield "stage", {"id": "music", "label": "CLAP music gate",
                            "narration": f"music gate unavailable ({type(exc).__name__})",
                            "spans": [], "music_prob": 0.0}

    # --- stage 5: diagnosis ---------------------------------------------------
    yield "stage", {"id": "embed", "label": "CLAP embedding + heads",
                    "narration": "Embedding the clean spans with CLAP and running "
                                 "the calibrated heads…", "spans": []}
    try:
        import os

        from cardiag import Classifier
        clf = Classifier.load(os.environ.get("CARDIAG_MODEL"))
        diag = clf.diagnose(str(path)).to_dict()
    except Exception as exc:
        yield "diagnosis", {"model_loaded": False,
                            "note": f"No trained model ({type(exc).__name__}). "
                                    f"Run `cardiag train --fixtures` to see a verdict.",
                            "music_prob": round(music_prob, 3)}
        yield "done", {"elapsed": round(time.time() - t0, 2)}
        return
    triage = None
    try:
        import os

        from cardiag import TriageClassifier
        triage = TriageClassifier.load(os.environ.get("CARDIAG_TRIAGE")).triage(str(path)).to_dict()
    except Exception:
        pass
    diag["model_loaded"] = True
    diag["triage"] = triage
    diag["music_prob"] = round(music_prob, 3)
    diag["calibrated"] = True
    yield "diagnosis", diag
    yield "done", {"elapsed": round(time.time() - t0, 2)}
