"""FastAPI app: upload a sound clip, get it cleaned and diagnosed locally.

The clip is run through the *same* ``clean()`` cascade as the training corpus,
then the model — so an uploaded clip is processed exactly like a training clip.
The model is loaded lazily and cached; if no trained model is present the app
still runs and returns the cleaning result (isolated spans, music flag).

    cardiag serve            # or: uvicorn cardiag.web.app:app --reload
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from cardiag import clean

app = FastAPI(title="cardiag", description="Diagnose a car fault from its sound.")
_STATIC = Path(__file__).parent / "static"

MAX_BYTES = 50 * 1024 * 1024          # 50 MB upload cap (a sound clip is tiny)
_OK_SUFFIX = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".webm", ".mp4"}
_LOCK = threading.Lock()              # serialize CLAP/torch (MPS not thread-safe)


@lru_cache(maxsize=1)
def _classifier():
    """Load the model once, lazily. Returns None if no model is available."""
    from cardiag import Classifier
    try:
        return Classifier.load(os.environ.get("CARDIAG_MODEL"))
    except (FileNotFoundError, ValueError):
        return None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


@app.get("/favicon.svg")
def favicon() -> FileResponse:
    return FileResponse(_STATIC / "favicon.svg")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _classifier() is not None}


def _safe_suffix(filename: str | None) -> str:
    """A whitelisted, sanitized file suffix — never trust the raw filename
    (path traversal, NUL bytes, megabyte-long fake extensions)."""
    suffix = Path(filename or "").suffix.lower()
    suffix = "".join(c for c in suffix if c.isalnum() or c == ".")[:8]
    return suffix if suffix in _OK_SUFFIX else ".wav"


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


# Small bounded cache of downloaded audio so the front end can PLAY a pasted-link
# clip (uploads play locally via an object URL; URL inputs are fetched server-side).
_AUDIO_DIR = Path(tempfile.gettempdir()) / "cardiag_audio"
_AUDIO_KEEP = 12


def _cache_audio(src_path: Path) -> str:
    """Move a downloaded clip into the audio cache; return its hex id. Evicts the
    oldest so the cache stays bounded."""
    import secrets
    _AUDIO_DIR.mkdir(exist_ok=True)
    jid = secrets.token_hex(8)
    dest = _AUDIO_DIR / f"{jid}{src_path.suffix.lower() or '.wav'}"
    shutil.move(str(src_path), dest)
    old = sorted(_AUDIO_DIR.glob("*"), key=lambda p: p.stat().st_mtime)[:-_AUDIO_KEEP]
    for p in old:
        p.unlink(missing_ok=True)
    return jid


@app.get("/api/audio/{jid}")
def get_audio(jid: str):
    """Serve a cached clip for playback. ``jid`` is validated as hex (no traversal)."""
    if not (jid and all(c in "0123456789abcdef" for c in jid) and len(jid) <= 32):
        return JSONResponse({"error": "bad id"}, status_code=400)
    hits = list(_AUDIO_DIR.glob(f"{jid}.*"))
    if not hits:
        return JSONResponse({"error": "expired"}, status_code=404)
    return FileResponse(hits[0])


@app.post("/api/diagnose/stream")
async def diagnose_stream(file: UploadFile | None = File(None),
                          url: str | None = Form(None)) -> StreamingResponse:
    """Stream the pipeline stage-by-stage as Server-Sent Events for the live UI.

    Accepts either an uploaded ``file`` or a ``url`` (YouTube/TikTok/Reddit). Each
    cleaning + diagnosis stage is emitted as it completes so the front end can
    animate the timeline. Heavy work runs in Starlette's threadpool (sync
    generator) and is serialized by ``_LOCK`` (torch/MPS is not thread-safe)."""
    from cardiag.web import explain

    workdir = tempfile.mkdtemp(prefix="cardiag_")
    src, title, path, err = "upload", "", None, None
    if url:
        src = explain.platform_of(url)
        title = url
    elif file is not None:
        path = Path(workdir) / f"upload{_safe_suffix(file.filename)}"
        total = 0
        with open(path, "wb") as fh:
            while chunk := await file.read(1 << 20):
                total += len(chunk)
                if total > MAX_BYTES:
                    err = f"file too large (>{MAX_BYTES // 1024 // 1024} MB)"
                    break
                fh.write(chunk)
        title = file.filename or "your clip"
        if total == 0:
            err = "empty upload"
    else:
        err = "provide a file or a url"

    def gen():
        nonlocal path, title
        try:
            if err:
                yield _sse("error", {"message": err})
                return
            if url:
                yield _sse("status", {"message": f"fetching audio from {src}…"})
                try:
                    path, got = explain.acquire_url(url, Path(workdir))
                    title = got or url
                except ValueError as e:
                    yield _sse("error", {"message": str(e)})
                    return
                jid = _cache_audio(path)                 # keep it so the UI can play it
                path = next(_AUDIO_DIR.glob(f"{jid}.*"))
                yield _sse("audio", {"url": f"/api/audio/{jid}"})
            with _LOCK:
                for name, payload in explain.explain(path, source=src, title=title):
                    yield _sse(name, payload)
        except Exception as e:                              # never leak a 500 mid-stream
            yield _sse("error", {"message": f"unexpected error: {type(e).__name__}"})
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/api/explain")
async def explain_why(file: UploadFile | None = File(None),
                      audio_id: str | None = Form(None)) -> JSONResponse:
    """Occlusion-saliency explanation of the fault/normal verdict — *why* the model
    decided. Accepts the same clip back (an upload, or a cached ``audio_id`` from a
    pasted-link run) and returns a time×frequency importance map. Heavy (re-embeds
    a grid of masked variants), so it's a deliberate opt-in, serialized by _LOCK."""
    from starlette.concurrency import run_in_threadpool

    from cardiag.audio import saliency

    tmp = None
    try:
        if audio_id:
            if not (all(c in "0123456789abcdef" for c in audio_id) and len(audio_id) <= 32):
                return JSONResponse({"error": "bad id"}, status_code=400)
            hits = list(_AUDIO_DIR.glob(f"{audio_id}.*"))
            if not hits:
                return JSONResponse({"error": "audio expired — re-run the clip"}, status_code=404)
            path = str(hits[0])
        elif file is not None:
            with tempfile.NamedTemporaryFile(suffix=_safe_suffix(file.filename),
                                             delete=False) as fh:
                tmp = fh.name
                total = 0
                while chunk := await file.read(1 << 20):
                    total += len(chunk)
                    if total > MAX_BYTES:
                        return JSONResponse({"error": "file too large"}, status_code=413)
                    fh.write(chunk)
            path = tmp
        else:
            return JSONResponse({"error": "provide a file or audio_id"}, status_code=400)

        model = os.environ.get("CARDIAG_MODEL")
        with _LOCK:
            res = await run_in_threadpool(saliency.occlusion_saliency, path, model)
        return JSONResponse(res)
    except (ValueError, FileNotFoundError, OSError) as e:
        return JSONResponse({"error": f"could not explain: {e}"}, status_code=400)
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


@app.post("/diagnose")
async def diagnose(file: UploadFile) -> JSONResponse:
    """Clean the uploaded clip, then diagnose it if a model is loaded. Hardened:
    size-capped chunked read, whitelisted suffix, errors -> 400/413 (never 500),
    inference serialized (torch/MPS is not thread-safe), temp file always removed."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=_safe_suffix(file.filename),
                                         delete=False) as tmp:
            tmp_path = tmp.name
            total = 0
            while chunk := await file.read(1 << 20):       # 1 MB chunks
                total += len(chunk)
                if total > MAX_BYTES:
                    return JSONResponse(
                        {"error": f"file too large (>{MAX_BYTES // 1024 // 1024} MB)"},
                        status_code=413)
                tmp.write(chunk)
        if total == 0:
            return JSONResponse({"error": "empty upload"}, status_code=400)

        clf = _classifier()
        with _LOCK:                                        # one clip on the model at a time
            if clf is None:
                res = clean(tmp_path)
                return JSONResponse({"model_loaded": False, "cleaning": res.to_dict()})
            result = clf.diagnose(tmp_path)
        payload = result.to_dict()
        payload["filename"] = file.filename
        payload["model_loaded"] = True
        return JSONResponse(payload)
    except (ValueError, FileNotFoundError, OSError) as e:
        return JSONResponse({"error": f"could not process audio: {e}"}, status_code=400)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
