"""FastAPI app: upload a sound clip, get it cleaned and diagnosed locally.

The clip is run through the *same* ``clean()`` cascade as the training corpus,
then the model — so an uploaded clip is processed exactly like a training clip.
The model is loaded lazily and cached; if no trained model is present the app
still runs and returns the cleaning result (isolated spans, music flag).

    cardiag serve            # or: uvicorn cardiag.web.app:app --reload
"""
from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from cardiag import clean

app = FastAPI(title="cardiag", description="Diagnose a car fault from its sound.")
_STATIC = Path(__file__).parent / "static"


@lru_cache(maxsize=1)
def _classifier():
    """Load the model once, lazily. Returns None if no model is available."""
    from cardiag import Classifier
    try:
        return Classifier.load(os.environ.get("CARDIAG_MODEL"))
    except FileNotFoundError:
        return None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _classifier() is not None}


@app.post("/diagnose")
async def diagnose(file: UploadFile) -> JSONResponse:
    """Clean the uploaded clip, then diagnose it if a model is loaded."""
    suffix = Path(file.filename or "clip.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        clf = _classifier()
        if clf is None:
            # No model — still show the user what cleaning isolated.
            res = clean(tmp_path)
            payload = {"model_loaded": False, "cleaning": res.to_dict()}
            return JSONResponse(payload)
        result = clf.diagnose(tmp_path)
        payload = result.to_dict()
        payload["filename"] = file.filename
        payload["model_loaded"] = True
        return JSONResponse(payload)
    finally:
        os.unlink(tmp_path)
