"""Browser end-to-end test of the local web app (Playwright).

Launches the real FastAPI app, opens the upload page in a headless browser,
uploads a synthetic clip, and asserts the result card renders. Runs CLAP-free:
with no model loaded the app returns the cleaning result, which the UI shows.

    pip install -e ".[web,dev]" && playwright install chromium
    pytest -m e2e
"""
from __future__ import annotations

import socket
import subprocess
import time

import numpy as np
import pytest
import soundfile as sf

pytestmark = pytest.mark.e2e


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def server():
    port = _free_port()
    proc = subprocess.Popen(
        ["python", "-m", "uvicorn", "cardiag.web.app:app",
         "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    # wait for health
    import urllib.request
    for _ in range(60):
        try:
            urllib.request.urlopen(base + "/health", timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    else:
        proc.terminate()
        pytest.fail("server did not start")
    yield base
    proc.terminate()


@pytest.fixture
def clip(tmp_path):
    p = tmp_path / "e2e.wav"
    sr = 48_000
    t = np.linspace(0, 3, sr * 3, endpoint=False)
    y = np.zeros_like(t)
    y[sr:2 * sr] = 0.4 * np.sin(2 * np.pi * 200 * t[sr:2 * sr])
    sf.write(p, y.astype(np.float32), sr)
    return str(p)


def test_upload_and_render(server, clip, page):
    page.goto(server)
    assert "cardiag" in page.title()
    page.set_input_files("#file", clip)
    # result card becomes visible after the /diagnose round-trip
    page.wait_for_selector("#result", state="visible", timeout=30_000)
    text = page.inner_text("#result")
    assert text.strip()  # verdict or "Cleaning only (no model loaded)"
