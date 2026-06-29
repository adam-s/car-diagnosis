"""Camoufox transport: the default browser for HTML/feed page scraping.

Camoufox is a hardened Firefox whose anti-fingerprinting lives at the C++ engine
level, so it passes WAFs that flag Chromium automation and never leaks the
headless tells. We use it for every page/feed fetch (Reddit listings + posts,
TikTok search). Media download stays on yt-dlp (a browser can't extract signed
CDN media), and YouTube search stays on yt-dlp's search API.

    from cardiag.scrape import Browser
    with Browser() as b:
        html = b.get("https://old.reddit.com/r/MechanicAdvice/")

Needs ``pip install -e ".[scrape]"`` and ``python -m camoufox fetch`` (downloads
the Camoufox Firefox once). ``playwright`` is pinned to 1.51.0 to match Camoufox's
Firefox 135 (newer Playwright crashes on Firefox's location-less pageError).
"""
from __future__ import annotations

import platform


def camoufox_available() -> bool:
    try:
        import camoufox  # noqa: F401
        return True
    except Exception:
        return False


def _default_headless():
    # virtual display (Xvfb) on Linux servers; headed on macOS dev; never True
    # for protected targets, but TikTok/Reddit tolerate headless Firefox here.
    return True if platform.system() == "Linux" else True


class Browser:
    """A stealth Firefox page. Context manager; reuse one session for many gets."""

    def __init__(self, headless: bool | str = "auto"):
        self.headless = _default_headless() if headless == "auto" else headless
        self._cm = None
        self.browser = None
        self.page = None

    def __enter__(self) -> Browser:
        from camoufox.sync_api import Camoufox
        self._cm = Camoufox(headless=self.headless, humanize=True)
        self.browser = self._cm.__enter__()
        self.page = self.browser.new_page()
        # Firefox emits location-less pageerrors that crash the driver; swallow.
        self.page.on("pageerror", lambda e: None)
        return self

    def get(self, url: str, timeout: int = 30000) -> str:
        """Navigate to ``url`` and return the rendered HTML (server-side markup
        for old.reddit). Raises on navigation failure."""
        self.page.goto(url, wait_until="domcontentloaded", timeout=timeout)
        return self.page.content()

    def __exit__(self, *exc):
        if self._cm is not None:
            self._cm.__exit__(*exc)
