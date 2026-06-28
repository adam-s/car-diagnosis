"""TikTok discovery via Camoufox (stealth Firefox) network interception.

yt-dlp can't read TikTok search (it's a signed in-app XHR), so we drive a browser
to each search page, let TikTok fire ``/api/search/item/full/``, and capture the
feed JSON off the wire. Camoufox is preferred over Chromium automation here: its
anti-fingerprinting is implemented at the Firefox C++ engine level, so it passes
WAFs that flag patched-Chromium and never leaks the headless tells.

Needs the stealth browser::

    pip install -e ".[scrape]"
    python -m camoufox fetch        # downloads the Camoufox Firefox (~600 MB)

Writes data/tiktok/worklist.jsonl (one video/line, dedup on id). Reuses the feed
parser from the patchright discover so both backends produce identical records.
"""
from __future__ import annotations

import json

from cardiag import paths
from cardiag.ingest.tiktok.discover import PROBLEM_QUERIES, extract_items

DATA = paths.TT_DATA


def run(queries=None, target: int = 20, headless: bool = True) -> int:
    """Drive Camoufox over the search queries, intercept the feed JSON, and
    append discovered videos to the worklist. Returns the count of new videos."""
    from camoufox.sync_api import Camoufox

    queries = queries or PROBLEM_QUERIES
    DATA.mkdir(parents=True, exist_ok=True)
    found: dict[str, dict] = {}
    state = {"q": ""}

    def on_response(resp):
        url = resp.url
        if "/api/search/item/full" not in url and "/api/recommend/item_list" not in url:
            return
        try:
            data = resp.json()
        except Exception:
            return
        extract_items(data, found, state["q"])

    with Camoufox(headless=headless, humanize=True) as browser:
        page = browser.new_page()
        # Firefox emits location-less pageerrors that crash the Playwright driver;
        # swallow them (same guard the camoufox_scraper wrapper uses).
        page.on("pageerror", lambda e: None)
        page.on("response", on_response)
        try:
            page.goto("https://www.tiktok.com/", wait_until="domcontentloaded",
                      timeout=30000)
            page.wait_for_timeout(3000)
        except Exception:
            pass
        for q in queries:
            state["q"] = q
            before = len(found)
            try:
                page.goto(f"https://www.tiktok.com/search/video?q={q.replace(' ', '%20')}",
                          wait_until="domcontentloaded", timeout=30000)
                page.wait_for_timeout(3500)
                stale = 0
                for _ in range(10):
                    n = len(found)
                    page.mouse.wheel(0, 4000)
                    page.wait_for_timeout(1300)
                    stale = stale + 1 if len(found) == n else 0
                    if len(found) >= before + target or stale >= 3:
                        break
            except Exception as e:
                print(f"  ! {q}: {type(e).__name__}", flush=True)
            print(f"  {q:<34} +{len(found) - before:>3} (total {len(found)})",
                  flush=True)

    wl = DATA / "worklist.jsonl"
    seen = {json.loads(l)["id"] for l in open(wl)} if wl.exists() else set()
    new = 0
    with open(wl, "a") as f:
        for v in found.values():
            if v["id"] not in seen:
                f.write(json.dumps(v) + "\n")
                seen.add(v["id"])
                new += 1
    print(f"+{new} new videos -> {wl}")
    return new
