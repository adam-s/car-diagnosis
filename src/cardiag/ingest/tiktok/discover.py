"""TikTok discovery via stealth-browser network interception (intercept2 pattern).

yt-dlp can't read TikTok search: it's a signed in-app XHR. So we drive a stealth
browser to each search page, let it fire /api/search/item/full/, and capture the
feed JSON off the wire. One browser session loops all queries.

Stealth (from intercept2 fingerprint-controller): patchright + chrome channel
gives webdriver=false and clean sec-ch-ua for free; we additionally strip
"HeadlessChrome" from navigator.userAgent (the one remaining bot tell) so this
runs HEADLESS: no visible window, scalable unsupervised.

    python -m cardiag.ingest.tiktok.discover                  # all PROBLEM_QUERIES
    python -m cardiag.ingest.tiktok.discover "one custom query"  # single query
    python -m cardiag.ingest.tiktok.discover --headed         # if ever blocked

-> data/worklist.jsonl (one video/line, dedup on id).
"""
import asyncio
import json
import sys

from cardiag import paths

# NOTE: patchright is the fallback browser; Camoufox is primary
# (discover_camoufox.py). Import patchright lazily inside run() so this module's
# shared PROBLEM_QUERIES / extract_items stay importable without it installed.

DATA = paths.TT_DATA
USER_DIR = DATA / "browser_profile"

# ~35 common faults, system-spanning. Doubles as the discovery query set AND the
# coarse cause taxonomy (Haiku normalizes overlay text into these).
PROBLEM_QUERIES = [
    "bad wheel bearing sound", "bad cv joint sound", "bad ball joint sound",
    "bad tie rod end sound", "bad control arm bushing noise", "bad sway bar link noise",
    "bad strut mount noise", "worn shock absorber noise", "bad wheel hub noise",
    "brake pad grinding noise", "warped brake rotor noise", "brake caliper sticking noise",
    "serpentine belt squeal", "belt tensioner noise", "bad idler pulley noise",
    "bad alternator bearing noise", "bad power steering pump whine", "bad water pump noise",
    "bad ac compressor noise", "engine rod knock sound", "engine lifter tick noise",
    "timing chain rattle noise", "exhaust leak ticking sound", "bad catalytic converter rattle",
    "bad fuel pump whine", "bad clutch release bearing noise", "bad throwout bearing sound",
    "bad u joint clunk", "bad differential whine", "bad transmission bearing noise",
    "bad turbo whine", "spark knock pinging sound", "loose heat shield rattle",
    "bad motor mount clunk", "valve tapping noise",
]

# Healthy-engine queries: the missing 'normal' class. The corpus is otherwise
# fault-dominated (every normal clip came from YouTube), so scraping these lets
# `cardiag train` learn fault-vs-normal from more than one source and breaks the
# recording-source confound documented in docs/MODEL_CARD.md. Labels are weak
# (a "healthy engine" clip could be clickbait); same honest weak supervision as
# the problem queries.
NORMAL_QUERIES = [
    "healthy engine idle sound", "what a good engine sounds like",
    "smooth running engine sound", "normal car idle sound", "quiet engine idle",
    "perfect engine purr", "new engine sound idle", "healthy engine bay sound",
    "well maintained engine sound", "normal exhaust note idle",
    "engine running smooth no noise", "good cold start engine sound",
]


def extract_items(obj, out, query):
    if isinstance(obj, dict):
        vid, author = obj.get("id"), obj.get("author")
        if (isinstance(vid, str) and vid.isdigit() and len(vid) > 15
                and isinstance(author, dict)):
            uid = author.get("uniqueId") or author.get("unique_id")
            if uid and vid not in out:
                out[vid] = {
                    "id": vid, "author": uid, "desc": (obj.get("desc") or "")[:160],
                    "url": f"https://www.tiktok.com/@{uid}/video/{vid}",
                    "music_id": (obj.get("music") or {}).get("id"),
                    "platform": "tiktok", "query": query,
                }
        for v in obj.values():
            extract_items(v, out, query)
    elif isinstance(obj, list):
        for v in obj:
            extract_items(v, out, query)


async def search_one(page, found, query, target):
    before_total = len(found)
    hits = {}

    async def on_response(resp):
        if "/api/search/item/full" not in resp.url and "/api/recommend/item_list" not in resp.url:
            return
        if "json" not in (resp.headers or {}).get("content-type", ""):
            return
        try:
            data = await resp.json()
        except Exception:
            return
        extract_items(data, hits, query)

    page.on("response", on_response)
    try:
        q = query.replace(" ", "%20")
        await page.goto(f"https://www.tiktok.com/search/video?q={q}",
                        wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(3500)
        stale = 0
        for _ in range(12):
            n = len(hits)
            if n >= target or stale >= 3:
                break
            await page.mouse.wheel(0, 4000)
            await page.wait_for_timeout(1300)
            stale = stale + 1 if len(hits) == n else 0
    except Exception as e:
        print(f"  ! {query}: {type(e).__name__}", flush=True)
    finally:
        page.remove_listener("response", on_response)
    found.update(hits)
    print(f"  {query:<34} +{len(found)-before_total:>3} (total {len(found)})", flush=True)


async def run(queries, target, headed):
    DATA.mkdir(exist_ok=True)
    found = {}
    from patchright.async_api import async_playwright  # lazy (fallback browser)
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            user_data_dir=str(USER_DIR), headless=not headed, channel="chrome",
            no_viewport=True, ignore_default_args=["--enable-automation"])
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        ua = await page.evaluate("navigator.userAgent")
        if "Headless" in ua:
            clean = ua.replace("HeadlessChrome", "Chrome")
            major = clean.split("Chrome/")[1].split(".")[0]
            client = await ctx.new_cdp_session(page)
            await client.send("Network.setUserAgentOverride", {
                "userAgent": clean,
                "userAgentMetadata": {
                    "platform": "macOS", "platformVersion": "15.0", "architecture": "arm",
                    "model": "", "mobile": False,
                    "brands": [{"brand": "Google Chrome", "version": major},
                               {"brand": "Chromium", "version": major},
                               {"brand": "Not)A;Brand", "version": "24"}]}})
            print(f"[stealth] UA HeadlessChrome -> Chrome (v{major})\n", flush=True)
        await page.goto("https://www.tiktok.com/", wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(3000)
        for q in queries:
            await search_one(page, found, q, target)
        await ctx.close()

    wl = DATA / "worklist.jsonl"
    seen = {json.loads(l)["id"] for l in open(wl)} if wl.exists() else set()
    new = 0
    with open(wl, "a") as f:
        for v in found.values():
            if v["id"] not in seen:
                f.write(json.dumps(v) + "\n"); seen.add(v["id"]); new += 1
    print(f"\n+{new} new videos; worklist now {len(seen)} -> {wl}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    headed = "--headed" in sys.argv
    queries = [args[0]] if args else PROBLEM_QUERIES
    asyncio.run(run(queries, target=20, headed=headed))
