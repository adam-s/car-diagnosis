"""Scrape r/MechanicAdvice fault-sound videos + diagnoses via old.reddit HTML.

Reddit blocks the .json API under load (403), but the old.reddit.com HTML pages
stay accessible without auth: the robust transport. Extraction is deterministic
(regex over structured HTML attrs + comment blocks); the only authenticated step
is yt-dlp pulling v.redd.it audio with the Firefox session cookies.

  FIND   : old.reddit /{new,top,hot}/ HTML + the "next" button (paginate)
  EXTRACT: post HTML page ?sort=top -> author/score/body, automod filtered
  AUDIO  : yt-dlp --cookies-from-browser firefox

Gentle throttle (rate-limit-safe), restart-safe (skips ledger ids). No LLM.
    python -m cardiag.ingest.reddit.scrape [max_pages_per_feed=12]
"""
import html
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

from cardiag import paths

# diagnosis-focused subs (video yields measured: carproblems 10/19,
# MechanicAdvice & AskMechanics ~6-7/25, others lower but on-topic)
SUBS = ["carproblems", "MechanicAdvice", "AskMechanics", "autorepair",
        "Justrolledintotheshop"]
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
# yt-dlp reads this browser's cookies to fetch v.redd.it audio. This decrypts
# your logged-in session; set CARDIAG_COOKIES_BROWSER=chrome/none to change, or
# accept that the Reddit audio step touches your Firefox profile. (See README.)
BROWSER = os.environ.get("CARDIAG_COOKIES_BROWSER", "firefox")
THROTTLE = 3.5              # be polite; HTML is tolerant but don't push it
DUR_MIN, DUR_MAX = 2, 180
BOTS = {"automoderator"}
FEEDS = [("new", ""), ("top", "?t=all"), ("top", "?t=year"),
         ("top", "?t=month"), ("hot", "")]

DATA = paths.REDDIT_DATA
AUDIO = DATA / "audio"
LEDGER = DATA / "posts.jsonl"

# Camoufox session (set by main() when the stealth browser is available). When
# set, page fetches go through Firefox: the project-wide scraping transport;
# otherwise fall back to a plain urllib request.
_SESSION = None


def get(url, tries=3):
    for i in range(tries):
        try:
            if _SESSION is not None:
                return _SESSION.get(url)
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            return urllib.request.urlopen(req, timeout=30).read().decode(
                "utf-8", "replace")
        except Exception as e:
            if getattr(e, "code", None) == 429 or i == tries - 1:
                if i == tries - 1:
                    raise
                time.sleep(15 * (i + 1))
            else:
                time.sleep(3 * (i + 1))


def media_id(url):
    """Stable id for the underlying video, so a repost across subs dedups once."""
    m = re.search(r"v\.redd\.it/(\w+)", url) \
        or re.search(r"(?:youtu\.be/|youtube\.com/(?:watch\?v=|shorts/))([\w-]+)",
                     url) or re.search(r"streamable\.com/(\w+)", url)
    return m.group(1) if m else url


def list_page(sub, feed, t, after):
    sep = "&" if t else "?"
    url = f"https://old.reddit.com/r/{sub}/{feed}/{t}"
    if after:
        url += f"{sep}count=25&after={after}"
    page = get(url)
    posts = []
    for m in re.finditer(r'<div [^>]*?data-fullname="(t3_\w+)"[^>]*?>', page):
        tag = m.group(0)
        durl = re.search(r'data-url="([^"]+)"', tag)
        perma = re.search(r'data-permalink="([^"]+)"', tag)
        dom = re.search(r'data-domain="([^"]+)"', tag)
        if durl and perma:
            posts.append({"fullname": m.group(1),
                          "url": html.unescape(durl.group(1)),
                          "permalink": "https://old.reddit.com"
                          + perma.group(1),
                          "domain": dom.group(1) if dom else ""})
    nxt = re.search(r'<span class="next-button"><a href="([^"]+)"', page)
    nxt_after = None
    if nxt:
        a = re.search(r"after=(t3_\w+)", html.unescape(nxt.group(1)))
        nxt_after = a.group(1) if a else None
    return posts, nxt_after


def is_video(p):
    d = p["domain"]
    return d == "v.redd.it" or "youtu" in d or d == "streamable.com"


def _clean(b):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", b))).strip()


def extract(permalink):
    page = get(permalink.rstrip("/") + "/?sort=top&limit=40")
    tm = re.search(r"<title>(.*?)</title>", page, re.S)
    title = html.unescape(tm.group(1)).split(" : ")[0][:200] if tm else ""
    ca = page.find('<div class="commentarea">')
    head, body = (page[:ca], page[ca:]) if ca > 0 else ("", page)
    sm = re.search(r'<div class="usertext-body[^"]*"[^>]*>\s*'
                   r'<div class="md">(.*?)</div>', head, re.S)
    selftext = _clean(sm.group(1))[:1000] if sm else ""
    comments = []
    for m in re.finditer(r'<div class="entry[^"]*">(.*?<div class="md">.*?</div>)',
                         body, re.S):
        chunk = m.group(1)
        au = re.search(r'class="author[^"]*"[^>]*>([^<]+)</a>', chunk)
        sc = re.search(r'(-?\d+)\s+points?', chunk)
        bd = re.search(r'<div class="md">(.*?)</div>', chunk, re.S)
        if not bd:
            continue
        author = au.group(1) if au else "?"
        text = _clean(bd.group(1))
        if author.lower() in BOTS or len(text) < 8 \
                or text.lower().startswith("thanks for posting"):
            continue
        comments.append({"author": author,
                         "score": int(sc.group(1)) if sc else 0,
                         "text": text[:800]})
    comments.sort(key=lambda c: -c["score"])
    return title, selftext, comments[:8]


def yt_meta(url):
    try:
        o = subprocess.run(
            ["yt-dlp", "--no-warnings", "--cookies-from-browser", BROWSER,
             "--print", "%(acodec)s\t%(duration)s", "--", url],
            capture_output=True, text=True, timeout=90).stdout.strip()
        ac, dur = o.split("\t")
        return ac, (float(dur) if dur not in ("NA", "") else 0.0)
    except Exception:
        return None, 0.0


def download_audio(url, out):
    try:
        subprocess.run(
            ["yt-dlp", "--no-warnings", "--cookies-from-browser", BROWSER,
             "-f", "ba", "-x", "--audio-format", "wav",
             "--postprocessor-args", "-ar 48000 -ac 1", "-o", str(out), "--", url],
            check=True, capture_output=True, timeout=180)
        return out.with_suffix(".wav").exists()
    except Exception:
        return False


def _scrape(max_pages=12):
    AUDIO.mkdir(parents=True, exist_ok=True)
    seen_post, seen_media = set(), set()
    if LEDGER.exists():
        for l in open(LEDGER):
            r = json.loads(l)
            seen_post.add(r["fullname"])
            seen_media.add(r.get("media_id") or media_id(r["url"]))
    print(f"resuming: {len(seen_post)} posts, {len(seen_media)} unique videos",
          flush=True)
    kept = scanned = reposts = 0
    with open(LEDGER, "a") as led:
        for sub in SUBS:
            for feed, t in FEEDS:
                after, pages = None, 0
                while pages < max_pages:
                    try:
                        posts, after = list_page(sub, feed, t, after)
                    except Exception as e:
                        print(f"  r/{sub} {feed}{t} failed "
                              f"({type(e).__name__}); next", flush=True)
                        break
                    pages += 1
                    for p in posts:
                        if p["fullname"] in seen_post or not is_video(p):
                            continue
                        seen_post.add(p["fullname"])
                        mid = media_id(p["url"])
                        if mid in seen_media:        # repost of a video we have
                            reposts += 1
                            continue
                        seen_media.add(mid)
                        scanned += 1
                        ac, dur = yt_meta(p["url"])
                        if not ac or ac in ("none", "NA") \
                                or not (DUR_MIN <= dur <= DUR_MAX):
                            continue
                        wav = AUDIO / f"{p['fullname']}.wav"
                        if not wav.exists() \
                                and not download_audio(p["url"], wav):
                            continue
                        try:
                            title, selftext, coms = extract(p["permalink"])
                        except Exception:
                            title, selftext, coms = "", "", []
                        led.write(json.dumps({
                            "fullname": p["fullname"], "subreddit": sub,
                            "media_id": mid, "url": p["url"],
                            "permalink": p["permalink"], "domain": p["domain"],
                            "duration": dur, "title": title,
                            "selftext": selftext, "comments": coms,
                            "file": f"data/audio/{p['fullname']}.wav"}) + "\n")
                        led.flush()
                        kept += 1
                        time.sleep(THROTTLE)
                    print(f"  r/{sub} {feed}{t} p{pages}: kept {kept} "
                          f"(scanned {scanned}, reposts {reposts})", flush=True)
                    if not after:
                        break
                    time.sleep(THROTTLE)
    print(f"\nDONE: kept {kept} of {scanned} scanned, {reposts} reposts "
          f"skipped -> {LEDGER}")


def main(max_pages=12, use_browser=True):
    """Scrape Reddit. Page fetches go through Camoufox (the project-wide stealth
    transport) when available; yt-dlp still pulls the v.redd.it audio."""
    global _SESSION
    session_cm = None
    if use_browser:
        try:
            from cardiag.scrape import Browser, camoufox_available
            if camoufox_available():
                session_cm = Browser()
                _SESSION = session_cm.__enter__()
                try:                       # land on old.reddit first (same-origin)
                    _SESSION.get("https://old.reddit.com/")
                except Exception:
                    pass
                print("reddit: fetching via Camoufox (stealth Firefox)", flush=True)
        except Exception as e:
            print(f"reddit: Camoufox unavailable ({type(e).__name__}); "
                  f"falling back to urllib", flush=True)
            session_cm = None
    try:
        _scrape(max_pages)
    finally:
        if session_cm is not None:
            session_cm.__exit__(None, None, None)
            _SESSION = None


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
