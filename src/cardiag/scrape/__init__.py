"""Shared scraping transport. Camoufox (stealth Firefox) is the default browser
for all HTML/feed page fetching; yt-dlp is used only for media download and
YouTube's search API, where a browser is not the right tool."""
from cardiag.scrape.browser import Browser, camoufox_available

__all__ = ["Browser", "camoufox_available"]
