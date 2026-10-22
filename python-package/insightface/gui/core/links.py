"""External link helpers for GUI-origin attribution."""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices

INSIGHTFACE_LINK_HOSTS = {"insightface.ai", "www.insightface.ai"}
GUI_UTM_SOURCE = "insightface_gui"


def add_gui_referrer(url: str, *, content: str | None = None) -> str:
    """Add GUI source attribution to InsightFace website URLs.

    Native desktop apps cannot reliably set the browser's HTTP Referer header
    when delegating a URL to the user's default browser. Query attribution is
    therefore the stable, privacy-visible path for website-side analytics.

    The `content` argument is accepted for API compatibility but intentionally
    not emitted; Vercel UTM reporting only needs `utm_source` for this app.
    """

    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or parts.hostname not in INSIGHTFACE_LINK_HOSTS:
        return url

    params = parse_qsl(parts.query, keep_blank_values=True)
    existing = {key for key, _ in params}
    if "utm_source" not in existing:
        params.append(("utm_source", GUI_UTM_SOURCE))

    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(params), parts.fragment))


def open_insightface_url(url: str, *, content: str | None = None) -> bool:
    """Open an InsightFace website URL with GUI attribution parameters."""

    return QDesktopServices.openUrl(QUrl(add_gui_referrer(url, content=content)))
