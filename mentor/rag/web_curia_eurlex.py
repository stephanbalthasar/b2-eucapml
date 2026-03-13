# mentor/rag/web_curia_eurlex.py
from __future__ import annotations

import html
import io
import re
from typing import List, Dict, Tuple
from urllib.parse import urlparse, parse_qs, unquote

import requests

# Optional HTML parsing (preferred). If bs4 is absent, we fall back to regex.
try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None  # fallback will be used

# Optional PDF text extraction. If PyPDF2 is absent, we skip PDFs.
try:
    from PyPDF2 import PdfReader  # type: ignore
except Exception:
    PdfReader = None


# -----------------------
# Config / constants
# -----------------------

_GOOGLE_SEARCH_URL = "https://www.google.com/search"
_DDG_SEARCH_URL = "https://html.duckduckgo.com/html/"

# Allowed domains (kept small and authoritative)
_ALLOWED_DOMAINS = ("curia.europa.eu", "eur-lex.europa.eu", "esma.europa.eu")


def _ua() -> str:
    """Minimal UA to reduce blocks in common environments."""
    return (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )


def _is_english_url(u: str) -> bool:
    u = u.lower()
    return ("/en/" in u) or u.endswith("en.pdf") or "lang=en" in u or "language=en" in u


# -----------------------
# URL classification
# -----------------------

def _classify(url: str) -> Tuple[int, str]:
    """
    Return (priority, label).
      1 -> CURIA press release (usually cpNNNNxx.pdf)
      2 -> EUR-Lex judgment summary (_SUM)
      3 -> Judgment text (EUR-Lex / CURIA docs)
      4 -> ESMA
      5 -> Other (still allowed domain)
    """
    u = url.lower()

    # 1) CURIA press release (press PDFs typically named cpNNNNNNen.pdf)
    if "curia.europa.eu" in u and u.endswith(".pdf") and "/upload/docs/" in u and "/application/pdf/" in u and "cp" in u:
        return (1, "CURIA_PRESS_RELEASE")

    # 2) EUR-Lex summary uses CELEX ... _SUM
    # e.g., https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:62013CJ0628_SUM
    if "eur-lex.europa.eu" in u and "_sum" in u:
        return (2, "EURLEX_SUMMARY")

    # 3) Judgment text (EUR-Lex or CURIA doc pages that are not summary)
    if ("eur-lex.europa.eu" in u or "curia.europa.eu" in u) and (
        "document" in u or "docid=" in u or "celex:" in u or "/juris/" in u
    ):
        return (3, "JUDGMENT")

    # 4) ESMA domain (Q&A, guidelines, statements)
    if "esma.europa.eu" in u:
        return (4, "ESMA")

    # 5) Other (within allowed domains)
    return (5, "OTHER")


# -----------------------
# Search result extraction
# -----------------------

def _extract_google_results_html(html_text: str) -> List[str]:
    """
    Extract result target URLs from a Google HTML SERP.

    Robust to:
      - relative redirect anchors: /url?q=...
      - absolute redirect anchors: https://www.google.com/url?...
      - direct anchors to allowed domains
    """
    urls: List[str] = []

    if BeautifulSoup is not None:
        try:
            soup = BeautifulSoup(html_text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"]

                # Ignore internal Google links
                if href.startswith("/search"):
                    continue
                if "google." in href and "/url?" not in href:
                    continue

                # 1) Relative redirect pattern: /url?q=<target>&...
                if href.startswith("/url?"):
                    qs = parse_qs(href.split("?", 1)[-1])
                    target = qs.get("q", [None])[0]
                    if target and target.startswith("http"):
                        urls.append(unquote(target))
                        continue

                # 2) Absolute redirect: https://www.google.com/url?....
                if href.startswith("https://www.google.com/url"):
                    qs = parse_qs(href.split("?", 1)[-1])
                    target = qs.get("q", [None])[0]
                    if target and target.startswith("http"):
                        urls.append(unquote(target))
                        continue

                # 3) Direct links
                if href.startswith("http"):
                    urls.append(unquote(href))
        except Exception:
            pass

    # Fallback regex parse if BS4 missing or failed
    if not urls:
        # relative /url?q=...
        for m in re.finditer(r'href="/url\?q=([^"&]+)', html_text or ""):
            urls.append(unquote(m.group(1)))
        # direct http(s) links in anchors
        for m in re.finditer(r'href="(https?://[^"]+)"', html_text or ""):
            urls.append(unquote(m.group(1)))

    # Remove Google junk and duplicates
    cleaned: List[str] = []
    seen: set[str] = set()
    for u in urls:
        if not u.startswith("http"):
            continue
        if "google." in u:
            continue
        if u.startswith("https://webcache.googleusercontent.com") or u.startswith("http://webcache.googleusercontent.com"):
            continue
        norm = u.split("#")[0]
        if norm not in seen:
            seen.add(norm)
            cleaned.append(norm)
    return cleaned


def _extract_ddg_results_html(html_text: str) -> List[str]:
    """
    Extract result URLs from DuckDuckGo's static HTML endpoint.
    Result links typically look like:
      /l/?kh=-1&uddg=https%3A%2F%2Feur-lex.europa.eu%2F...
    """
    urls: List[str] = []

    if BeautifulSoup is not None:
        try:
            soup = BeautifulSoup(html_text, "html.parser")
            for a in soup.select("a.result__a[href]"):
                href = a["href"]
                if href.startswith("/l/?"):
                    qs = parse_qs(href.split("?", 1)[-1])
                    target = qs.get("uddg", [None])[0]
                    if target and target.startswith("http"):
                        urls.append(unquote(target))
                elif href.startswith("http"):
                    urls.append(unquote(href))
        except Exception:
            pass

    # Fallback regex parse if BS4 missing or failed
    if not urls:
        for m in re.finditer(r'href="(/l/\?[^"]+)"', html_text or ""):
            qs = parse_qs(m.group(1).split("?", 1)[-1])
            target = qs.get("uddg", [None])[0]
            if target and target.startswith("http"):
                urls.append(unquote(target))
        for m in re.finditer(r'href="(https?://[^"]+)"', html_text or ""):
            urls.append(unquote(m.group(1)))

    # De‑dupe
    cleaned: List[str] = []
    seen: set[str] = set()
    for u in urls:
        if not u.startswith("http"):
            continue
        norm = u.split("#")[0]
        if norm not in seen:
            seen.add(norm)
            cleaned.append(norm)
    return cleaned


def _filter_rank_urls(urls: List[str], prefer_en: bool) -> List[Tuple[str, int, str, int]]:
    """
    Keep allowed domains, classify & rank.
    Returns list of tuples (url, priority, label, lang_penalty) ordered by (priority, penalty).
    """
    ranked: List[Tuple[str, int, str, int]] = []
    seen: set[str] = set()

    for u in urls:
        parsed = urlparse(u)
        host = parsed.netloc.lower()
        if not any(host.endswith(dom) for dom in _ALLOWED_DOMAINS):
            continue

        # De‑dupe by normalized URL (strip fragment)
        norm = u.split("#")[0]
        if norm in seen:
            continue
        seen.add(norm)

        prio, label = _classify(u)
        penalty = 0
        if prefer_en and not _is_english_url(u):
            penalty = 1

        ranked.append((u, prio, label, penalty))

    ranked.sort(key=lambda t: (t[1], t[3]))
    return ranked


# -----------------------
# Fetching & extraction
# -----------------------

def _http_get(url: str, *, timeout: float) -> Tuple[int, bytes, str]:
    """Return (status_code, content, content_type)."""
    try:
        r = requests.get(
            url,
            headers={"User-Agent": _ua(), "Accept": "*/*"},
            timeout=timeout,
            allow_redirects=True,
        )
        return r.status_code, (r.content or b""), (r.headers.get("Content-Type", "") or "")
    except Exception:
        return 0, b"", ""


def _extract_text_from_pdf(content: bytes) -> str:
    if not content or PdfReader is None:
        return ""
    try:
        reader = PdfReader(io.BytesIO(content))
        # Press releases state the holding up front; 1–2 pages are enough.
        pages = min(2, len(reader.pages))
        out = []
        for i in range(pages):
            try:
                out.append(reader.pages[i].extract_text() or "")
            except Exception:
                pass
        return " ".join(out)
    except Exception:
        return ""


def _extract_text_from_html(content: bytes) -> str:
    # Try UTF‑8 then Latin‑1
    try:
        text = content.decode("utf-8", errors="replace")
    except Exception:
        text = content.decode("latin-1", errors="replace")

    if BeautifulSoup is None:
        # Rough tag strip fallback
        return re.sub(r"<[^>]+>", " ", text or "")

    soup = BeautifulSoup(text, "html.parser")

    # Heuristics: prefer visible paragraphs in the main body
    body = soup.body or soup
    paras: List[str] = []

    # EUR‑Lex summaries often have readable <p> blocks near the top;
    # CURIA HTML press pages (if any) are also short and <p>-based.
    for p in body.find_all("p"):
        t = p.get_text(" ", strip=True)
        if t and len(t.split()) > 4:
            paras.append(t)
        if len(paras) >= 8:
            break

    if paras:
        return " ".join(paras)

    # Fallback: all text
    return soup.get_text(" ", strip=True)


def _to_snippet(text: str, *, limit: int = 700) -> str:
    """Normalize whitespace and trim to <= limit at a sentence boundary if possible."""
    t = html.unescape(text or "")
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) <= limit:
        return t

    # Try to end near the limit at a sentence boundary within the last 160 chars window
    window_start = max(0, limit - 160)
    window = t[window_start:limit]
    # Look for '.', '!' or '?' followed by space or end
    m = None
    for m_ in re.finditer(r"?=\s|$", window):
        m = m_
    if m:
        end = window_start + m.end()
        return t[:end].strip()
    return t[:limit].strip()


def _lang_from_url(url: str) -> str:
    u = url.lower()
    if "/de/" in u or u.endswith("de.pdf"):
        return "DE"
    if "/fr/" in u or u.endswith("fr.pdf"):
        return "FR"
    if _is_english_url(u):
        return "EN"
    return "EN"  # default


# -----------------------
# Public retriever
# -----------------------

class CuriaEurlexRetriever:
    """
    Google-first retriever for official EU legal sources (CURIA, EUR-Lex, ESMA).

    • Queries Google with a site filter.
    • If SERP is empty (consent/captcha), falls back to DuckDuckGo HTML (still site-filtered).
    • Keeps the first few hits from allowed domains; de-duplicates; prefers EN.
    • Ranks: CURIA press release > EUR-Lex summary > judgment > ESMA > other.
    • Fetches each candidate and extracts a compact snippet (<= 700 chars).
    • Returns up to 'top_k' snippets: List[Dict] with keys: text, source_url, source_type, lang.
    """

    def __init__(self, lang: str = "EN", timeout_sec: float = 6.0):
        self.lang = (lang or "EN").upper()
        self.timeout = float(timeout_sec)

    def retrieve(self, query: str, keywords: List[str] | None = None, top_k: int = 4) -> List[Dict]:
        if not (query or "").strip():
            return []

        site_filter = "site:eur-lex.europa.eu OR site:curia.europa.eu OR site:esma.europa.eu"

        # ---- Google pass 1: quoted query ----
        q1 = f'{site_filter} "{query.strip()}"'
        g_urls = self._google_to_urls(q1)

        # ---- Google pass 2: unquoted query (still site‑filtered) ----
        if not g_urls:
            q2 = f"{site_filter} {query.strip()}"
            g_urls = self._google_to_urls(q2)

        urls = g_urls

        # ---- DuckDuckGo fallback (only if Google yielded nothing) ----
        if not urls:
            d1 = self._ddg_to_urls(f'{site_filter} "{query.strip()}"')
            if not d1:
                d1 = self._ddg_to_urls(f"{site_filter} {query.strip()}")
            urls = d1

        if not urls:
            return []

        # 2) Keep only top 5 candidates (as requested), ranked & EN‑preferred
        ranked = _filter_rank_urls(urls, prefer_en=(self.lang == "EN"))
        candidates = ranked[:5]

        # 3) Fetch → extract → build snippets
        out: List[Dict] = []
        for url, prio, label, penalty in candidates:
            status, content, ctype = _http_get(url, timeout=self.timeout)
            if status != 200 or not content:
                continue

            if "pdf" in (ctype or "").lower() or url.lower().endswith(".pdf"):
                text = _extract_text_from_pdf(content)
            else:
                text = _extract_text_from_html(content)

            if not text:
                continue

            snippet = _to_snippet(text, limit=700)
            # Require a minimal length so we don't inject noise
            if len(snippet.split()) < 12:
                continue

            out.append({
                "text": snippet,
                "source_url": url,
                "source_type": label,
                "lang": _lang_from_url(url),
            })

            if len(out) == top_k:
                break

        return out

    # ---- helpers ----

    def _google_to_urls(self, full_query: str) -> List[str]:
        """Run a Google query (site-filtered), parse the HTML, and return a list of target URLs."""
        try:
            r = requests.get(
                _GOOGLE_SEARCH_URL,
                params={"q": full_query, "hl": self.lang.lower(), "num": "10", "safe": "off"},
                headers={"User-Agent": _ua(), "Accept": "text/html,*/*"},
                timeout=self.timeout,
            )
            if r.status_code != 200:
                return []
            text = r.text or ""

            # Detect consent / unusual-traffic interstitials → treat as empty so we try fallback
            tl = text.lower()
            if ("consent.google" in tl) or ("before you continue to google" in tl) or ("unusual traffic" in tl):
                return []

            return _extract_google_results_html(text)
        except Exception:
            return []

    def _ddg_to_urls(self, full_query: str) -> List[str]:
        """Run a DuckDuckGo HTML query (site-filtered), parse the HTML, and return target URLs."""
        try:
            r = requests.post(  # DDG HTML endpoint prefers POST form submissions
                _DDG_SEARCH_URL,
                data={"q": full_query},
                headers={"User-Agent": _ua(), "Accept": "text/html,*/*"},
                timeout=self.timeout,
            )
            if r.status_code != 200:
                return []
            return _extract_ddg_results_html(r.text or "")
        except Exception:
            return []
