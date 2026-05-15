import asyncio
import datetime
import json
import os
import re
import typing as t
from urllib.parse import urljoin

import cloudscraper  # type: ignore
import requests
from bs4 import BeautifulSoup

import logging

logger = logging.getLogger(__name__)

# ── proxy configuration ───────────────────────────────────────────────────────
# Set WEBSHARE_PROXIES env var as newline- or comma-separated "ip:port:user:pass" entries.
# Falls back to the hardcoded list below if the env var is not set.

# _DEFAULT_PROXY_LIST = """
# 142.111.48.253:7030:zipdfqff:429jraf4ct0f
# 23.95.150.145:6114:zipdfqff:429jraf4ct0f
# 45.38.107.97:6014:zipdfqff:429jraf4ct0f
# 198.23.243.226:6361:zipdfqff:429jraf4ct0f
# 84.247.60.125:6095:zipdfqff:429jraf4ct0f
# 104.239.107.47:5699:zipdfqff:429jraf4ct0f
# 23.27.208.120:5830:zipdfqff:429jraf4ct0f
# 23.229.19.94:8689:zipdfqff:429jraf4ct0f
# 2.57.20.2:6983:zipdfqff:429jraf4ct0f
# 198.154.89.151:6242:zipdfqff:429jraf4ct0f
# """.strip()

_DEFAULT_PROXY_LIST = """
142.111.48.253:7030:rxmizirn:juw4hs60mfzn
23.95.150.145:6114:rxmizirn:juw4hs60mfzn
45.38.107.97:6014:rxmizirn:juw4hs60mfzn
38.154.203.95:5863:rxmizirn:juw4hs60mfzn
198.23.243.226:6361:rxmizirn:juw4hs60mfzn
84.247.60.125:6095:rxmizirn:juw4hs60mfzn
104.239.107.47:5699:rxmizirn:juw4hs60mfzn
23.27.208.120:5830:rxmizirn:juw4hs60mfzn
23.229.19.94:8689:rxmizirn:juw4hs60mfzn
2.57.20.2:6983:rxmizirn:juw4hs60mfzn
""".strip()


def _build_proxy_urls(raw: str) -> t.List[str]:
    urls = []
    for line in re.split(r"[\n,]+", raw):
        line = line.strip()
        if not line:
            continue
        parts = line.split(":")
        if len(parts) == 4:
            ip, port, user, password = parts
            urls.append(f"http://{user}:{password}@{ip}:{port}")
        elif len(parts) == 2:
            # already "host:port" with no auth
            urls.append(f"http://{line}")
        else:
            urls.append(line)
    return urls


WEBSHARE_PROXIES: t.List[str] = _build_proxy_urls(
    os.environ.get("WEBSHARE_PROXIES", _DEFAULT_PROXY_LIST)
)

# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_datetime(raw: t.Optional[str]) -> t.Optional[str]:
    if not raw:
        return None
    raw = raw.strip()
    if raw.isdigit():
        ts = int(raw)
        if ts > 1e11:   # unix ms → s
            ts //= 1000
        try:
            return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).isoformat()
        except Exception:
            pass
    return raw


def _split_csv(value: t.Optional[str]) -> t.List[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


_SLUG_RE = re.compile(r'^[a-z0-9]+(-[a-z0-9]+)+$')   # e.g. mcc-th-seo, ksc-mcc-coi
_BOOL_STRINGS = {"true", "false"}

def _parse_prefixed_tags(value: t.Optional[str]) -> t.List[str]:
    """Parse semicolon-separated tags with optional 'type:value' prefixes.

    e.g. 'tag:Bikini;tag:Celebrity Body;celebrity:Amy Schumer' → ['Bikini', 'Celebrity Body', 'Amy Schumer']
    Internal CMS slugs (all-lowercase-hyphenated) and booleans are filtered out.
    """
    if not value:
        return []
    out = []
    for part in value.split(";"):
        part = part.strip()
        val = part.split(":", 1)[1].strip() if ":" in part else part
        if not val:
            continue
        if val.lower() in _BOOL_STRINGS:
            continue
        if _SLUG_RE.match(val):   # internal CMS slug — skip
            continue
        if val not in out:
            out.append(val)
    return out


# ── JSON-LD extraction ────────────────────────────────────────────────────────

_ARTICLE_TYPES = {"Article", "NewsArticle", "BlogPosting", "TechArticle", "ReportageNewsArticle"}

def _parse_json_ld(soup: BeautifulSoup) -> t.Dict[str, t.Any]:
    """Extract structured metadata from schema.org JSON-LD blocks."""
    result: t.Dict[str, t.Any] = {}

    for script in soup.find_all("script", type="application/ld+json"):
        if not script.string:
            continue
        try:
            data = json.loads(script.string)
        except Exception:
            continue

        # Flatten @graph arrays and top-level objects into one list
        items: t.List[t.Any] = []
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("@graph", [data])

        for item in items:
            if not isinstance(item, dict):
                continue
            raw_type = item.get("@type", "")
            # @type can be a string or a list (e.g. ["ItemList", "Car"])
            type_set = set(raw_type) if isinstance(raw_type, list) else {raw_type}
            if not (type_set & _ARTICLE_TYPES):
                continue

            if not result.get("author"):
                author = item.get("author")
                if isinstance(author, list) and author:
                    first = author[0]
                    result["author"] = first.get("name") if isinstance(first, dict) else str(first)
                elif isinstance(author, dict):
                    result["author"] = author.get("name")
                elif isinstance(author, str):
                    result["author"] = author

            if not result.get("published_at"):
                result["published_at"] = _parse_datetime(item.get("datePublished"))

            if not result.get("keywords"):
                kw = item.get("keywords")
                if isinstance(kw, list):
                    result["keywords"] = [str(k) for k in kw if k]
                elif isinstance(kw, str):
                    result["keywords"] = _split_csv(kw)

            if not result.get("section"):
                section = item.get("articleSection")
                if isinstance(section, list) and section:
                    result["section"] = section[0]
                elif isinstance(section, str):
                    result["section"] = section

            if not result.get("language"):
                result["language"] = item.get("inLanguage")

    return result


# ── fast regex path ───────────────────────────────────────────────────────────

def _re_prop(html: str, prop: str) -> t.Optional[str]:
    for pat in (
        rf'<meta[^>]+property=["\'][^"\']*{re.escape(prop)}[^"\']*["\'][^>]+content=["\']([^"\']+)["\']',
        rf'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\'][^"\']*{re.escape(prop)}[^"\']*["\']',
    ):
        m = re.search(pat, html, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def _re_name(html: str, name: str) -> t.Optional[str]:
    for pat in (
        rf'<meta[^>]+name=["\'][^"\']*{re.escape(name)}[^"\']*["\'][^>]+content=["\']([^"\']+)["\']',
        rf'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\'][^"\']*{re.escape(name)}[^"\']*["\']',
    ):
        m = re.search(pat, html, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def fast_extract_metadata(html: str, url: str) -> t.Dict[str, t.Any]:
    title = _re_prop(html, "og:title") or _re_prop(html, "twitter:title")
    if not title:
        m = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        title = m.group(1).strip() if m else None

    # Prefer twitter:image (always a single, specific tag) over the first og:image
    # which may be a generic site logo when a page has multiple og:image entries.
    twitter_img = _re_name(html, "twitter:image") or _re_prop(html, "twitter:image")
    og_img = _re_prop(html, "og:image")
    image = twitter_img or og_img
    if image:
        image = urljoin(url, image)

    canonical = _re_prop(html, "og:url")
    if not canonical:
        m = re.search(r'<link[^>]+rel=["\']canonical["\'][^>]+href=["\']([^"\']+)["\']', html, re.IGNORECASE)
        if not m:
            m = re.search(r'<link[^>]+href=["\']([^"\']+)["\'][^>]+rel=["\']canonical["\']', html, re.IGNORECASE)
        canonical = m.group(1).strip() if m else None

    lang_m = re.search(r'<html[^>]+lang=["\']([^"\']+)["\']', html, re.IGNORECASE)
    language = (
        (lang_m.group(1).strip() if lang_m else None)
        or _re_prop(html, "og:locale")
        or _re_name(html, "language")
    )

    # Merge tags from mrf:tags (prefixed semicolons) and post-tags (csv)
    tags: t.List[str] = []
    mrf = _re_prop(html, "mrf:tags")
    if mrf:
        tags = _parse_prefixed_tags(mrf)
    if not tags:
        tags = _split_csv(_re_name(html, "post-tags"))

    categories = (
        _split_csv(_re_name(html, "post-cats"))
        or _split_csv(_re_prop(html, "mrf:sections"))
        or _split_csv(_re_name(html, "categories"))
        or _split_csv(_re_name(html, "category"))
    )

    return {
        "url": url,
        "canonical_url": canonical,
        "title": title,
        "thumbnail": image,
        "description": _re_prop(html, "og:description") or _re_prop(html, "twitter:description") or _re_name(html, "description"),
        "site_name": _re_prop(html, "og:site_name") or _re_name(html, "application-name"),
        "content_type": _re_prop(html, "og:type"),
        "author": None,
        "published_at": None,
        "keywords": _split_csv(_re_name(html, "keywords")),
        "categories": categories,
        "section": _re_prop(html, "mrf:sections"),
        "tags": tags,
        "language": language,
    }


# ── BeautifulSoup path ────────────────────────────────────────────────────────

def extract_metadata(html: str, url: str) -> t.Dict[str, t.Any]:
    soup = BeautifulSoup(html, "html.parser")

    def get_meta(*keys: str) -> t.Optional[str]:
        for key in keys:
            for attr in ("property", "name", "itemprop"):
                tag = soup.find("meta", attrs={attr: key})
                if tag and tag.get("content"):  # type: ignore
                    return tag["content"].strip()  # type: ignore
        return None

    def get_meta_all(*keys: str) -> t.List[str]:
        seen: t.Set[str] = set()
        out: t.List[str] = []
        for key in keys:
            for attr in ("property", "name", "itemprop"):
                for tag in soup.find_all("meta", attrs={attr: key}):
                    val = (tag.get("content") or "").strip()
                    if val and val not in seen:
                        seen.add(val)
                        out.append(val)
        return out

    # TITLE
    title = (
        get_meta("og:title", "twitter:title", "name")
        or (soup.title.string.strip() if soup.title else None)  # type: ignore
    )

    # THUMBNAIL — when multiple og:image tags exist (e.g. generic site logo + product
    # images), twitter:image is always a single specific tag so prefer it.
    og_images = [
        tag["content"].strip()
        for tag in soup.find_all("meta", property="og:image")
        if tag.get("content")
    ]
    twitter_img = get_meta("twitter:image")
    if len(og_images) > 1 and twitter_img:
        image = twitter_img
    elif og_images:
        image = og_images[0]
    else:
        image = twitter_img or get_meta("image", "thumbnail")
    if not image:
        link_img = soup.find("link", rel="image_src")
        if link_img and link_img.get("href"):  # type: ignore
            image = link_img["href"]  # type: ignore
    if image:
        image = urljoin(url, image)  # type: ignore

    # CANONICAL URL
    canonical = get_meta("og:url", "twitter:url", "url", "mrf:canonical")
    if not canonical:
        link_can = soup.find("link", rel="canonical")
        if link_can and link_can.get("href"):  # type: ignore
            canonical = link_can["href"].strip()  # type: ignore

    # DESCRIPTION
    description = get_meta("og:description", "twitter:description", "description")

    # SITE NAME
    site_name = get_meta("og:site_name", "application-name")

    # CONTENT TYPE
    content_type = get_meta("og:type")

    # TAGS  — mrf:tags (prefixed semicolons) → post-tags (csv) → article:tag
    tags: t.List[str] = _parse_prefixed_tags(get_meta("mrf:tags"))
    if not tags:
        tags = _split_csv(get_meta("post-tags"))
    if not tags:
        tags = get_meta_all("article:tag")

    # KEYWORDS  — meta keywords (csv)
    keywords = _split_csv(get_meta("keywords"))

    # CATEGORIES  — post-cats → mrf:sections → categories/category meta
    categories = (
        _split_csv(get_meta("post-cats"))
        or _split_csv(get_meta("mrf:sections"))
        or _split_csv(get_meta("categories", "category"))
    )

    # SECTION  — mrf:sections (first value) → article:section
    section = (
        (categories[0] if categories else None)
        or get_meta("article:section", "mrf:sections")
    )

    # AUTHOR
    author = get_meta("article:author", "author", "twitter:creator")

    # PUBLISHED DATE
    published_at = _parse_datetime(
        get_meta("article:published_time", "article:modified_time", "date", "pubdate", "publish_date", "og:updated_time")
    )
    if not published_at:
        time_tag = soup.find("time", attrs={"datetime": True})
        if time_tag:
            published_at = _parse_datetime(time_tag["datetime"])  # type: ignore

    # LANGUAGE — html[lang] → og:locale → meta[name=language]
    html_tag = soup.find("html")
    language = (html_tag.get("lang") or "").strip() or None if html_tag else None  # type: ignore
    if not language:
        language = get_meta("og:locale") or get_meta("language")

    # JSON-LD  — fills gaps left by meta tags
    ld = _parse_json_ld(soup)
    author = author or ld.get("author")
    published_at = published_at or ld.get("published_at")
    keywords = keywords or ld.get("keywords", [])
    section = section or ld.get("section")
    language = language or ld.get("language")
    for kw in ld.get("keywords", []):
        if kw not in tags:
            tags.append(kw)

    # window.pageInfo  — fills gaps for sites like Kansas City Star / McClatchy
    pi = _parse_page_info(html)
    keywords = keywords or pi.get("keywords", [])
    section = section or pi.get("section")
    if not categories and pi.get("topic"):
        categories = [pi["topic"]]

    return {
        "url": url,
        "canonical_url": canonical,
        "title": title,
        "thumbnail": image,
        "description": description,
        "site_name": site_name,
        "content_type": content_type,
        "author": author,
        "published_at": published_at,
        "keywords": keywords,
        "categories": categories,
        "section": section,
        "tags": tags,
        "language": language,
    }


# ── window.pageInfo extraction (McClatchy / KC Star pattern) ─────────────────

def _parse_page_info(html: str) -> t.Dict[str, t.Any]:
    """Extract metadata embedded in window.pageInfo = {...} JS blocks."""
    result: t.Dict[str, t.Any] = {}
    if "window.pageInfo" not in html:
        return result

    # keywordSchemas.hum — human-curated topic keywords, best signal
    hum_m = re.search(r"'hum'\s*:\s*\[([^\]]+)\]", html)
    if hum_m:
        keywords = re.findall(r"'([^']+)'", hum_m.group(1))
        if keywords:
            result["keywords"] = keywords

    # marketInfo.clipped_taxonomy — clean section label e.g. 'Sports'
    tax_m = re.search(r"'marketInfo\.clipped_taxonomy'\s*:\s*'([^']+)'", html)
    if tax_m and tax_m.group(1).strip():
        result["section"] = tax_m.group(1).strip()

    # mcc_story_topics — topic/category code e.g. 'PRO_SPORTS_FOOTBALL'
    topic_m = re.search(r"'mcc_story_topics'\s*:\s*'([^']+)'", html)
    if topic_m and topic_m.group(1).strip():
        result["topic"] = topic_m.group(1).strip()

    return result


# ── bot/challenge detection ───────────────────────────────────────────────────

_CHALLENGE_TITLES = frozenset({
    "just a moment",
    "attention required",
    "please wait",
    "access denied",
    "ddos-guard",
    "security check",
    "enable javascript",
    "checking your browser",
})

_CHALLENGE_MARKERS = (
    'cf-browser-verification',
    '__cf_chl_opt',
    'id="cf-wrapper"',
    'id="ddos-guard"',
    'data-translate="checking_browser"',
)

def _is_challenge_page(data: t.Dict[str, t.Any], html: str) -> bool:
    title = (data.get("title") or "").lower().strip(" .")
    if any(ct in title for ct in _CHALLENGE_TITLES):
        return True
    return any(marker in html for marker in _CHALLENGE_MARKERS)


# ── fetch helpers ─────────────────────────────────────────────────────────────

# Tried in order until one returns valid metadata.
# Social/bot crawlers are whitelisted by Cloudflare and many CDNs.
_USER_AGENTS: t.List[t.Tuple[str, str]] = [
    ("chrome", (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )),
    ("slack",     "Slackbot-LinkExpanding 1.0 (+https://api.slack.com/robots)"),
    ("facebook",  "facebookexternalhit/1.1 (+http://www.facebook.com/externalhit_uatext.php)"),
    ("twitter",   "Twitterbot/1.0"),
    ("linkedin",  "LinkedInBot/1.0 (compatible; Mozilla/5.0; Apache-HttpClient/4.x)"),
    ("whatsapp",  "WhatsApp/2.24.10 A"),
]

_BASE_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
}


def _enrich_from_page_info(data: t.Dict[str, t.Any], html: str) -> t.Dict[str, t.Any]:
    pi = _parse_page_info(html)
    if not data.get("keywords") and pi.get("keywords"):
        data["keywords"] = pi["keywords"]
    if not data.get("section") and pi.get("section"):
        data["section"] = pi["section"]
    if not data.get("categories") and pi.get("topic"):
        data["categories"] = [pi["topic"]]
    return data


def _extract_from_html(html: str, url: str, label: str) -> t.Dict[str, t.Any]:
    fast_data = fast_extract_metadata(html[:50000], url)
    if _is_challenge_page(fast_data, html[:50000]):
        raise RuntimeError(f"Bot challenge page ({label})")
    if fast_data["title"] and fast_data["thumbnail"]:
        return _enrich_from_page_info(fast_data, html)
    logger.info(f"[Fast Extract] missed title/thumbnail — falling back to BS4 ({label})")
    data = extract_metadata(html, url)
    if _is_challenge_page(data, html[:10000]):
        raise RuntimeError(f"Bot challenge page ({label})")
    return data


async def _fetch_with_ua(
    url: str,
    name: str,
    ua: str,
    timeout: int = 10,
    proxy: t.Optional[str] = None,
) -> t.Dict[str, t.Any]:
    headers = {**_BASE_HEADERS, "User-Agent": ua}
    if name == "chrome":
        headers["Referer"] = "https://www.google.com/"
    proxies = {"http": proxy, "https": proxy} if proxy else None
    resp = await asyncio.to_thread(
        requests.get, url, headers=headers, timeout=timeout, proxies=proxies
    )
    resp.raise_for_status()
    try:
        html = resp.text
    except Exception:
        html = resp.content.decode("utf-8", errors="ignore")
    return _extract_from_html(html, url, name)


async def fetch_metadata_cloudscraper(
    url: str,
    timeout: int = 15,
    proxy: t.Optional[str] = None,
) -> t.Dict[str, t.Any]:
    def _fetch():
        scraper = cloudscraper.create_scraper(  # type: ignore
            browser={"browser": "chrome", "platform": "linux", "mobile": False}
        )
        proxies = {"http": proxy, "https": proxy} if proxy else None
        return scraper.get(url, timeout=timeout, proxies=proxies).text

    html = await asyncio.to_thread(_fetch)
    return _extract_from_html(html, url, "cloudscraper")


async def get_url_preview(
    url: str,
    proxies: t.Optional[t.List[str]] = None,
) -> t.Optional[t.Dict[str, t.Any]]:
    """Fetch Open Graph / meta preview for *url*.

    Args:
        url: The page to fetch.
        proxies: List of proxy URLs to rotate through. Defaults to
            ``WEBSHARE_PROXIES`` (loaded from env or the hardcoded list).
            Pass ``[]`` to force a direct connection with no proxy.
    """
    if proxies is None:
        proxies = WEBSHARE_PROXIES
    # Always include a direct-connection slot as final fallback
    proxy_slots: t.List[t.Optional[str]] = list(proxies) + [None]

    for name, ua in _USER_AGENTS:
        for proxy in proxy_slots:
            label = f"{name}{'@proxy' if proxy else ''}"
            try:
                data = await _fetch_with_ua(url, name, ua, proxy=proxy)
                logger.info(f"[{label}] succeeded for {url}")
                return data
            except Exception:
                pass

    for proxy in proxy_slots:
        label = f"cloudscraper{'@proxy' if proxy else ''}"
        try:
            data = await fetch_metadata_cloudscraper(url, proxy=proxy)
            logger.info(f"[{label}] succeeded for {url}")
            return data
        except Exception as e:
            logger.info(f"[{label}] failed for {url}: {e}")

    logger.warning(f"[get_url_preview] all fetchers failed for {url} — returning None")
    return None
