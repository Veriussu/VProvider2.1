# ─────────────────────────────────────────────────────────────
#  Modül:    web_tools.py
#  Amaç:     VProvider2.1 — yerel modeller için "internet erişimi"
#            araçları. Yalnızca arama/fetch anında ağ kullanılır;
#            arka planda hiçbir iş/kaynak tüketimi yoktur.
#            Tamamen ücretsiz (DuckDuckGo web API'si, anahtar yok).
#  Bağımlılık: httpx yalnızca çağrı anında içe aktarılır (lazy).
# ─────────────────────────────────────────────────────────────
import html
import ipaddress
import json
import re
import socket
from typing import Any
from urllib.parse import unquote, urljoin, urlparse

from .config import settings

# Boş bekleyen güvenlik zamanı (SSRF için yok; fetch sınırlı kap)
_FETCH_HOSTNAME_CHARS = 253

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0 "
    "VProvider/2.1"
)

# HTML sorgusu bot-korumasına takılmaması için tam tarayıcı başlıkları
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
    "sec-ch-ua": '"Chromium";v="126", "Google Chrome";v="126", "Not-A.Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Linux"',
    "Upgrade-Insecure-Requests": "1",
    "Referer": "https://duckduckgo.com/",
}

# Sayfa çekme (fetch_url) başlıkları — genel tarayıcı görünümü
_FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
    "Upgrade-Insecure-Requests": "1",
}

# ------------------------------------------------------------------
# Açık/kapalı durumu (DB'de kalıcı; DB yoksa/yazılamıyorsa ayar değeri)
# ------------------------------------------------------------------


def is_enabled() -> bool:
    """İnternet erişimi durumu: DB toggle önceliklidir."""
    from .user_store import get_store

    try:
        raw = get_store().get_setting("web_tools_enabled")
    except Exception:
        raw = ""
    if raw in ("1", "true", "on"):
        return True
    if raw in ("0", "false", "off"):
        return False
    return settings.web_tools_enabled


def set_enabled(value: bool) -> None:
    """Toggle durumunu DB'de kalıcılaştırır."""
    from .user_store import get_store

    get_store().set_setting("web_tools_enabled", "1" if value else "0")


# ------------------------------------------------------------------
# Araç şemaları (OpenAI biçimi)
# ------------------------------------------------------------------

_TOOL_WEB_SEARCH = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "İnternette anahtar kelimeyle (DuckDuckGo) ücretsiz arama yapar. "
            "Metin sonuçları ve görseller döner. Yanıtında bulduğu görselleri "
            "![başlık](resim_bağlantısı) biçiminde ekleyebilirsin. "
            "Güncel/haber/olgu sorusu geldiğinde mutlaka bu aracı kullan."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Aranacak anahtar kelime / soru (Türkçe veya İngilizce).",
                }
            },
            "required": ["query"],
        },
    },
}

_TOOL_FETCH_URL = {
    "type": "function",
    "function": {
        "name": "fetch_url",
        "description": (
            "Verilen bir web sayfasını çeker ve metin içeriğini döner. "
            "Yalnızca genel (internet erişilebilir) adresler okunabilir; "
            "yerel ağ adresleri engellenir (güvenlik). Sayfadaki resim "
            "bağlantıları da listelenir."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Okunacak sayfanın tam adresi (http/https).",
                }
            },
            "required": ["url"],
        },
    },
}


def tool_specs() -> list[dict[str, Any]]:
    """İnternet araçlarının OpenAI uyumlu şema listesi."""
    return [json.loads(json.dumps(_TOOL_WEB_SEARCH)), json.loads(json.dumps(_TOOL_FETCH_URL))]


def _tool_names() -> set[str]:
    return {"web_search", "fetch_url"}


def is_builtin(tool_call: dict) -> bool:
    """Araç çağrısının yerleşik internet aracı olup olmadığını söyler."""
    fn = (tool_call.get("function") or {}) if isinstance(tool_call, dict) else {}
    return fn.get("name") in _tool_names()


# ------------------------------------------------------------------
# Gemma tarzı metin araç çağrısı ayrıştırıcısı
# ------------------------------------------------------------------

_GEMMA_CALL_RE = re.compile(
    r"<\|tool_call>call:([A-Za-z_][A-Za-z0-9_]*)\{(.*?)\}<tool_call\|>", re.S
)
_GEMMA_ARG_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*):<\|"\|>(.*?)<\|"\|>', re.S)


def _parse_gemma_args(body: str) -> dict:
    """Gemma4 biçimindeki argüman metnini sözlüğe çevirir.

    Örnek: query:<|"|>Türkiye'nin başkenti?<|"|>  ->  {"query": "..."}
    """
    out: dict[str, Any] = {}
    for m in _GEMMA_ARG_RE.finditer(body or ""):
        out[m.group(1)] = m.group(2)
    return out


def parse_tool_calls(content: str) -> list[dict[str, Any]]:
    """Modelin metin olarak ürettiği araç çağrılarını yakalar.

    Bazı modeller (ör. Gemma 4) araç çağrısını yapılandırılmış JSON yerine
    '<|tool_call>call:ad{...}<tool_call|>' metni olarak üretir. Bu işlev bu
    işaretçileri OpenAI işlev çağrısı biçimine çevirir (argumentler sözlük).
    """
    calls: list[dict[str, Any]] = []
    for m in _GEMMA_CALL_RE.finditer(content or ""):
        calls.append(
            {
                "id": f"call_web_{len(calls) + 1}",
                "type": "function",
                "function": {"name": m.group(1), "arguments": _parse_gemma_args(m.group(2))},
            }
        )
    return calls


# ------------------------------------------------------------------
# Gale/argüman ayrıştırıcıları
# ------------------------------------------------------------------

def _parse_args(arguments: Any) -> dict:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            return json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            return {}
    return {}


def run_tool(tool_call: dict) -> str:
    """Bir araç çağrısını çalıştırır; sonucu modelin okuyacağı metni döner."""
    fn = (tool_call.get("function") or {}) if isinstance(tool_call, dict) else {}
    name = fn.get("name")
    args = _parse_args(fn.get("arguments"))
    try:
        if name == "web_search":
            return web_search(args.get("query", ""))
        if name == "fetch_url":
            return fetch_url(args.get("url", ""))
        return f"Hata: bilinmeyen araç '{name}'."
    except Exception as exc:  # noqa: BLE001  — sonuç metni olarak model dönsün
        return f"Araç hatası: {exc}"


# ------------------------------------------------------------------
# SSRF koruması
# ------------------------------------------------------------------

def _is_private(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return True
    if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
        return True
    if addr.is_multicast or addr.is_unspecified:
        return True
    if isinstance(addr, ipaddress.IPv6Address) and (addr.ipv4_mapped is not None):
        return _is_private(str(addr.ipv4_mapped))
    return False


def _ssrf_blocked(url: str) -> tuple[bool, str]:
    """Adresin yerel/özel ağa (internal) gidip gitmediğini kontrol eder.

    Dönüş: (engellendi mi, sebep metni). Görünen ad DNS çözülüp her IP
    ayrı ayrı denetlenir; böylece DNS rebinding riski de önlenir.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return True, "yalnızca http/https adresleri desteklenir."
    host = parsed.hostname or ""
    if not host:
        return True, "geçersiz adres."
    if len(host) > _FETCH_HOSTNAME_CHARS:
        return True, "adres uzunluğu geçersiz."
    # Görünen adı önce sözlüksel kontrol (hızlı yol)
    host_lower = host.lower().rstrip(".")
    if host_lower == "localhost" or host_lower.endswith(".local"):
        return True, "yerel makineye erişim yasak."
    try:
        ip = ipaddress.ip_address(host_lower)
        if _is_private(str(ip)):
            return True, "yerel/özel ağ adresine erişim yasak."
    except ValueError:
        pass  # alan adı; DNS üzerinden çözülecek

    # DNS çözümle (tüm IP'leri denetle)
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror:
        return True, "alan adı çözülemedi (DNS hatası)."
    seen: set[str] = set()
    for info in infos:
        ip = info[4][0]
        if ip in seen:
            continue
        seen.add(ip)
        if _is_private(ip):
            return True, "hedef adres yerel/özel ağa çözülüyor; erişim yasak."
    return False, ""


# ------------------------------------------------------------------
# sayfa çekme (fetch_url)
# ------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<script[^>]*>.*?</script>|<style[^>]*>.*?</style>", re.S | re.I)
_WS_RE = re.compile(r"[ \t\r\f\v]{2,}|\n{3,}")
_BAD_SCHEME_RE = re.compile(r"[\x00-\x1f\x7f]")


def _html_to_text(raw: str, max_chars: int) -> str:
    raw = _BAD_SCHEME_RE.sub("", raw)
    raw = _SCRIPT_RE.sub(" ", raw)
    text = _TAG_RE.sub(" ", raw)
    text = html.unescape(text)
    text = _WS_RE.sub(" ", text).strip()
    return text[:max_chars]


def _images_from_html(raw: str, base: str, max_images: int = 6) -> list[str]:
    """Sayfadaki görsel bağlantılarını toplar (göreli adresler base'e çevrilir)."""
    urls: list[str] = []
    for m in re.finditer(r'<img[^>]+src\s*=\s*["\']([^"\']+)["\']', raw, re.I):
        src = html.unescape(m.group(1))
        full = urljoin(base, src) if not src.startswith(("data:", "blob:")) else ""
        if full and full.startswith(("http://", "https://")) and full not in urls:
            urls.append(full)
        if len(urls) >= max_images:
            break
    return urls


def fetch_url(url: str, max_chars: int | None = None) -> str:
    """Bir sayfayı güvenli biçimde çeker; metin içeriğini döner."""
    import httpx  # lazy: yalnızca kullanım anında

    url = (url or "").strip()
    if not url:
        return "Hata: url parametresi boş."
    if not url.startswith(("http://", "https://")):
        return "Hata: url, http:// veya https:// ile başlamalı."
    max_chars = max_chars or settings.web_fetch_max_chars
    max_bytes = settings.web_fetch_max_bytes
    timeout = settings.web_tool_timeout

    blocked, reason = _ssrf_blocked(url)
    if blocked:
        return f"Hata: sayfa okunamadı — {reason}"

    current = url
    redirects = 0
    final_text: list[str] = []
    with httpx.Client(
        headers=_FETCH_HEADERS,
        timeout=timeout,
        follow_redirects=False,
        http2=True,
    ) as client:
        while True:
            blocked, reason = _ssrf_blocked(current)
            if blocked:
                return f"Hata: yönlendirme engellendi — {reason}"
            try:
                resp = client.get(current)
            except Exception as exc:  # noqa: BLE001
                return f"Hata: sayfa çekilemedi — {exc}"
            if resp.status_code in (301, 302, 303, 307, 308):
                if redirects >= 5:
                    return "Hata: çok fazla yönlendirme."
                redirects += 1
                current = urljoin(current, resp.headers.get("location", ""))
                continue
            if resp.status_code != 200:
                    return f"Hata: sayfa durum kodu {resp.status_code}."
            body = resp.content[:max_bytes]
            ctype = resp.headers.get("content-type", "").lower()
            if "json" in ctype:
                try:
                    parsed = json.loads(body.decode("utf-8", "replace"))
                    chunk = json.dumps(parsed, ensure_ascii=False, indent=1)[:max_chars]
                except json.JSONDecodeError:
                    chunk = body.decode("utf-8", "replace")[:max_chars]
            elif "html" in ctype or ctype.startswith("text/") or not ctype:
                raw = body.decode("utf-8", "replace")
                chunk = _html_to_text(raw, max_chars)
                final_text.append(f"Adres: {current}\nİçerik:\n{chunk}")
                imgs = _images_from_html(raw, current)
                if imgs:
                    final_text.append(
                        "Sayfadaki görseller:\n" + "\n".join(f"- {u}" for u in imgs)
                    )
            else:
                chunk = body.decode("utf-8", "replace")[:max_chars]
                final_text.append(f"Adres: {current}\nİçerik:\n{chunk}")
            break
    return "\n\n".join(final_text)


# ------------------------------------------------------------------
# DuckDuckGo arama (web_search) — anahtar gerektirmeyen uçlar
# ------------------------------------------------------------------

def _ddg_vqd(query: str, timeout: float) -> str:
    import httpx  # lazy

    with httpx.Client(headers={"User-Agent": _UA}, timeout=timeout, follow_redirects=True) as client:
        resp = client.get("https://duckduckgo.com/", params={"q": query})
        m = re.search(r"vqd=([\d\-]+)", resp.text)
        return m.group(1) if m else ""


def _ddg_text(query: str, max_results: int, region: str, timeout: float) -> list[dict]:
    import httpx  # lazy

    lines: list[dict] = []
    with httpx.Client(
        headers=_BROWSER_HEADERS,
        timeout=timeout,
        follow_redirects=True,
    ) as client:
        resp = client.get("https://html.duckduckgo.com/html/", params={"q": query, "kl": region})
        if resp.status_code != 200:
            return lines
    results = re.findall(
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        resp.text,
        re.S,
    )
    snippets = re.findall(
        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', resp.text, re.S
    )
    for i, (href, title_raw) in enumerate(results[:max_results]):
        url = _uddg_decode(href)
        title = _html_to_text(title_raw, 200)
        snippet = _html_to_text(snippets[i], 400) if i < len(snippets) else ""
        lines.append({"title": title, "url": url, "snippet": snippet})
    return lines


def _uddg_decode(href: str) -> str:
    """DuckDuckGo html aramasındaki yönlendirme bağlantısını asıl adrese çevirir."""
    decoded = html.unescape(href)
    m = re.search(r"[?&]uddg=([^&]+)", decoded)
    if m:
        return unquote(m.group(1))
    if decoded.startswith("//"):
        return "https:" + decoded
    return decoded


def _ddg_images(query: str, count: int, region: str, timeout: float) -> list[dict]:
    import httpx  # lazy

    vqd = _ddg_vqd(query, timeout) or _ddg_vqd(query[:3], timeout)
    if not vqd:
        return []
    out: list[dict] = []
    with httpx.Client(
        headers={"User-Agent": _UA, "Referer": "https://duckduckgo.com/"},
        timeout=timeout,
        follow_redirects=True,
    ) as client:
        resp = client.get(
            "https://duckduckgo.com/i.js",
            params={
                "l": region,
                "o": "json",
                "q": query,
                "vqd": vqd,
                "f": ",,,",
                "p": "1",
            },
        )
        if resp.status_code != 200:
            return []
        try:
            payload = resp.json()
        except Exception:  # noqa: BLE001
            return []
    seen: set[str] = set()
    for item in payload.get("results", []):
        thumb = item.get("thumbnail") or item.get("image") or ""
        if not thumb or thumb in seen:
            continue
        if not thumb.startswith(("http://", "https://")):
            continue
        seen.add(thumb)
        out.append(
            {
                "title": (item.get("title") or "").strip() or query,
                "thumbnail": thumb,
                "image": item.get("image") or thumb,
                "source": item.get("url") or "",
                "width": item.get("width"),
                "height": item.get("height"),
            }
        )
        if len(out) >= count:
            break
    return out


def web_search(query: str, max_results: int | None = None, max_images: int | None = None) -> str:
    """DuckDuckGo'da ücretsiz metin + görsel araması yapar.

    Sonuç modelin okuyabileceği düz metin olarak döner; görseller
    ![başlık](adres) biçiminde eklenir.
    """
    query = (query or "").strip()
    if not query:
        return "Hata: query parametresi boş."
    max_results = max_results or settings.web_search_max_results
    max_images = max_images if max_images is not None else settings.web_search_max_images
    region = settings.web_search_region
    timeout = settings.web_tool_timeout

    text_lines = _ddg_text(query, max_results, region, timeout)
    images = _ddg_images(query, max_images, region, timeout)

    parts: list[str] = [f'İnternet arama sonuçları ("{query}"):']
    if not text_lines and not images:
        parts.append("Sonuç bulunamadı.")
    for i, res in enumerate(text_lines, 1):
        parts.append(
            f"{i}. {res['title']}\n   URL: {res['url']}\n   Özet: {res['snippet'] or '-'}"
        )
    if images:
        parts.append("\nGörseller (bulundu, anahtar bilgi olabilir):")
        for i, im in enumerate(images, 1):
            alt = im["title"]
            parts.append(f"{i}. ![ {alt} ]({im['thumbnail']})  (kaynak: {im['source']})")
    return "\n".join(parts)