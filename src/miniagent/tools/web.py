"""网络工具：web_search 与 web_fetch。

只用 httpx 发请求，HTML 解析走标准库 HTMLParser；
所有 URL 先解析 DNS 再校验 IP，只连接校验过的地址（Host / SNI 仍用原域名），
避免"字符串看起来合法、解析后落到内网"的绕过。
"""

from __future__ import annotations

import ipaddress
import json
import re
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any
from urllib.parse import SplitResult, urljoin, urlsplit, urlunsplit

import httpx

from miniagent.errors import ToolError
from miniagent.tools.base import MAX_OUTPUT_CHARS, ToolOutput, truncate_output

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
DEFAULT_TIMEOUT = 10.0
MAX_REDIRECTS = 3
MAX_RESPONSE_BYTES = 512 * 1024
MAX_SEARCH_RESULTS = 10
MAX_FETCH_CHARS = MAX_OUTPUT_CHARS
TEXT_TYPES = ("text/html", "application/xhtml+xml", "text/plain", "application/json")
SKIP_TAGS = {"script", "style", "noscript", "template", "svg"}
BLOCK_TAGS = {
    "p",
    "div",
    "section",
    "article",
    "header",
    "footer",
    "li",
    "tr",
    "br",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "pre",
    "blockquote",
}

Resolver = Callable[[str, int], list[str]]


def _parse_ip(raw: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    """把带 zone 后缀的地址转成 ipaddress；IPv4-mapped IPv6 还原成 IPv4 再判断。"""
    ip = ipaddress.ip_address(raw.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    return ip


def _is_public(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """is_global 已覆盖回环、私网、链路本地、保留段、组播与未指定地址。"""
    return ip.is_global


def _resolve(host: str, port: int) -> list[str]:
    """默认解析器：返回该主机的全部地址，由调用方统一做公网校验。"""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ToolError(f"域名解析失败：{host}（{exc}）") from exc
    addresses: list[str] = []
    for info in infos:
        address = info[4][0]
        if address not in addresses:
            addresses.append(address)
    if not addresses:
        raise ToolError(f"域名解析结果为空：{host}")
    return addresses


def _check_url(url: str) -> SplitResult:
    """只允许 http/https，且不允许在 URL 里夹带用户名密码。"""
    parts = urlsplit(url)
    if parts.scheme.lower() not in {"http", "https"}:
        raise ToolError(f"只允许 http/https 链接，收到：{url[:120]}")
    if not parts.hostname:
        raise ToolError(f"链接缺少主机名：{url[:120]}")
    if parts.username or parts.password:
        raise ToolError("链接不允许携带用户名或密码")
    try:
        _ = parts.port
    except ValueError as exc:
        raise ToolError(f"链接端口非法：{url[:120]}") from exc
    return parts


def _host_header(host: str, port: int, scheme: str) -> str:
    default_port = 443 if scheme == "https" else 80
    shown = f"[{host}]" if ":" in host else host
    return shown if port == default_port else f"{shown}:{port}"


def _short(text: str, limit: int = 80) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "..."


@dataclass
class FetchResult:
    url: str
    status: int
    content_type: str
    text: str
    size: int
    redirects: int
    truncated: bool


class WebFetcher:
    """带 SSRF 防护的 HTTP 客户端；transport / resolver 可注入以便离线测试。"""

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        max_bytes: int = MAX_RESPONSE_BYTES,
        max_redirects: int = MAX_REDIRECTS,
        resolver: Resolver | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.max_redirects = max_redirects
        self._resolver = resolver or _resolve
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout),
            follow_redirects=False,
            trust_env=False,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def get(self, url: str, *, allow_private: bool = False) -> FetchResult:
        """抓取 URL。allow_private 只给运维配置的检索后端用，绝不用于模型传入的地址。"""
        return self.request("GET", url, allow_private=allow_private)

    def post_json(
        self, url: str, payload: dict[str, Any], *, allow_private: bool = False
    ) -> FetchResult:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return self.request(
            "POST",
            url,
            body=body,
            content_type="application/json",
            allow_private=allow_private,
        )

    def request(
        self,
        method: str,
        url: str,
        *,
        body: bytes | None = None,
        content_type: str = "",
        allow_private: bool = False,
    ) -> FetchResult:
        current = _check_url(url)
        try:
            return self._request(method, current, body, content_type, allow_private)
        except httpx.TimeoutException as exc:
            raise ToolError(f"请求超时（{self.timeout:g}s）") from exc
        except httpx.HTTPError as exc:
            raise ToolError(f"请求失败：{type(exc).__name__}: {exc}") from exc

    def _request(
        self,
        method: str,
        current: SplitResult,
        body: bytes | None,
        content_type: str,
        allow_private: bool,
    ) -> FetchResult:
        redirects = 0
        while True:
            scheme = current.scheme.lower()
            host = current.hostname or ""
            port = current.port or (443 if scheme == "https" else 80)
            addresses = self._resolver(host, port)
            if not allow_private:
                for address in addresses:
                    ip = _parse_ip(address)
                    if not _is_public(ip):
                        raise ToolError(
                            f"拒绝访问内网或本机地址：{host} 解析到 {ip}"
                        )
            ip = addresses[0]
            netloc = f"[{ip}]" if ":" in ip else ip
            default_port = 443 if scheme == "https" else 80
            if port != default_port:
                netloc = f"{netloc}:{port}"
            target = urlunsplit(
                (scheme, netloc, current.path or "/", current.query, "")
            )
            headers = {
                "Host": _host_header(host, port, scheme),
                "User-Agent": "miniagent/0.1 (+local)",
                "Accept": (
                    "text/html,application/xhtml+xml,text/plain,"
                    "application/json;q=0.9,*/*;q=0.5"
                ),
            }
            if content_type:
                headers["Content-Type"] = content_type
            extensions = {"sni_hostname": host} if scheme == "https" else {}
            with self._client.stream(
                method, target, headers=headers, content=body, extensions=extensions
            ) as response:
                status = response.status_code
                if status in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location or method != "GET":
                        return FetchResult(
                            current.geturl(), status, "", "", 0, redirects, False
                        )
                    if redirects >= self.max_redirects:
                        raise ToolError(f"重定向次数超过上限（{self.max_redirects} 次）")
                    current = _check_url(urljoin(current.geturl(), location))
                    redirects += 1
                    continue
                content_type = response.headers.get("content-type", "")
                raw = bytearray()
                truncated = False
                for chunk in response.iter_bytes():
                    remaining = self.max_bytes - len(raw)
                    if remaining <= 0:
                        truncated = True
                        break
                    if len(chunk) > remaining:
                        raw.extend(chunk[:remaining])
                        truncated = True
                        break
                    raw.extend(chunk)
                encoding = response.charset_encoding or "utf-8"
                try:
                    text = bytes(raw).decode(encoding, errors="replace")
                except LookupError:
                    text = bytes(raw).decode("utf-8", errors="replace")
                return FetchResult(
                    current.geturl(),
                    status,
                    content_type.split(";", 1)[0].strip().lower(),
                    text,
                    len(raw),
                    redirects,
                    truncated,
                )


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip = 0
        self._in_title = False
        self._parts: list[str] = []
        self.title = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in SKIP_TAGS:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIP_TAGS and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif not self._skip:
            self._parts.append(data)


def extract_text(markup: str) -> tuple[str, str]:
    """HTML 转纯文本：去脚本样式、块级元素换行、压缩空白。"""
    parser = _TextExtractor()
    parser.feed(markup)
    parser.close()
    text = "".join(parser._parts)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return " ".join(parser.title.split()), text.strip()


MISSING_KEY_HINT = (
    "未配置 Tavily API key：请设置环境变量 TAVILY_API_KEY，"
    "或在 config.toml 写入 tavily_api_key"
)


def _parse_tavily(payload: str, limit: int) -> list[dict[str, str]]:
    """Tavily 的 /search 返回 {results: [{title, url, content, score}]}。"""
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ToolError(f"Tavily 返回的不是合法 JSON：{exc}") from exc
    if not isinstance(data, dict):
        raise ToolError("Tavily 返回结构不是 JSON 对象")
    raw_results = data.get("results")
    if not isinstance(raw_results, list):
        raise ToolError("Tavily 返回结构缺少 results 数组")
    results: list[dict[str, str]] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        title = " ".join(str(item.get("title") or "").split())
        url = str(item.get("url") or "").strip()
        if not title or not url.startswith(("http://", "https://")):
            continue
        results.append(
            {
                "title": title,
                "url": url,
                "snippet": " ".join(str(item.get("content") or "").split()),
            }
        )
        if len(results) >= limit:
            break
    return results


class WebSearchTool:
    name = "web_search"
    description = (
        "联网搜索公开网页，返回标题、链接和摘要。适合查找最新信息或未知主题；"
        "需要正文时再用 web_fetch 抓取具体链接。"
    )
    read_only = True
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词"},
            "max_results": {
                "type": "integer",
                "description": f"返回条数，默认 5，最多 {MAX_SEARCH_RESULTS}",
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        fetcher: WebFetcher | None = None,
        *,
        api_key: str = "",
    ) -> None:
        self._fetcher = fetcher or WebFetcher()
        self._api_key = api_key.strip()

    def run(self, args: dict[str, Any]) -> ToolOutput:
        query = str(args.get("query") or "").strip()
        if not query:
            raise ToolError("查询词不能为空")
        limit = int(args.get("max_results", 5))
        limit = max(1, min(limit, MAX_SEARCH_RESULTS))
        if not self._api_key:
            # 搜索是可选能力：缺 key 只在真正调用时报错，不影响进程启动
            raise ToolError(MISSING_KEY_HINT)

        start = time.perf_counter()
        result = self._fetcher.post_json(
            TAVILY_SEARCH_URL,
            {
                "api_key": self._api_key,
                "query": query,
                "max_results": limit,
                "search_depth": "basic",
            },
        )
        elapsed = int((time.perf_counter() - start) * 1000)
        parsed = _parse_tavily(result.text, limit) if result.status == 200 else []
        detail = (
            f"query={_short(query)} backend=tavily status={result.status} "
            f"results={len(parsed)} bytes={result.size} ms={elapsed}"
        )
        if result.status != 200:
            return ToolOutput(
                f"搜索失败：HTTP {result.status}（Tavily）。"
                "可以稍后重试，或改用已知的直达链接配合 web_fetch。",
                detail,
                fail=True,
            )
        if not parsed:
            return ToolOutput(f"没有找到与“{query}”相关的结果。", detail)

        lines: list[str] = []
        for index, item in enumerate(parsed, start=1):
            lines.append(f"{index}. {item['title']}")
            lines.append(f"   {item['url']}")
            if item["snippet"]:
                lines.append(f"   {item['snippet']}")
        return ToolOutput(truncate_output("\n".join(lines)), detail)


class WebFetchTool:
    name = "web_fetch"
    description = (
        "抓取一个 http/https 网页并转成纯文本。只用于公开地址；"
        "内网、本机与重定向到内网的地址都会被拒绝。"
    )
    read_only = True
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "要抓取的 http/https 链接"},
            "max_chars": {
                "type": "integer",
                "description": f"最多返回字符数，默认 {MAX_FETCH_CHARS}",
            },
        },
        "required": ["url"],
    }

    def __init__(self, fetcher: WebFetcher | None = None) -> None:
        self._fetcher = fetcher or WebFetcher()

    def run(self, args: dict[str, Any]) -> ToolOutput:
        url = str(args.get("url") or "").strip()
        if not url:
            raise ToolError("URL 不能为空")
        limit = int(args.get("max_chars", MAX_FETCH_CHARS))
        limit = max(500, min(limit, MAX_FETCH_CHARS))
        start = time.perf_counter()
        result = self._fetcher.get(url)
        elapsed = int((time.perf_counter() - start) * 1000)
        detail = (
            f"url={_short(url)} status={result.status} type={result.content_type or '-'} "
            f"bytes={result.size} redirects={result.redirects} ms={elapsed}"
            + (" truncated=true" if result.truncated else "")
        )
        if not 200 <= result.status < 300:
            return ToolOutput(
                f"抓取失败：HTTP {result.status}（{result.url}）。",
                detail,
                fail=True,
            )
        if result.content_type not in TEXT_TYPES:
            return ToolOutput(
                f"不支持的内容类型：{result.content_type or '未知'}。"
                "web_fetch 只处理 HTML、纯文本与 JSON。",
                detail,
                fail=True,
            )
        if result.content_type in {"text/html", "application/xhtml+xml"}:
            title, body = extract_text(result.text)
        else:
            title, body = "", result.text.strip()
        if not body:
            body = "（页面没有可读文本）"
        header = [f"来源: {result.url}"]
        if title:
            header.insert(0, f"标题: {title}")
        header.append(f"HTTP {result.status} · {result.content_type} · {result.size} 字节")
        return ToolOutput(truncate_output("\n".join(header) + "\n\n" + body, limit), detail)


def web_tools(api_key: str = "") -> list[Any]:
    fetcher = WebFetcher()
    return [WebSearchTool(fetcher, api_key=api_key), WebFetchTool(fetcher)]
