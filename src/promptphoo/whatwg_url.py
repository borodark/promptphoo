"""A small WHATWG URL implementation (the subset of Node's `URL` that the sanitizer relies on).

Python's `urllib.parse` neither validates nor normalizes the way `new URL()` does (default ports,
dot segments, percent-encode sets, host lowercasing, `searchParams` reserialization), and the
sanitizer's "did anything change" comparisons depend on Node's exact `href`. This follows the
URL Standard's basic URL parser for absolute URLs plus path-absolute input against a base.

Known gaps: IDNA uses Python's IDNA 2003 codec rather than UTS #46, and IPv6 serialization uses
`ipaddress` (which matches the spec's compression rule but not every embedded-IPv4 spelling).
"""

from __future__ import annotations

import ipaddress
import re

SPECIAL_SCHEMES: dict[str, int | None] = {
    "ftp": 21,
    "file": None,
    "http": 80,
    "https": 443,
    "ws": 80,
    "wss": 443,
}


class InvalidUrlError(ValueError):
    """TypeError: Invalid URL (Node's message)."""

    def __init__(self, url: str) -> None:
        super().__init__("Invalid URL")
        self.input = url


def _c0(c: str) -> bool:
    return c <= "\x1f" or c > "~"


_FRAGMENT_SET = frozenset(' "<>`')
_QUERY_SET = frozenset(' "#<>')
_SPECIAL_QUERY_SET = frozenset(" \"#<>'")
_PATH_SET = frozenset(' "#<>?`{}')
_USERINFO_SET = frozenset(' "#<>?`{}/:;=@[\\]^|')
# application/x-www-form-urlencoded: everything except ASCII alphanumerics and *-._
_FORM_SAFE = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789*-._")

_FORBIDDEN_HOST = frozenset("\x00\t\n\r #/:<>?@[\\]^|")
_FORBIDDEN_DOMAIN = _FORBIDDEN_HOST | frozenset("%\x7f") | frozenset(chr(i) for i in range(0x20))


def _pct(c: str) -> str:
    return "".join(f"%{b:02X}" for b in c.encode("utf-8", "surrogatepass"))


def percent_encode(value: str, extra: frozenset[str]) -> str:
    return "".join(_pct(c) if _c0(c) or c in extra else c for c in value)


def percent_decode_bytes(value: str) -> bytes:
    data = value.encode("utf-8", "surrogatepass")
    out = bytearray()
    i = 0
    hexdigits = b"0123456789abcdefABCDEF"
    while i < len(data):
        b = data[i]
        if (
            b == 0x25
            and i + 2 < len(data)
            and data[i + 1] in hexdigits
            and data[i + 2] in hexdigits
        ):
            out.append(int(data[i + 1 : i + 3], 16))
            i += 3
        else:
            out.append(b)
            i += 1
    return bytes(out)


def form_urlencoded_parse(query: str) -> list[list[str]]:
    pairs: list[list[str]] = []
    for seq in query.split("&"):
        if not seq:
            continue
        name, _, value = seq.partition("=")
        name = name.replace("+", " ")
        value = value.replace("+", " ")
        pairs.append(
            [
                percent_decode_bytes(name).decode("utf-8", "replace"),
                percent_decode_bytes(value).decode("utf-8", "replace"),
            ]
        )
    return pairs


def _form_encode(value: str) -> str:
    out = []
    for c in value:
        if c == " ":
            out.append("+")
        elif c in _FORM_SAFE:
            out.append(c)
        else:
            out.append(_pct(c))
    return "".join(out)


def form_urlencoded_serialize(pairs: list[list[str]]) -> str:
    return "&".join(f"{_form_encode(k)}={_form_encode(v)}" for k, v in pairs)


def _parse_ipv4_number(part: str) -> int | None:
    if not part:
        return None
    radix = 10
    if len(part) >= 2 and part[:2] in ("0x", "0X"):
        part = part[2:]
        radix = 16
    elif len(part) >= 2 and part[0] == "0":
        part = part[1:]
        radix = 8
    if part == "":
        return 0
    try:
        allowed = {10: "0123456789", 8: "01234567", 16: "0123456789abcdefABCDEF"}[radix]
        if any(c not in allowed for c in part):
            return None
        return int(part, radix)
    except ValueError:
        return None


def _ends_in_number(host: str) -> bool:
    parts = host.split(".")
    if parts[-1] == "":
        if len(parts) == 1:
            return False
        parts.pop()
    last = parts[-1]
    if last and all(c in "0123456789" for c in last):
        return True
    return _parse_ipv4_number(last) is not None and last[:2] in ("0x", "0X")


def _parse_ipv4(host: str, url: str) -> str:
    parts = host.split(".")
    if parts[-1] == "" and len(parts) > 1:
        parts.pop()
    if len(parts) > 4:
        raise InvalidUrlError(url)
    numbers: list[int] = []
    for part in parts:
        n = _parse_ipv4_number(part)
        if n is None:
            raise InvalidUrlError(url)
        numbers.append(n)
    if any(n > 255 for n in numbers[:-1]):
        raise InvalidUrlError(url)
    if numbers[-1] >= 256 ** (5 - len(numbers)):
        raise InvalidUrlError(url)
    ipv4 = numbers[-1]
    for i, n in enumerate(numbers[:-1]):
        ipv4 += n * 256 ** (3 - i)
    return ".".join(str((ipv4 >> s) & 0xFF) for s in (24, 16, 8, 0))


def _parse_host(value: str, special: bool, url: str) -> str:
    if value.startswith("["):
        if not value.endswith("]"):
            raise InvalidUrlError(url)
        try:
            addr = ipaddress.IPv6Address(value[1:-1])
        except ValueError:
            raise InvalidUrlError(url) from None
        return f"[{addr.compressed}]"
    if not special:
        if any(c in _FORBIDDEN_HOST for c in value):
            raise InvalidUrlError(url)
        return percent_encode(value, frozenset())
    try:
        domain = percent_decode_bytes(value).decode("utf-8")
    except UnicodeDecodeError:
        raise InvalidUrlError(url) from None
    if domain.isascii():
        ascii_domain = domain.lower()
    else:
        try:
            ascii_domain = domain.encode("idna").decode("ascii").lower()
        except UnicodeError:
            raise InvalidUrlError(url) from None
    if not ascii_domain or any(c in _FORBIDDEN_DOMAIN for c in ascii_domain):
        raise InvalidUrlError(url)
    if _ends_in_number(ascii_domain):
        return _parse_ipv4(ascii_domain, url)
    return ascii_domain


def _is_single_dot(seg: str) -> bool:
    return seg == "." or seg.lower() == "%2e"


def _is_double_dot(seg: str) -> bool:
    return seg.lower() in ("..", ".%2e", "%2e.", "%2e%2e")


def _is_windows_drive_letter(seg: str) -> bool:
    return len(seg) == 2 and seg[0].isascii() and seg[0].isalpha() and seg[1] in ":|"


class URL:
    """Subset of WHATWG `URL`: href/toString, the component getters and the setters we need."""

    def __init__(self, url: str, base: str | None = None) -> None:
        self.scheme = ""
        self.username = ""
        self.password = ""
        self.host: str | None = None
        self.port: int | None = None
        self.path: list[str] | str = []
        self.query: str | None = None
        self.fragment: str | None = None
        self._parse(url, URL(base) if base is not None else None)

    # ---- parsing -------------------------------------------------------------------------------

    @property
    def special(self) -> bool:
        return self.scheme in SPECIAL_SCHEMES

    def _parse(self, original: str, base: URL | None) -> None:
        s = original
        # Strip leading/trailing C0 control or space; remove ASCII tab or newline.
        s = s.lstrip("".join(chr(i) for i in range(0x21)))
        s = s.rstrip("".join(chr(i) for i in range(0x21)))
        s = s.replace("\t", "").replace("\n", "").replace("\r", "")

        m = re.match(r"^([A-Za-z][A-Za-z0-9+.\-]*):", s)
        if m:
            self.scheme = m.group(1).lower()
            rest = s[m.end() :]
            if self.scheme == "file":
                self._parse_file(rest, original)
            elif self.special:
                rest = rest.lstrip("/\\")
                self._parse_authority_and_rest(rest, original)
            elif rest.startswith("//"):
                self._parse_authority_and_rest(rest[2:], original)
            elif rest.startswith("/"):
                self._parse_path_query_fragment(rest, original)
            else:
                self._parse_opaque(rest)
            return

        if base is None or not (s.startswith("/") or (base.special and s.startswith("\\"))):
            raise InvalidUrlError(original)
        # Relative to base (only path-absolute input is supported).
        self.scheme = base.scheme
        if self.special and len(s) > 1 and s[1] in "/\\":
            self._parse_authority_and_rest(s.lstrip("/\\"), original)
            return
        self.username, self.password = base.username, base.password
        self.host, self.port = base.host, base.port
        self._parse_path_query_fragment(s, original)

    def _parse_opaque(self, rest: str) -> None:
        path, frag_sep, fragment = rest.partition("#")
        path, query_sep, query = path.partition("?")
        self.path = percent_encode(path, frozenset())
        if query_sep:
            self.query = percent_encode(query, _QUERY_SET)
        if frag_sep:
            self.fragment = percent_encode(fragment, _FRAGMENT_SET)

    def _parse_file(self, rest: str, original: str) -> None:
        self.host = ""
        if len(rest) >= 2 and rest[0] in "/\\" and rest[1] in "/\\":
            rest = rest[2:]
            end = len(rest)
            for i, c in enumerate(rest):
                if c in "/\\?#":
                    end = i
                    break
            host = rest[:end]
            rest = rest[end:]
            if host and not _is_windows_drive_letter(host):
                h = _parse_host(host, True, original)
                self.host = "" if h == "localhost" else h
            elif host:
                rest = "/" + host + rest
        self._parse_path_query_fragment(rest, original)

    def _parse_authority_and_rest(self, rest: str, original: str) -> None:
        end = len(rest)
        for i, c in enumerate(rest):
            if c in "/?#" or (self.special and c == "\\"):
                end = i
                break
        authority = rest[:end]
        rest = rest[end:]
        at = authority.rfind("@")
        if at != -1:
            userinfo = authority[:at]
            authority = authority[at + 1 :]
            user, colon, pw = userinfo.partition(":")
            self.username = percent_encode(user, _USERINFO_SET)
            if colon:
                self.password = percent_encode(pw, _USERINFO_SET)
        host = authority
        port_str: str | None = None
        bracket_end = authority.rfind("]")
        colon_idx = authority.rfind(":")
        if colon_idx != -1 and colon_idx > bracket_end:
            host = authority[:colon_idx]
            port_str = authority[colon_idx + 1 :]
        if at != -1 and host == "":
            raise InvalidUrlError(original)
        if host == "":
            if self.special:
                raise InvalidUrlError(original)
            self.host = ""
        else:
            self.host = _parse_host(host, self.special, original)
        if port_str:
            if not port_str.isascii() or not port_str.isdigit():
                raise InvalidUrlError(original)
            port = int(port_str)
            if port > 65535:
                raise InvalidUrlError(original)
            self.port = None if SPECIAL_SCHEMES.get(self.scheme, -1) == port else port
        self._parse_path_query_fragment(rest, original)

    def _parse_path_query_fragment(self, rest: str, original: str) -> None:
        path, frag_sep, fragment = rest.partition("#")
        path, query_sep, query = path.partition("?")
        self.path = self._parse_path(path)
        if query_sep:
            self.query = percent_encode(query, _SPECIAL_QUERY_SET if self.special else _QUERY_SET)
        if frag_sep:
            self.fragment = percent_encode(fragment, _FRAGMENT_SET)

    def _parse_path(self, path: str) -> list[str]:
        if self.special:
            path = path.replace("\\", "/")
        if path.startswith("/"):
            path = path[1:]
        elif not path:
            return [""] if self.special else []
        segments: list[str] = []
        raw = path.split("/")
        for i, seg in enumerate(raw):
            last = i == len(raw) - 1
            if _is_double_dot(seg):
                if segments and not (
                    self.scheme == "file"
                    and len(segments) == 1
                    and _is_windows_drive_letter(segments[0])
                ):
                    segments.pop()
                if last:
                    segments.append("")
            elif _is_single_dot(seg):
                if last:
                    segments.append("")
            else:
                segments.append(percent_encode(seg, _PATH_SET))
        return segments

    # ---- getters -------------------------------------------------------------------------------

    @property
    def href(self) -> str:
        out = self.scheme + ":"
        if self.host is not None:
            out += "//"
            if self.username or self.password:
                out += self.username
                if self.password:
                    out += ":" + self.password
                out += "@"
            out += self.host
            if self.port is not None:
                out += f":{self.port}"
        out += self.pathname
        if self.query is not None:
            out += "?" + self.query
        if self.fragment is not None:
            out += "#" + self.fragment
        return out

    def __str__(self) -> str:
        return self.href

    def to_string(self) -> str:
        return self.href

    @property
    def pathname(self) -> str:
        if isinstance(self.path, str):
            return self.path
        prefix = "/." if self.host is None and len(self.path) > 1 and self.path[0] == "" else ""
        return prefix + "".join("/" + seg for seg in self.path)

    @pathname.setter
    def pathname(self, value: str) -> None:
        if isinstance(self.path, str):
            return
        value = value.replace("\t", "").replace("\n", "").replace("\r", "")
        if self.special or value.startswith("/") or not value:
            self.path = self._parse_path(value)
        else:
            self.path = self._parse_path("/" + value)

    @property
    def search(self) -> str:
        return "" if not self.query else "?" + self.query

    @search.setter
    def search(self, value: str) -> None:
        if value == "":
            self.query = None
            return
        if value.startswith("?"):
            value = value[1:]
        self.query = percent_encode(value, _SPECIAL_QUERY_SET if self.special else _QUERY_SET)

    @property
    def hash(self) -> str:
        return "" if not self.fragment else "#" + self.fragment

    @hash.setter
    def hash(self, value: str) -> None:
        if value == "":
            self.fragment = None
            return
        if value.startswith("#"):
            value = value[1:]
        value = value.replace("\t", "").replace("\n", "").replace("\r", "")
        self.fragment = percent_encode(value, _FRAGMENT_SET)

    def _cannot_have_credentials(self) -> bool:
        return self.host is None or self.host == "" or self.scheme == "file"

    def set_username(self, value: str) -> None:
        if not self._cannot_have_credentials():
            self.username = percent_encode(value, _USERINFO_SET)

    def set_password(self, value: str) -> None:
        if not self._cannot_have_credentials():
            self.password = percent_encode(value, _USERINFO_SET)

    # ---- searchParams --------------------------------------------------------------------------

    def search_params_entries(self) -> list[tuple[str, str]]:
        return [(k, v) for k, v in form_urlencoded_parse(self.query or "")]

    def search_params_set(self, name: str, value: str) -> None:
        pairs = form_urlencoded_parse(self.query or "")
        found = False
        out: list[list[str]] = []
        for pair in pairs:
            if pair[0] == name:
                if not found:
                    out.append([name, value])
                    found = True
            else:
                out.append(pair)
        if not found:
            out.append([name, value])
        serialized = form_urlencoded_serialize(out)
        self.query = serialized if serialized else None
