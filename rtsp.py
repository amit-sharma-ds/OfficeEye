"""Helpers for reading, building and masking RTSP urls.

The camera password is part of the url, so the url is never sent to the
browser as-is: the password is replaced by MASK. When the browser sends the
url back untouched, unmask() puts the saved password in again.
"""

import re
from urllib.parse import quote

MASK = "***"
DEFAULT_PORT = 554
DEFAULT_PATH = "/stream1"
DEFAULT_USER = "admin"

_SCHEME = re.compile(r"^(?P<scheme>[a-zA-Z][a-zA-Z0-9+.\-]*)://(?P<rest>.*)$", re.S)
_IP_LIKE = re.compile(r"^[A-Za-z0-9._\-]+$")


def parse(url):
    """Split an rtsp url into its parts. Returns {} if it is not a url."""
    url = (url or "").strip()
    m = _SCHEME.match(url)
    if not m:
        return {}
    rest = m.group("rest")
    cut = rest.find("/")
    authority = rest if cut < 0 else rest[:cut]
    path = DEFAULT_PATH if cut < 0 else rest[cut:]
    user = password = ""
    if "@" in authority:
        # rsplit: a password may itself contain '@'
        cred, hostport = authority.rsplit("@", 1)
        if ":" in cred:
            user, password = cred.split(":", 1)
        else:
            user = cred
    else:
        hostport = authority
    host, port = hostport, DEFAULT_PORT
    if hostport.startswith("["):  # ipv6
        end = hostport.find("]")
        host = hostport[: end + 1]
        tail = hostport[end + 1 :]
        if tail.startswith(":") and tail[1:].isdigit():
            port = int(tail[1:])
    elif ":" in hostport:
        host, tail = hostport.rsplit(":", 1)
        if tail.isdigit():
            port = int(tail)
        else:
            host = hostport
    return {
        "scheme": m.group("scheme").lower(),
        "user": user,
        "password": password,
        "host": host,
        "port": port,
        "path": path or DEFAULT_PATH,
    }


def build(host, user=DEFAULT_USER, password="", port=DEFAULT_PORT,
          path=DEFAULT_PATH, scheme="rtsp", encode=True):
    """Build an rtsp url. encode=True percent-escapes user/password."""
    host = (host or "").strip()
    if not host:
        return ""
    cred = ""
    if user:
        u = quote(user, safe="") if encode else user
        cred = u
        if password:
            p = quote(password, safe="") if encode else password
            cred += ":" + p
        cred += "@"
    portpart = f":{port}" if port else ""
    if not path.startswith("/"):
        path = "/" + path
    return f"{scheme}://{cred}{host}{portpart}{path}"


def mask(url):
    """Replace the password inside a url with ***, for showing in the UI."""
    p = parse(url)
    if not p:
        return url or ""
    if not p["password"]:
        return url
    return build(p["host"], p["user"], MASK, p["port"], p["path"], p["scheme"],
                 encode=False)


def unmask(submitted, saved_url):
    """Put the saved password back into a url the browser sent with ***."""
    submitted = (submitted or "").strip()
    if MASK not in submitted:
        return submitted
    saved = parse(saved_url)
    secret = saved.get("password", "")
    sub = parse(submitted)
    if not sub:
        return submitted
    return build(sub["host"], sub["user"], secret, sub["port"], sub["path"],
                 sub["scheme"], encode=False)


def host_of(url):
    return parse(url).get("host", "")


def with_host(url, host):
    """Same url pointing at a different ip / hostname."""
    p = parse(url)
    if not p:
        return build(host)
    return build(host, p["user"], p["password"], p["port"], p["path"],
                 p["scheme"], encode=False)


def looks_like_host(value):
    value = (value or "").strip()
    return bool(value) and bool(_IP_LIKE.match(value))
