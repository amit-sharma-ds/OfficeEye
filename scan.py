"""Find IP cameras on the local network so a changed DHCP address never has
to be hunted for by hand.

The work is strictly read-only toward the network: a TCP connect to port 554
to see who is listening, then an RTSP OPTIONS / DESCRIBE handshake (the same
trick probe_rtsp.py uses) to learn which stream path and login a camera
accepts - no writes to any camera, and no credentials beyond the handful of
common defaults plus whatever is already saved in config.json.

A scan runs in a background thread and reports progress, so the HTTP request
that kicks it off returns straight away.
"""

import base64
import hashlib
import re
import socket
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
import rtsp

RTSP_PORT = 554

# Stream paths different camera makes answer at. Ordered roughly by how common
# they are on the small cameras this office uses.
PATHS = [
    "/stream1", "/stream2",
    "/Streaming/Channels/101", "/Streaming/Channels/102",      # Hikvision
    "/cam/realmonitor?channel=1&subtype=0",                    # Dahua
    "/h264Preview_01_main",                                    # Reolink
    "/live/ch0", "/live", "/live0.264", "/11", "/12",
    "/ch01.264", "/video1", "/0", "/1", "/",
]
# The only logins we ever try: blank, and the usual factory defaults.
CREDS = [("admin", ""), ("admin", "123456"), ("admin", "admin"),
         ("admin", "12345"), ("admin", "password")]


# ---------------------------------------------------------------- RTSP probe
# Copied from probe_rtsp.py (not imported - that lives in a scratchpad). Speaks
# just enough RTSP to find a working path + login without the ~8s cost of
# opening the stream with OpenCV.

def _talk(ip, request, timeout=2.5):
    try:
        with socket.create_connection((ip, RTSP_PORT), timeout) as sock:
            sock.settimeout(timeout)
            sock.sendall(request.encode())
            data = b""
            while b"\r\n\r\n" not in data and len(data) < 8192:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
            return data.decode("utf-8", "replace")
    except OSError as exc:
        return "ERR " + str(exc)


def _status_of(reply):
    m = re.match(r"RTSP/1\.0 (\d+)", reply or "")
    return int(m.group(1)) if m else None


def _digest_header(user, pwd, url, challenge):
    realm = re.search(r'realm="([^"]*)"', challenge)
    nonce = re.search(r'nonce="([^"]*)"', challenge)
    if not (realm and nonce):
        return None
    realm, nonce = realm.group(1), nonce.group(1)
    md5 = lambda s: hashlib.md5(s.encode()).hexdigest()
    ha1 = md5("{}:{}:{}".format(user, realm, pwd))
    ha2 = md5("DESCRIBE:{}".format(url))
    resp = md5("{}:{}:{}".format(ha1, nonce, ha2))
    return ('Digest username="{}", realm="{}", nonce="{}", uri="{}", '
            'response="{}"'.format(user, realm, nonce, url, resp))


def _describe(ip, path, user, pwd, cseq=2):
    url = "rtsp://{}:{}{}".format(ip, RTSP_PORT, path)
    base = ("DESCRIBE {} RTSP/1.0\r\nCSeq: {}\r\n"
            "User-Agent: probe\r\nAccept: application/sdp\r\n".format(url, cseq))
    reply = _talk(ip, base + "\r\n")
    code = _status_of(reply)
    if code != 401:
        return code, reply
    if "Digest" in reply:
        auth = _digest_header(user, pwd, url, reply)
    else:
        token = base64.b64encode("{}:{}".format(user, pwd).encode()).decode()
        auth = "Basic " + token
    if not auth:
        return code, reply
    reply = _talk(ip, base + "Authorization: {}\r\n\r\n".format(auth))
    return _status_of(reply), reply


def _resolution(sdp):
    """Pull a WxH out of the SDP if the camera happens to announce one."""
    m = re.search(r"a=x-dimensions:\s*(\d+)\s*,\s*(\d+)", sdp or "")
    if m:
        return "{}x{}".format(m.group(1), m.group(2))
    m = re.search(r"a=framesize:\S*\s+(\d+)-(\d+)", sdp or "")
    if m:
        return "{}x{}".format(m.group(1), m.group(2))
    return ""


# ---------------------------------------------------------------- the network

def local_ip():
    """This machine's own IPv4. Connecting a UDP socket to a public address
    picks the right outbound interface without sending a single packet."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return ""
    finally:
        sock.close()


def local_subnet():
    """(base, hosts) for the local /24, e.g. ("192.168.1", [".1"..".254"])."""
    ip = local_ip()
    if not ip or ip.count(".") != 3:
        return "", []
    base = ip.rsplit(".", 1)[0]
    hosts = ["{}.{}".format(base, n) for n in range(1, 255)]
    return base, hosts


def port_open(ip, port=RTSP_PORT, timeout=1.0):
    sock = socket.socket()
    sock.settimeout(timeout)
    try:
        return sock.connect_ex((ip, port)) == 0
    except OSError:
        return False
    finally:
        sock.close()


def _combos():
    """Logins + paths to try, best guesses first: whatever the saved cameras
    already use, then the common factory defaults."""
    seen, combos = set(), []

    def add(user, pwd, path):
        key = (user, pwd, path)
        if key not in seen:
            seen.add(key)
            combos.append(key)

    for cam in config.cameras():
        p = rtsp.parse(cam.get("url", ""))
        if p and p.get("host"):
            add(p.get("user", ""), p.get("password", ""),
                p.get("path") or "/stream1")
    for user, pwd in CREDS:
        for path in PATHS:
            add(user, pwd, path)
    return combos


def probe_host(ip, combos):
    """Does a real camera live at ip? Returns its details or None."""
    opts = _talk(ip, "OPTIONS rtsp://{}:{}/ RTSP/1.0\r\nCSeq: 1\r\n\r\n".format(
        ip, RTSP_PORT))
    server = ""
    m = re.search(r"Server: *(.+)", opts or "")
    if m:
        server = m.group(1).strip()

    misses = 0
    for user, pwd, path in combos:
        code, reply = _describe(ip, path, user, pwd)
        if code == 200:
            return {
                "ip": ip,
                "url": rtsp.build(ip, user, pwd, RTSP_PORT, path, encode=True),
                "server": server,
                "resolution": _resolution(reply),
            }
        if reply.startswith("ERR"):
            # The host stopped answering; don't keep hammering a dead end.
            misses += 1
            if misses >= 3:
                break
        else:
            misses = 0
    return None


# ---------------------------------------------------------------- scanner

class Scanner:
    """Runs one network scan at a time in a background thread and keeps a
    little progress record the UI can poll."""

    def __init__(self):
        self._lock = threading.Lock()
        self._thread = None
        self._state = "idle"          # idle | scanning | done
        self._subnet = ""
        self._checked = 0
        self._total = 0
        self._error = ""
        self._found = []              # [{ip, url, server, resolution}]

    def is_running(self):
        with self._lock:
            return self._state == "scanning"

    def start(self):
        """Begin a scan. Returns False if one is already running."""
        with self._lock:
            if self._state == "scanning":
                return False
            self._state = "scanning"
            self._subnet = ""
            self._checked = 0
            self._total = 0
            self._error = ""
            self._found = []
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            return True

    def public(self):
        """A browser-safe snapshot: passwords in the urls are masked, and
        already_added is worked out fresh against the saved cameras."""
        saved = {rtsp.host_of(c.get("url", "")) for c in config.cameras()}
        saved.discard("")
        with self._lock:
            found = [{
                "ip": f["ip"],
                "url": rtsp.mask(f["url"]),
                "server": f["server"],
                "resolution": f["resolution"],
                "already_added": f["ip"] in saved,
            } for f in self._found]
            return {
                "state": self._state,
                "subnet": self._subnet,
                "checked": self._checked,
                "total": self._total,
                "error": self._error,
                "found": found,
            }

    def found_url(self, ip):
        """The real (unmasked) url for a scanned ip, for adding it."""
        with self._lock:
            for f in self._found:
                if f["ip"] == ip:
                    return f["url"]
        return ""

    # -------- the background job --------

    def _run(self):
        try:
            base, hosts = local_subnet()
            with self._lock:
                self._subnet = base
                self._total = len(hosts)
            if not hosts:
                with self._lock:
                    self._error = "Could not work out the local network"
                return

            combos = _combos()
            open_hosts = []
            with ThreadPoolExecutor(max_workers=64) as pool:
                futures = {pool.submit(port_open, ip): ip for ip in hosts}
                for fut in as_completed(futures):
                    ip = futures[fut]
                    try:
                        listening = fut.result()
                    except Exception:
                        listening = False
                    with self._lock:
                        self._checked += 1
                    if listening:
                        open_hosts.append(ip)

            # A handful of hosts at most, so probe them in a small pool.
            open_hosts.sort(key=_ip_key)
            with ThreadPoolExecutor(max_workers=min(8, len(open_hosts) or 1)) \
                    as pool:
                for result in pool.map(lambda ip: probe_host(ip, combos),
                                       open_hosts):
                    if result:
                        with self._lock:
                            self._found.append(result)
                            self._found.sort(key=lambda f: _ip_key(f["ip"]))
        except Exception as exc:                       # never leave it stuck
            with self._lock:
                self._error = "Scan failed: {}".format(exc)
        finally:
            with self._lock:
                self._state = "done"


def _ip_key(ip):
    try:
        return tuple(int(p) for p in ip.split("."))
    except ValueError:
        return (0,)


scanner = Scanner()
