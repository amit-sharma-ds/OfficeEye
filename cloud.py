"""Cloud view: the office app sends what it sees to a copy of itself on a
cloud host (Render), so the pages open from anywhere.

The cameras are on the office network, which a cloud server cannot reach,
so the office PC does the watching and *sends*:

- the live status (counts, alerts, camera states) every few seconds,
- camera pictures - only of cameras somebody is looking at right then,
- the event rows (alerts, phone use, counts) and their photos,
- the camera running-time rows.

The cloud copy stores that and serves the same pages, read only, behind a
password. It never sees a camera link or password.

    office   Settings -> Cloud view: the cloud link, and a push token
    cloud    environment: OFFICEEYE_PASSWORD   the password to log in
                          OFFICEEYE_PUSH_TOKEN the same push token
             (OFFICEEYE_MODE=cloud; a Render service is cloud by itself)
"""

import gzip
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import threading
import time
from copy import deepcopy
from datetime import datetime, timedelta
from urllib.parse import quote, urlencode

from flask import Response, jsonify, redirect, render_template, request, session
from werkzeug.security import safe_join

import config
import events
import photos
import rtsp

log = logging.getLogger("cloud")

MODE = (os.environ.get("OFFICEEYE_MODE")
        or ("cloud" if os.environ.get("RENDER") else "office")).lower()
CLOUD = MODE == "cloud"
PASSWORD = os.environ.get("OFFICEEYE_PASSWORD", "")
PUSH_TOKEN = os.environ.get("OFFICEEYE_PUSH_TOKEN", "")
READ_ONLY = "This is the cloud view - change it on the office PC."
OFFICE_GONE = 90          # seconds without a push before the office counts as away


# ================================================================ cloud side

def _empty_status():
    return {
        "cameras": {}, "total": 0, "reporting": 0, "on_phone_total": 0,
        "peak_today": None, "alerting": [], "latest_alert": 0,
        "alerts_today": 0, "phone_today": 0,
        "server_time": datetime.now().isoformat(timespec="seconds"),
        "detection": {"enabled": False, "state": "off", "error": "", "device": "",
                      "batch": 0, "available": [],
                      "perf": {"per_second": 0, "ms_per_frame": 0, "load": 0,
                               "lag_count": None, "lag_restricted": None}},
    }


class Store:
    """What the office sent last, kept in memory. Event rows, running-time
    rows and photos go to the normal database and photo folder instead."""

    def __init__(self):
        self.lock = threading.Lock()
        self.status = None
        self.cameras = []          # the Cameras-page listing, without links
        self.settings = {}         # office, detection, photos sections
        self.frames = {}           # camera -> (jpeg, key, received at)
        self.asked = {}            # camera -> when a page last asked for its picture
        self.watching = {}         # camera -> open live views
        self.pushed_at = 0.0
        self.viewer_at = 0.0       # when a logged-in page last polled the status
        self.login = None          # (salt, sha256) of the cloud password, sent by the office

    def want(self):
        """Pictures the office should send, and how often (s)."""
        now = time.time()
        with self.lock:
            out = {c: 5.0 for c, t in self.asked.items() if now - t < 20}
            out.update({c: 1.0 for c, n in self.watching.items() if n > 0})
        return out

    def status_payload(self):
        with self.lock:
            st = deepcopy(self.status) if self.status else _empty_status()
            ago = time.time() - self.pushed_at if self.pushed_at else None
        st["cloud"] = {"office_seen_ago": round(ago, 1) if ago is not None else None,
                       "office_seen_at": datetime.fromtimestamp(self.pushed_at)
                       .isoformat(timespec="seconds") if self.pushed_at else None}
        if ago is None or ago > OFFICE_GONE:
            # Nothing is live: say so instead of showing old numbers as now.
            st["detection"]["state"] = "office offline"
            for cam in st["cameras"].values():
                cam.update(state="error", error="the office PC is not sending",
                           count=None, on_phone=None, fresh=False)
                if isinstance(cam.get("alert"), dict):
                    cam["alert"]["active"] = False
            st.update(total=0, reporting=0, on_phone_total=0, alerting=[])
        return st

    def listing(self, group=None):
        with self.lock:
            cams = deepcopy(self.cameras)
        return [c for c in cams if group is None or c.get("group") == group]


store = Store()


class ManagerView:
    """The camera manager as the pages see it, answered from the last push."""

    def listing(self, group=None):
        return store.listing(group)

    def status(self):
        return store.status_payload().get("cameras") or {}

    def get(self, cam_id):
        return None


class PipelineView:
    def live(self):
        return {}

    def status(self):
        return store.status_payload()["detection"]


def section(name):
    """A settings section: the office's copy on the cloud side."""
    if CLOUD:
        with store.lock:
            pushed = deepcopy(store.settings.get(name))
        return pushed or config.get(name)
    return config.get(name)


def frame_response(cam_id, blank):
    with store.lock:
        store.asked[cam_id] = store.viewer_at = time.time()
        hit = store.frames.get(cam_id)
    if hit and time.time() - hit[2] < 300:
        return Response(hit[0], mimetype="image/jpeg", headers={"Cache-Control": "no-store"})
    cam = (store.status_payload().get("cameras") or {}).get(cam_id) or {}
    text = ("no IP set" if cam.get("state") == "off" else
            "office PC offline" if cam.get("error") == "the office PC is not sending" else
            "waiting for the office PC...")
    return Response(blank(text), mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


def live_response(cam_id, blank):
    def gen():
        with store.lock:
            store.watching[cam_id] = store.watching.get(cam_id, 0) + 1
        try:
            last, sent_at, started = None, 0.0, time.time()
            while time.time() - started < 1800:           # half an hour, then reopen
                with store.lock:
                    hit = store.frames.get(cam_id)
                jpg, key = (hit[0], hit[1]) if hit else (blank("waiting for the office PC..."), "blank")
                now = time.time()
                if key != last or now - sent_at > 2.0:
                    last, sent_at = key, now
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                time.sleep(0.1)
        finally:
            with store.lock:
                store.watching[cam_id] = max(0, store.watching.get(cam_id, 1) - 1)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-store"})


NO_PASSWORD = """<!doctype html><meta charset="utf-8"><title>OfficeEye</title>
<meta name="robots" content="noindex, nofollow">
<body style="font:15px system-ui;padding:40px;max-width:640px">
<h2>OfficeEye cloud view - waiting for the office PC</h2>
<p>The office PC has not connected yet, so there is no password to log in with.
On the office PC open <b>Settings -> Cloud view</b> and set <b>Send to the cloud</b>
to <i>On</i>; this page then asks for the password shown there. (It can take a
minute after the cloud view wakes up.)</p>"""


# The office's push token is checked against its SHA-256, kept in the
# repository (cloud_push.sha256), so a fresh cloud host needs no settings.
# The hash does not give the token away. OFFICEEYE_PUSH_TOKEN, when set on
# the host, is used instead.
_HASH_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloud_push.sha256")


def _token_hash():
    try:
        with open(_HASH_FILE, encoding="utf-8") as f:
            return f.read().strip().lower()
    except OSError:
        return ""


def _pw_hash(salt, password):
    return hashlib.sha256((salt + ":" + password).encode()).hexdigest()


def _login_key():
    """What a session is tied to: the host's password, or the one the office
    sent. A new password logs everybody out."""
    if PASSWORD:
        return hashlib.sha256(("env:" + PASSWORD).encode()).hexdigest()[:16]
    with store.lock:
        return store.login[1][:16] if store.login else None

_failures = {}             # ip -> [times of wrong passwords]


def _json_body():
    data = request.get_data()
    if request.headers.get("Content-Encoding") == "gzip":
        data = gzip.decompress(data)
    return json.loads(data or b"{}")


def _token_ok():
    sent = request.headers.get("Authorization", "")
    if PUSH_TOKEN:
        return hmac.compare_digest(sent.encode(), ("Bearer " + PUSH_TOKEN).encode())
    want = _token_hash()
    if not want or not sent.startswith("Bearer "):
        return False
    return hmac.compare_digest(hashlib.sha256(sent[7:].encode()).hexdigest(), want)


def _photo_names(meta):
    try:
        meta = json.loads(meta) if isinstance(meta, str) else (meta or {})
    except ValueError:
        return []
    out = []
    for shot in meta.get("photos") or []:
        if isinstance(shot, dict):
            out += [shot.get("full")] + list(shot.get("crops") or [])
    return [n for n in out if isinstance(n, str) and n]


def _clean_row(r):
    return {"id": int(r["id"]), "ts": str(r["ts"]), "day": str(r["day"]),
            "camera": str(r["camera"]), "kind": str(r["kind"]),
            "label": str(r.get("label") or ""), "value": float(r.get("value") or 0),
            "meta": r["meta"] if isinstance(r.get("meta"), str) else json.dumps(r.get("meta") or {})}


def _install_cloud(app):
    # Session cookies are signed with a secret nobody outside can know: the
    # host's settings, or (with none) a random one until the office sends its
    # password - then one made from that, so logins survive a restart.
    app.secret_key = (os.environ.get("OFFICEEYE_SECRET") or
                      (hashlib.sha256(("officeeye:" + PASSWORD + ":" + PUSH_TOKEN).encode()).hexdigest()
                       if PASSWORD else secrets.token_hex(32)))
    app.config.update(SESSION_COOKIE_SECURE=bool(os.environ.get("RENDER")),
                      SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                      PERMANENT_SESSION_LIFETIME=timedelta(days=30),
                      MAX_CONTENT_LENGTH=32 * 1024 * 1024)

    @app.before_request
    def _gate():
        p = request.path
        if p.startswith("/api/cloud/push/") or p.startswith("/static/") or p in ("/login", "/healthz"):
            return None
        key = _login_key()
        logged_in = bool(key) and session.get("in") == key
        if p == "/api/status" and not logged_in:
            return jsonify({"login_required": True})          # health checks get a 200
        if not key:
            return Response(NO_PASSWORD, 503, mimetype="text/html")
        if not logged_in:
            if p.startswith(("/api/", "/frame/", "/snapshot/", "/live/", "/photos/")):
                return jsonify({"ok": False, "message": "Log in first"}), 401
            return redirect("/login?next=" + quote(request.full_path.rstrip("?")))
        if request.method not in ("GET", "HEAD"):
            return jsonify({"ok": False, "message": READ_ONLY}), 403
        if p == "/api/status":
            store.viewer_at = time.time()
        return None

    @app.get("/healthz")
    def healthz():
        return "ok"

    @app.route("/login", methods=["GET", "POST"])
    def login():
        nxt = request.values.get("next") or "/"
        if not nxt.startswith("/") or nxt.startswith("//"):
            nxt = "/"
        key = _login_key()
        if not key:
            return Response(NO_PASSWORD, 503, mimetype="text/html")
        error = ""
        if request.method == "POST":
            ip = request.headers.get("X-Forwarded-For", request.remote_addr or "").split(",")[0].strip()
            now = time.time()
            recent = [t for t in _failures.get(ip, []) if now - t < 600]
            typed = request.form.get("password", "")
            user = request.form.get("username", "").strip().lower()
            if PASSWORD:
                right = hmac.compare_digest(typed.encode(), PASSWORD.encode()) and \
                    user == os.environ.get("OFFICEEYE_USER", "admin").lower()
            else:
                with store.lock:
                    salt, want, name = store.login
                right = hmac.compare_digest(_pw_hash(salt, typed), want) and \
                    hmac.compare_digest(user.encode(), name.lower().encode())
            if len(recent) >= 10:
                error = "Too many wrong passwords. Try again in a few minutes."
            elif right:
                _failures.pop(ip, None)
                session.clear()
                session["in"] = key
                session.permanent = True
                return redirect(nxt)
            else:
                time.sleep(1)
                _failures[ip] = recent + [now]
                error = "Wrong password."
        return render_template("login.html", error=error, next=nxt)

    @app.get("/logout")
    def logout():
        session.clear()
        return redirect("/login")

    @app.post("/api/cloud/push/state")
    def push_state():
        if not _token_ok():
            return jsonify({"ok": False, "message": "wrong push token"}), 401
        body = _json_body()
        login = body.get("login") or {}
        with store.lock:
            store.status = body.get("status") or _empty_status()
            store.cameras = body.get("cameras") or []
            store.settings = body.get("settings") or {}
            store.pushed_at = time.time()
            new = (login.get("salt"), login.get("hash"), str(login.get("user") or "admin"))
            if isinstance(new[0], str) and isinstance(new[1], str) and len(new[1]) == 64 \
                    and new != store.login:
                store.login = new
                if not PASSWORD and not os.environ.get("OFFICEEYE_SECRET"):
                    app.secret_key = hashlib.sha256(
                        ("session:" + login["salt"] + login["hash"]).encode()).hexdigest()
        return jsonify({"ok": True, "want": store.want(),
                        "viewers": time.time() - store.viewer_at < 60,
                        "events_last_id": events.max_id(),
                        "have_days": bool(events.camera_days("2000-01-01", "2999-12-31")[:1])})

    @app.post("/api/cloud/push/frame/<cam_id>")
    def push_frame(cam_id):
        if not _token_ok():
            return jsonify({"ok": False, "message": "wrong push token"}), 401
        jpg = request.get_data()
        if not jpg.startswith(b"\xff\xd8") or len(jpg) > 4 * 1024 * 1024:
            return jsonify({"ok": False, "message": "not a jpeg"}), 400
        with store.lock:
            store.frames[cam_id] = (jpg, request.args.get("key", ""), time.time())
        return jsonify({"ok": True})

    @app.post("/api/cloud/push/events")
    def push_events():
        if not _token_ok():
            return jsonify({"ok": False, "message": "wrong push token"}), 401
        rows = []
        for r in _json_body().get("rows") or []:
            try:
                rows.append(_clean_row(r))
            except (KeyError, TypeError, ValueError):
                continue
        events.store_rows(rows)
        missing = []
        for r in rows:
            for name in _photo_names(r["meta"]):
                path = safe_join(photos.PHOTO_DIR, name)
                if path and not os.path.exists(path) and name not in missing:
                    missing.append(name)
        return jsonify({"ok": True, "last_id": events.max_id(), "missing": missing[:200]})

    @app.post("/api/cloud/push/photo/<path:name>")
    def push_photo(name):
        if not _token_ok():
            return jsonify({"ok": False, "message": "wrong push token"}), 401
        path = safe_join(photos.PHOTO_DIR, name)
        data = request.get_data()
        if not path or not data.startswith(b"\xff\xd8"):
            return jsonify({"ok": False, "message": "bad photo"}), 400
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        return jsonify({"ok": True})

    @app.post("/api/cloud/push/uptime")
    def push_uptime():
        if not _token_ok():
            return jsonify({"ok": False, "message": "wrong push token"}), 401
        body = _json_body()
        days = [str(d) for d in body.get("days") or []]
        events.store_uptime(body.get("uptime") or [], days)
        if body.get("camera_days"):
            events.store_camera_days(body["camera_days"])
        return jsonify({"ok": True})


# ================================================================ office side

_RTSP = re.compile(r"rtsp://[^\s'\"]+")


def _mask_links(text):
    return _RTSP.sub(lambda m: rtsp.mask(m.group(0)), text or "")


class Uplink:
    """Sends the office's view to the cloud copy. Two threads: one for the
    status, records and photos (small - they must never wait), one for
    pictures, only of cameras somebody is looking at in the cloud view. Each
    keeps its connection open, so a far-away cloud costs one round trip per
    request instead of a new HTTPS handshake every time."""

    def __init__(self):
        self._status_fn = self._render_fn = self._listing_fn = None
        self.state = "off"
        self.error = ""
        self.ok_at = 0.0
        self.sent = []             # (time, bytes) of the last minute
        self.frames_sent = []
        self.viewers = False
        self.want = {}             # camera -> seconds between pictures
        self._lock = threading.Lock()

    def start(self, status_fn, render_fn, listing_fn):
        self._status_fn, self._render_fn, self._listing_fn = status_fn, render_fn, listing_fn
        threading.Thread(target=self._run, name="cloud-uplink", daemon=True).start()
        threading.Thread(target=self._run_frames, name="cloud-frames", daemon=True).start()

    def info(self):
        cfg = config.get("cloud")
        now = time.time()
        with self._lock:
            self.sent = [s for s in self.sent if now - s[0] < 60]
            self.frames_sent = [t for t in self.frames_sent if now - t < 60]
            kb, frames = round(sum(b for _, b in self.sent) / 1024), len(self.frames_sent)
        return {"enabled": bool(cfg.get("enabled")), "url": cfg.get("url", ""),
                "token": cfg.get("token", ""), "password": cfg.get("password", ""),
                "username": cfg.get("username") or "admin",
                "state": self.state, "error": self.error,
                "ok_ago": round(now - self.ok_at, 1) if self.ok_at else None,
                "kb_per_min": kb, "frames_per_min": frames, "viewers": self.viewers}

    # ---------- http ----------

    @staticmethod
    def _on(cfg):
        return bool(cfg.get("enabled") and cfg.get("url") and cfg.get("token"))

    def _post(self, http, cfg, path, data, ctype, gz=False):
        headers = {"Authorization": "Bearer " + cfg["token"], "Content-Type": ctype}
        if gz:
            data = gzip.compress(data, 5)
            headers["Content-Encoding"] = "gzip"
        r = http.post(cfg["url"].rstrip("/") + path, data=data, headers=headers, timeout=30)
        if r.status_code >= 400:
            raise RuntimeError("cloud answered {} {}".format(r.status_code, r.text[:200]))
        with self._lock:
            self.sent.append((time.time(), len(data)))
        return r.json() if r.content else {}

    def _json(self, http, cfg, path, obj):
        return self._post(http, cfg, path, json.dumps(obj, default=str).encode(),
                          "application/json", gz=True)

    # ---------- status, records, photos ----------

    def _state_body(self):
        st = self._status_fn()
        for cam in st.get("cameras", {}).values():
            cam["error"] = _mask_links(cam.get("error"))
        cams = []
        for c in self._listing_fn():
            c = dict(c)
            c["url"], c["prefilled"] = "", False
            cams.append(c)
        cloud = config.get("cloud")
        body = {"status": st, "cameras": cams,
                "settings": {k: config.get(k) for k in ("office", "detection", "photos")}}
        if cloud.get("salt") and cloud.get("hash"):
            # Only a salted hash of the login password leaves this PC.
            body["login"] = {"salt": cloud["salt"], "hash": cloud["hash"],
                             "user": cloud.get("username") or "admin"}
        return body

    def _run(self):
        import requests
        http = None
        pushed_id = None           # newest event id the cloud has
        have_days = True
        last_state = last_events = last_refresh = last_uptime = last_days = 0.0
        missing, last_alert, backoff = [], None, 0
        while True:
            cfg = config.get("cloud")
            if not self._on(cfg):
                self.state, pushed_id, self.want = "off", None, {}
                time.sleep(3)
                continue
            try:
                http = http or requests.Session()
                now = time.time()
                latest = events.latest_id("restricted_alert")
                # Every 2 s while somebody looks at the cloud view, else every
                # 10 s - and at once when a new restricted alert starts.
                if now - last_state >= (2 if self.viewers else 10) or latest != last_alert:
                    r = self._json(http, cfg, "/api/cloud/push/state", self._state_body())
                    last_state, last_alert = now, latest
                    self.want, self.viewers = r.get("want") or {}, bool(r.get("viewers"))
                    have_days = r.get("have_days", True)
                    cloud_last = int(r.get("events_last_id") or 0)
                    if pushed_id is None or cloud_last < pushed_id:
                        # First contact, or the cloud lost its disk (a restart):
                        # send the last three days again.
                        start = events.first_id_since(
                            (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"))
                        pushed_id = max(cloud_last, (start or 1) - 1)
                        last_days = 0.0

                if pushed_id is not None and now - last_events >= 3:
                    last_events = now
                    new = events.rows_after(pushed_id, 1000)
                    rows = list(new)
                    if now - last_refresh >= 30:
                        last_refresh = now
                        rows += events.rows_of_day(["restricted_alert", "phone_use", "washroom_in"],
                                                   datetime.now().strftime("%Y-%m-%d"))
                    if rows:
                        r = self._json(http, cfg, "/api/cloud/push/events", {"rows": rows})
                        if new:                     # only new rows move the mark on
                            pushed_id = new[-1]["id"]
                        missing = list(dict.fromkeys(missing + (r.get("missing") or [])))

                for name in missing[:6]:
                    path = os.path.join(photos.PHOTO_DIR, name)
                    if os.path.exists(path):
                        with open(path, "rb") as f:
                            self._post(http, cfg, "/api/cloud/push/photo/" + quote(name),
                                       f.read(), "image/jpeg")
                    missing.remove(name)

                if now - last_uptime >= 60:
                    last_uptime = now
                    today = datetime.now()
                    days = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in (1, 0)]
                    body = {"days": days, "uptime": events.uptime_rows(days[0])}
                    if not have_days or now - last_days >= 600:
                        last_days = now
                        body["camera_days"] = events.camera_days(
                            (today - timedelta(days=400)).strftime("%Y-%m-%d"), days[1])
                    self._json(http, cfg, "/api/cloud/push/uptime", body)

                self.state, self.error, self.ok_at, backoff = "ok", "", time.time(), 0
                time.sleep(0.3)
            except Exception as e:                              # noqa: BLE001
                http = None
                self.state = "error"
                self.error = _mask_links(str(e))[:300]
                backoff = min(60, backoff * 2 or 5)
                log.warning("cloud uplink: %s (retry in %s s)", self.error, backoff)
                time.sleep(backoff)

    # ---------- pictures ----------

    def _run_frames(self):
        import requests
        http, sent_key, sent_at = None, {}, {}
        while True:
            cfg, want = config.get("cloud"), dict(self.want)
            if not self._on(cfg) or not want or self.state != "ok":
                time.sleep(1)
                continue
            now = time.time()
            due = sorted((sent_at.get(c, 0) + float(e) - now, c, float(e)) for c, e in want.items())
            wait, cam, every = due[0]
            if wait > 0:
                time.sleep(min(wait, 0.5))
                continue
            # A tile on the camera wall is small; the live view gets more.
            jpg, key = self._render_fn(cam, 640 if every <= 1.5 else 480, True,
                                       quality=65, caption=False)
            sent_at[cam] = time.time()
            if jpg is None or key == sent_key.get(cam):
                continue                            # nothing new on that camera
            k = "-".join(str(x) for x in key) if isinstance(key, tuple) else str(key)
            try:
                http = http or requests.Session()
                self._post(http, cfg, "/api/cloud/push/frame/{}?{}".format(
                    quote(cam), urlencode({"key": k})), jpg, "image/jpeg")
            except Exception as e:                          # noqa: BLE001
                http = None
                log.warning("cloud pictures: %s", _mask_links(str(e))[:200])
                time.sleep(5)
                continue
            sent_key[cam] = key
            with self._lock:
                self.frames_sent.append(time.time())


uplink = Uplink()


def _install_office(app):
    @app.get("/api/cloud/uplink")
    def cloud_uplink():
        return jsonify(uplink.info())

    @app.post("/api/cloud/settings")
    def cloud_settings():
        body = request.get_json(silent=True) or {}
        cfg = config.get("cloud")
        patch = {}
        if "url" in body:
            url = str(body["url"]).strip().rstrip("/")
            if url and not re.match(r"^https?://[^\s/]+", url):
                return jsonify({"ok": False, "message": "The cloud link must start with https://"})
            patch["url"] = url
        if body.get("new_token") or not cfg.get("token"):
            patch["token"] = secrets.token_urlsafe(32)
        if "username" in body:
            name = str(body["username"] or "").strip()
            if not re.match(r"^[A-Za-z0-9._@-]{3,40}$", name):
                return jsonify({"ok": False, "message": "Username: 3 to 40 letters or digits"})
            patch["username"] = name
        password = str(body.get("password") or "").strip()
        if password and len(password) < 6:
            return jsonify({"ok": False, "message": "The cloud password needs at least 6 characters"})
        if password or not cfg.get("password"):
            # The password to open the cloud link. It stays on this PC; the
            # cloud copy only gets a salted hash of it.
            password = password or "OE-" + secrets.token_urlsafe(6)
            salt = secrets.token_hex(16)
            patch.update(password=password, salt=salt, hash=_pw_hash(salt, password))
        if "enabled" in body:
            patch["enabled"] = bool(body["enabled"])
        config.update("cloud", patch)
        return jsonify({"ok": True, "message": "Cloud view settings saved", "uplink": uplink.info()})


def install(app):
    """Hook the cloud view into the Flask app: login and push endpoints on
    the cloud side, the uplink settings on the office side."""
    @app.context_processor
    def _mode():
        return {"cloud_mode": CLOUD}

    if CLOUD:
        _install_cloud(app)
    else:
        _install_office(app)
