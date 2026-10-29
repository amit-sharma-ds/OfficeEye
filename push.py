"""Phone alerts (web push) for the installed app (PWA).

A phone that turns on "Phone alerts" (in the app, on the cloud link or on
this PC) hands its push address to this office PC, which keeps them in
data/push.json together with its own key pair (VAPID). A thread watches the
records: a new restricted-area alert, something at night (Night vision), a
new vehicle in the parking or a camera with a problem (Camera health) is
sent to every phone at once - the app does not have to be open.

The office PC sends them itself (it has the records and the key), so the
cloud copy needs no secret. Push works only on https or localhost: on the
cloud link, not on the office network's http address.
"""

import base64
import json
import logging
import os
import threading
import time
from datetime import datetime

import config
import events

log = logging.getLogger(__name__)

PUSH_FILE = os.path.join(config.DATA_DIR, "push.json")
KEY_FILE = os.path.join(config.DATA_DIR, "push_key.pem")


def _b64(raw):
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


class Push:
    CHECK_EVERY = 3.0          # seconds between looks at the records

    def __init__(self):
        self._lock = threading.Lock()
        self._vapid = None
        self._thread = None

    # ---------- keys and phones ----------

    def _key(self):
        """This PC's VAPID key pair, made on first use."""
        from py_vapid import Vapid02
        if self._vapid is None:
            if os.path.exists(KEY_FILE):
                self._vapid = Vapid02.from_file(KEY_FILE)
            else:
                os.makedirs(config.DATA_DIR, exist_ok=True)
                v = Vapid02()
                v.generate_keys()
                v.save_key(KEY_FILE)
                self._vapid = v
        return self._vapid

    def public_key(self):
        """What a browser needs to subscribe (the applicationServerKey)."""
        from cryptography.hazmat.primitives import serialization
        with self._lock:
            pub = self._key().public_key.public_bytes(serialization.Encoding.X962,
                                                      serialization.PublicFormat.UncompressedPoint)
        return _b64(pub)

    def _load(self):
        try:
            with open(PUSH_FILE, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            data = {}
        data.setdefault("phones", [])
        return data

    def _save(self, data):
        tmp = PUSH_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, PUSH_FILE)

    def phones(self):
        with self._lock:
            return len(self._load()["phones"])

    def subscribe(self, sub, device=""):
        """Keep a phone's push address. Returns it, cleaned, or None."""
        if not isinstance(sub, dict) or not str(sub.get("endpoint", "")).startswith("https://"):
            return None
        keys = sub.get("keys") or {}
        if not keys.get("p256dh") or not keys.get("auth"):
            return None
        clean = {"endpoint": str(sub["endpoint"]),
                 "keys": {"p256dh": str(keys["p256dh"]), "auth": str(keys["auth"])},
                 "device": str(device or "")[:80],
                 "since": datetime.now().isoformat(timespec="seconds")}
        with self._lock:
            data = self._load()
            data["phones"] = [p for p in data["phones"] if p["endpoint"] != clean["endpoint"]] + [clean]
            self._save(data)
        return clean

    def unsubscribe(self, endpoint):
        with self._lock:
            data = self._load()
            before = len(data["phones"])
            data["phones"] = [p for p in data["phones"] if p["endpoint"] != endpoint]
            self._save(data)
            return before - len(data["phones"])

    # ---------- sending ----------

    def send(self, message, only=None):
        """Send one alert to every phone (or just `only`). Returns how many
        got it. Phones that have turned alerts off are forgotten."""
        from pywebpush import WebPushException, webpush
        with self._lock:
            phones = [only] if only else list(self._load()["phones"])
            vapid = self._key()
        sub = (config.get("cloud").get("url") or "").strip() or "mailto:admin@officeeye.local"
        sent, gone = 0, []
        for p in phones:
            try:
                webpush({"endpoint": p["endpoint"], "keys": p["keys"]}, json.dumps(message),
                        vapid_private_key=vapid, vapid_claims={"sub": sub}, ttl=3600, timeout=15)
                sent += 1
            except WebPushException as e:
                code = getattr(e.response, "status_code", None)
                if code in (404, 410):                  # turned off on the phone
                    gone.append(p["endpoint"])
                else:
                    log.warning("phone alert not sent (%s): %s", code, str(e)[:160])
            except Exception as e:                          # noqa: BLE001
                log.warning("phone alert not sent: %s", str(e)[:160])
        for endpoint in gone:
            self.unsubscribe(endpoint)
        return sent

    @staticmethod
    def message(row):
        """The alert for a new record, or None if it is not one to send."""
        cam = (config.camera(row["camera"]) or {}).get("name", row["camera"])
        at = str(row.get("ts", ""))[11:16]
        if row["kind"] == "restricted_alert":
            n = int(row.get("value") or 1)
            return {"title": "Restricted area - {}".format(cam), "urgent": True,
                    "body": "{} {} seen at {}".format(n, "person" if n == 1 else "people", at),
                    "url": "/restricted", "tag": "alert-{}".format(row["id"])}
        if row["kind"] == "night_activity":
            what = row.get("label") or "activity"
            title = {"person": "Night - somebody is in the office", "movement": "Night - movement",
                     "lights on": "Night - lights switched on",
                     "lights off": "Night - lights switched off"}.get(what, "Night - " + what)
            return {"title": title, "body": "{} at {}".format(cam, at), "url": "/night",
                    "tag": "night-{}".format(row["id"]), "urgent": what == "person"}
        meta = row.get("meta") or {}
        if isinstance(meta, str):              # rows_after() gives the meta as written
            try:
                meta = json.loads(meta)
            except ValueError:
                meta = {}
        shot = (meta.get("photos") or [{}])[0].get("full")
        image = "/photos/" + shot if shot else None
        if row["kind"] == "parking" and not meta.get("was_there"):
            late = bool(meta.get("after_hours"))
            what = {"car": "Car", "bike": "Bike", "cycle": "Cycle"}.get(meta.get("type"), "Vehicle")
            return {"title": "New vehicle{} - {}".format(" after hours" if late else "", cam),
                    "body": "{} came at {}".format(what, at), "url": "/parking", "image": image,
                    "tag": "parking-{}".format(row["id"]), "urgent": late}
        if row["kind"] == "camera_health":
            return {"title": "{} - {}".format(cam, meta.get("title") or row.get("label")),
                    "body": "{} Since {}.".format(meta.get("text") or "", str(meta.get("started") or "")[11:16]),
                    "url": "/health", "image": image, "tag": "health-{}".format(row["id"]),
                    "urgent": row.get("label") in ("covered", "moved")}
        return None

    def _run(self):
        last = events.max_id()                  # only what happens from now on
        while True:
            time.sleep(self.CHECK_EVERY)
            try:
                rows = events.rows_after(last, 1000)
                if rows:
                    last = rows[-1]["id"]
                if not self.phones():
                    continue
                for row in rows:
                    msg = self.message(row)
                    if msg:
                        n = self.send(msg)
                        log.info("phone alert '%s' sent to %s phone(s)", msg["title"], n)
            except Exception:                           # noqa: BLE001
                log.exception("phone alerts")
                time.sleep(10)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="phone-alerts", daemon=True)
        self._thread.start()


sender = Push()
