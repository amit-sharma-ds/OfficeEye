"""OfficeEye's MongoDB database (`office_eye`).

The app keeps working on its local SQLite database (fast, and nothing stops
when the internet drops); this module connects to MongoDB and copies every
record there in the background.

Connection: MONGO_URI, from the environment or the `.env` file in the project
folder, ending in the database name: mongodb+srv://user:pass@host/office_eye.
It is never written to a file or a log by this code. One client is created
and shared by everything (get_db).

Collections - shaped by what OfficeEye records (it does not identify
people, so there are no employee or attendance records):

    cameras               one per camera: name, purpose, IP, zones
                          (_id = camera id; no RTSP link or password)
    detections            everything the cameras saw: people counts,
                          phone uses, restricted-area alerts, washroom
                          visits and checks (_id = the local id;
                          kind, ts, day, camera, label, value, meta)
    camera_uptime_hourly  minutes each camera ran per hour (_id = day|hour|camera)
    camera_uptime_daily   the same per day                (_id = day|camera)
    sync_state            how far the copy has got

Photos stay on the office PC; the records name their files.
"""

import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timedelta

import config
import events

log = logging.getLogger("mongo")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = "office_eye"
CHANGING = ["restricted_alert", "phone_use", "washroom_visit", "washroom_in"]   # get photos / end times later


class MongoSetupError(RuntimeError):
    """MongoDB cannot be used - the message says why, without the password."""


# ---------------------------------------------------------------- connection

def _load_env(path=os.path.join(BASE_DIR, ".env")):
    """KEY=value lines from .env into the environment (what is set already wins)."""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                if key.startswith("export "):
                    key = key[7:].strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        pass


def _clean(text):
    """An error message with any connection string's password taken out."""
    return re.sub(r"(mongodb(?:\+srv)?://[^:/@\s]+:)[^@\s]+@", r"\1***@", str(text))


_lock = threading.Lock()
_client = None
_db = None


def get_db():
    """The shared `office_eye` database. Connects on first use and reuses the
    connection; raises MongoSetupError with a plain reason if it cannot."""
    global _client, _db
    with _lock:
        if _db is not None:
            return _db
        _load_env()
        uri = os.environ.get("MONGO_URI", "").strip()
        if not uri:
            raise MongoSetupError("MONGO_URI is not set - add it to the .env file in the project folder")
        try:
            from pymongo import MongoClient
            from pymongo.errors import ConfigurationError, OperationFailure, ServerSelectionTimeoutError
        except ImportError:
            raise MongoSetupError("pymongo is not installed - py -3.12 -m pip install pymongo") from None

        def connect():
            cl = MongoClient(uri, serverSelectionTimeoutMS=20000, appname="OfficeEye")
            cl.admin.command("ping")
            return cl

        try:
            try:
                client = connect()
            except ConfigurationError as e:
                if "nameserver" not in str(e).lower() and "dns" not in str(e).lower():
                    raise
                # Python's DNS library cannot always read Windows' DNS settings;
                # the mongodb+srv address is then looked up with a public resolver.
                import dns.resolver
                r = dns.resolver.Resolver(configure=False)
                r.nameservers = ["8.8.8.8", "1.1.1.1"]
                dns.resolver.default_resolver = r
                client = connect()
        except OperationFailure as e:
            raise MongoSetupError("MongoDB refused the user name or password in MONGO_URI ({})".format(
                _clean(e).split(",")[0])) from None
        except ServerSelectionTimeoutError:
            raise MongoSetupError("Could not reach MongoDB - check the internet, and that Atlas -> "
                                  "Network Access allows this PC's IP") from None
        except ConfigurationError as e:
            raise MongoSetupError("MONGO_URI looks wrong: {}".format(_clean(e))) from None
        except Exception as e:                               # noqa: BLE001
            raise MongoSetupError("Could not connect to MongoDB: {}".format(_clean(e))) from None
        name = client.get_default_database(default=DEFAULT_DB).name
        _client, _db = client, client[name]
        _db.detections.create_index([("day", 1), ("kind", 1)])
        _db.detections.create_index([("camera", 1), ("ts", -1)])
        log.info("MongoDB connected, database %s", name)
        return _db


def reset():
    """Drop the shared connection (after an error), so the next call reconnects."""
    global _client, _db
    with _lock:
        if _client is not None:
            try:
                _client.close()
            except Exception:                                # noqa: BLE001
                pass
        _client = _db = None


# ---------------------------------------------------------------- reading and writing

def _upsert(coll, docs):
    from pymongo import ReplaceOne
    if docs:
        get_db()[coll].bulk_write([ReplaceOne({"_id": d["_id"]}, d, upsert=True) for d in docs],
                                  ordered=False)
    return len(docs)


def detection_doc(row):
    """A local event row as a `detections` document."""
    meta = row.get("meta")
    try:
        meta = json.loads(meta) if isinstance(meta, str) else (meta or {})
    except ValueError:
        meta = {}
    return {"_id": row["id"], "ts": row["ts"], "day": row["day"], "camera": row["camera"],
            "kind": row["kind"], "label": row.get("label", ""), "value": row.get("value", 0),
            "meta": meta}


def insert_detections(rows):
    """Write (or rewrite) local event rows. Returns how many."""
    return _upsert("detections", [detection_doc(r) for r in rows])


def find_detections(day=None, kind=None, camera=None, limit=100):
    """Newest first."""
    q = {k: v for k, v in (("day", day), ("kind", kind), ("camera", camera)) if v}
    return list(get_db().detections.find(q).sort("ts", -1).limit(int(limit)))


def save_cameras(listing):
    """The camera list - no RTSP link or password is ever stored."""
    now = datetime.now().isoformat(timespec="seconds")
    docs = [{"_id": c["id"], "name": c.get("name", ""), "group": c.get("group", ""),
             "ip": c.get("ip", ""), "zone": c.get("zone") or [], "ignore": c.get("ignore") or [],
             "updated_at": now} for c in listing]
    return _upsert("cameras", docs)


def find_cameras(group=None):
    return list(get_db().cameras.find({"group": group} if group else {}).sort("_id", 1))


def save_uptime_hourly(rows):
    return _upsert("camera_uptime_hourly",
                   [dict(r, _id="{}|{}|{}".format(r["day"], r["hour"], r["camera"])) for r in rows])


def save_uptime_daily(rows):
    return _upsert("camera_uptime_daily", [dict(r, _id="{}|{}".format(r["day"], r["camera"])) for r in rows])


def find_uptime_daily(first, last, camera=None):
    q = {"day": {"$gte": first, "$lte": last}}
    if camera:
        q["camera"] = camera
    return list(get_db().camera_uptime_daily.find(q).sort([("day", -1), ("camera", 1)]))


# ---------------------------------------------------------------- the background copy

class Sync:
    def __init__(self):
        self.state, self.error, self.ok_at, self.copied = "off", "", 0.0, 0
        self._listing = None

    def info(self):
        _load_env()
        on = bool(os.environ.get("MONGO_URI")) and config.get("mongodb").get("enabled", True)
        return {"enabled": on, "db": _db.name if _db is not None else None,
                "state": self.state, "error": self.error, "copied": self.copied,
                "ok_ago": round(time.time() - self.ok_at, 1) if self.ok_at else None}

    def start(self, listing=None):
        self._listing = listing
        threading.Thread(target=self._run, name="mongo-sync", daemon=True).start()

    def _run(self):
        last_refresh = last_uptime = last_days = last_cams = 0.0
        backoff = 0
        while True:
            _load_env()
            if not os.environ.get("MONGO_URI") or not config.get("mongodb").get("enabled", True):
                self.state = "off"
                time.sleep(10)
                continue
            try:
                db = get_db()
                st = db.sync_state.find_one({"_id": "detections"}) or {}
                last_id = int(st.get("last_id") or 0)
                rows = events.rows_after(last_id, 2000)
                insert_detections(rows)
                if rows:
                    db.sync_state.replace_one(
                        {"_id": "detections"},
                        {"_id": "detections", "last_id": rows[-1]["id"],
                         "at": datetime.now().isoformat(timespec="seconds")}, upsert=True)
                    self.copied += len(rows)
                now = time.time()
                today = datetime.now()
                if now - last_refresh >= 120:
                    last_refresh = now
                    insert_detections(events.rows_of_day(CHANGING, today.strftime("%Y-%m-%d")))
                if self._listing and now - last_cams >= 300:
                    last_cams = now
                    save_cameras(self._listing())
                if now - last_uptime >= 300:
                    last_uptime = now
                    save_uptime_hourly(events.uptime_rows((today - timedelta(days=1)).strftime("%Y-%m-%d")))
                if now - last_days >= 1800:
                    last_days = now
                    save_uptime_daily(events.camera_days((today - timedelta(days=400)).strftime("%Y-%m-%d"),
                                                         today.strftime("%Y-%m-%d")))
                self.state, self.error, self.ok_at, backoff = "ok", "", time.time(), 0
                # Catching up on old records: straight on; else every 30 s.
                time.sleep(1 if len(rows) == 2000 else 30)
            except Exception as e:                           # noqa: BLE001
                reset()
                self.state, self.error = "error", _clean(e).split("\n")[0][:300]
                backoff = min(300, backoff * 2 or 15)
                log.warning("MongoDB copy: %s (retry in %s s)", self.error, backoff)
                time.sleep(backoff)


sync = Sync()
