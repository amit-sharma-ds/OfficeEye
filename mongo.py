"""MongoDB copy of OfficeEye's records.

The app keeps working on its local SQLite database (fast, and nothing stops
when the internet drops); this copies every record to MongoDB Atlas in the
background, database `officeeye`:

    events        alerts, phone uses, washroom visits, people counts ...
                  (_id = the local id; meta as a sub-document)
    uptime        camera running time per hour   (_id = day|hour|camera)
    camera_days   camera running time per day    (_id = day|camera)
    sync_state    how far the copy has got

Photos stay on the office PC; the records name their files.

The connection lives in data/config.json ("mongodb": {"uri", "db",
"enabled"}) - never in the repository.
"""

import json
import logging
import threading
import time
from datetime import datetime, timedelta

import config
import events

log = logging.getLogger("mongo")

CHANGING = ["restricted_alert", "phone_use", "washroom_visit", "washroom_in"]   # get photos / end times later


def _client(uri):
    from pymongo import MongoClient
    try:
        cl = MongoClient(uri, serverSelectionTimeoutMS=20000, appname="OfficeEye")
    except Exception as e:                                   # noqa: BLE001
        if "nameserver" not in str(e).lower() and "dns" not in str(e).lower():
            raise
        # Python's DNS library cannot always read Windows' DNS settings; the
        # mongodb+srv address then needs a public resolver to be found.
        import dns.resolver
        r = dns.resolver.Resolver(configure=False)
        r.nameservers = ["8.8.8.8", "1.1.1.1"]
        dns.resolver.default_resolver = r
        cl = MongoClient(uri, serverSelectionTimeoutMS=20000, appname="OfficeEye")
    cl.admin.command("ping")
    return cl


def _doc(row):
    meta = row.get("meta")
    try:
        meta = json.loads(meta) if isinstance(meta, str) else (meta or {})
    except ValueError:
        meta = {}
    return {"_id": row["id"], "ts": row["ts"], "day": row["day"], "camera": row["camera"],
            "kind": row["kind"], "label": row["label"], "value": row["value"], "meta": meta}


class Sync:
    def __init__(self):
        self.state, self.error, self.ok_at, self.copied = "off", "", 0.0, 0
        self._db = None

    def info(self):
        cfg = config.get("mongodb")
        return {"enabled": bool(cfg.get("enabled") and cfg.get("uri")), "db": cfg.get("db", "officeeye"),
                "state": self.state, "error": self.error, "copied": self.copied,
                "ok_ago": round(time.time() - self.ok_at, 1) if self.ok_at else None}

    def start(self):
        threading.Thread(target=self._run, name="mongo-sync", daemon=True).start()

    def _upsert(self, coll, docs):
        from pymongo import ReplaceOne
        if docs:
            self._db[coll].bulk_write([ReplaceOne({"_id": d["_id"]}, d, upsert=True) for d in docs],
                                      ordered=False)

    def _run(self):
        last_refresh = last_uptime = last_days = 0.0
        backoff = 0
        while True:
            cfg = config.get("mongodb")
            if not (cfg.get("enabled") and cfg.get("uri")):
                self.state, self._db = "off", None
                time.sleep(10)
                continue
            try:
                if self._db is None:
                    self._db = _client(cfg["uri"])[cfg.get("db") or "officeeye"]
                    self._db.events.create_index([("day", 1), ("kind", 1)])
                    self._db.events.create_index("camera")
                    log.info("MongoDB connected, database %s", self._db.name)
                st = self._db.sync_state.find_one({"_id": "events"}) or {}
                last_id = int(st.get("last_id") or 0)
                rows = events.rows_after(last_id, 2000)
                self._upsert("events", [_doc(r) for r in rows])
                if rows:
                    last_id = rows[-1]["id"]
                    self._db.sync_state.replace_one({"_id": "events"}, {"_id": "events", "last_id": last_id,
                                                    "at": datetime.now().isoformat(timespec="seconds")},
                                                    upsert=True)
                    self.copied += len(rows)
                now = time.time()
                today = datetime.now()
                if now - last_refresh >= 120:
                    last_refresh = now
                    self._upsert("events", [_doc(r) for r in events.rows_of_day(
                        CHANGING, today.strftime("%Y-%m-%d"))])
                if now - last_uptime >= 300:
                    last_uptime = now
                    first = (today - timedelta(days=1)).strftime("%Y-%m-%d")
                    self._upsert("uptime", [dict(r, _id="{}|{}|{}".format(r["day"], r["hour"], r["camera"]))
                                            for r in events.uptime_rows(first)])
                if now - last_days >= 1800:
                    last_days = now
                    rows_d = events.camera_days((today - timedelta(days=400)).strftime("%Y-%m-%d"),
                                                today.strftime("%Y-%m-%d"))
                    self._upsert("camera_days", [dict(r, _id="{}|{}".format(r["day"], r["camera"]))
                                                 for r in rows_d])
                self.state, self.error, self.ok_at, backoff = "ok", "", time.time(), 0
                # Catching up on old records: straight on; else every 30 s.
                time.sleep(1 if len(rows) == 2000 else 30)
            except Exception as e:                           # noqa: BLE001
                self._db = None
                self.state, self.error = "error", str(e).split("\n")[0][:300]
                backoff = min(300, backoff * 2 or 15)
                log.warning("MongoDB copy: %s (retry in %s s)", self.error, backoff)
                time.sleep(backoff)


sync = Sync()
