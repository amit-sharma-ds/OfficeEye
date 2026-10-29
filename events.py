"""Event log. Every detection writes a row here, and the dashboard and any
report read from here. SQLite keeps it to one file, no server to install."""

import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "events.db")

_lock = threading.Lock()
_conn = None
_peak_cache = {}

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       TEXT    NOT NULL,          -- ISO local time
    day      TEXT    NOT NULL,          -- YYYY-MM-DD, for fast per-day queries
    camera   TEXT    NOT NULL,          -- camera id
    kind     TEXT    NOT NULL,          -- person_count, phone_use, restricted_alert, ...
    label    TEXT    DEFAULT '',
    value    REAL    DEFAULT 0,
    meta     TEXT    DEFAULT '{}'       -- json blob, detector specific (photos, ...)
);
CREATE INDEX IF NOT EXISTS idx_events_day  ON events(day);
CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind);
CREATE INDEX IF NOT EXISTS idx_events_kind_day ON events(kind, day);

-- How long each camera was connected, hour by hour. Time the app itself was
-- not running is in neither column: nobody was watching.
CREATE TABLE IF NOT EXISTS uptime (
    day         TEXT    NOT NULL,       -- YYYY-MM-DD
    hour        INTEGER NOT NULL,       -- 0-23
    camera      TEXT    NOT NULL,
    online      REAL    NOT NULL DEFAULT 0,   -- seconds connected
    offline     REAL    NOT NULL DEFAULT 0,   -- seconds not connected
    drops       INTEGER NOT NULL DEFAULT 0,   -- times the stream was lost
    last_online TEXT,                         -- ISO time last seen connected
    PRIMARY KEY (day, hour, camera)
);

-- One row per camera per finished day, written just after midnight from the
-- hourly rows above. Kept for good, so the history outlives the 90 days.
CREATE TABLE IF NOT EXISTS camera_days (
    day         TEXT    NOT NULL,
    camera      TEXT    NOT NULL,
    name        TEXT    NOT NULL DEFAULT '',  -- the camera's name that day
    running     REAL    NOT NULL DEFAULT 0,   -- seconds connected
    off         REAL    NOT NULL DEFAULT 0,   -- seconds not connected
    not_watched REAL    NOT NULL DEFAULT 0,   -- seconds the app was not running
    drops       INTEGER NOT NULL DEFAULT 0,
    last_online TEXT,
    PRIMARY KEY (day, camera)
);

-- Records deleted by hand (Restricted area page), so the cloud view and
-- the MongoDB copy drop them too.
CREATE TABLE IF NOT EXISTS deleted (
    id          INTEGER PRIMARY KEY,      -- the event's id
    at          TEXT    NOT NULL          -- ISO time it was deleted
);
"""


def _get():
    global _conn
    if _conn is None:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.executescript(SCHEMA)
        _conn.commit()
    return _conn


def log(kind, camera="", label="", value=0, meta=None, ts=None):
    """Write one row and return its id."""
    ts = ts or datetime.now()
    with _lock:
        conn = _get()
        cur = conn.execute(
            "INSERT INTO events (ts, day, camera, kind, label, value, meta)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ts.isoformat(timespec="seconds"), ts.strftime("%Y-%m-%d"), camera,
             kind, label, float(value), json.dumps(meta or {})),
        )
        conn.commit()
        return cur.lastrowid


def update_meta(event_id, patch):
    """Merge keys into an event's meta - photos of an alert that is still
    going on, or when it ended. A None value removes the key."""
    with _lock:
        conn = _get()
        row = conn.execute("SELECT meta FROM events WHERE id = ?",
                           (event_id,)).fetchone()
        if row is None:
            return
        meta = _meta(row["meta"])
        for key, value in patch.items():
            if value is None:
                meta.pop(key, None)
            else:
                meta[key] = value
        conn.execute("UPDATE events SET meta = ? WHERE id = ?",
                     (json.dumps(meta), event_id))
        conn.commit()


def _meta(text):
    try:
        return json.loads(text or "{}")
    except ValueError:
        return {}


def _row(r):
    d = dict(r)
    d["meta"] = _meta(d.get("meta"))
    return d


def recent(limit=100, kind=None, day=None, camera=None, with_photos=False):
    sql = "SELECT * FROM events"
    where, args = [], []
    if kind:
        where.append("kind = ?")
        args.append(kind)
    if day:
        where.append("day = ?")
        args.append(day)
    if camera:
        where.append("camera = ?")
        args.append(camera)
    if with_photos:
        where.append("meta LIKE '%\"photos\"%'")
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id DESC LIMIT ?"
    args.append(int(limit))
    with _lock:
        rows = _get().execute(sql, args).fetchall()
    return [_row(r) for r in rows]


def latest_id(kind):
    with _lock:
        row = _get().execute("SELECT MAX(id) AS id FROM events WHERE kind = ?",
                             (kind,)).fetchone()
    return row["id"] or 0


def latest_of(kinds):
    """Newest id among these kinds (0 if none)."""
    with _lock:
        row = _get().execute("SELECT MAX(id) AS id FROM events WHERE kind IN ({})".format(
            ",".join("?" * len(kinds))), list(kinds)).fetchone()
    return row["id"] or 0


def after_of(kinds, after_id, limit=20):
    """Rows of these kinds newer than after_id, oldest first."""
    with _lock:
        rows = _get().execute("SELECT * FROM events WHERE id > ? AND kind IN ({}) ORDER BY id LIMIT ?".format(
            ",".join("?" * len(kinds))), [after_id, *kinds, int(limit)]).fetchall()
    return [_row(r) for r in rows]


def get(event_id):
    with _lock:
        row = _get().execute("SELECT * FROM events WHERE id = ?",
                             (event_id,)).fetchone()
    return _row(row) if row else None


def delete(event_id, kind):
    """Delete one record by hand. Returns its meta (to remove its photos),
    or None when there is no such record of that kind."""
    with _lock:
        conn = _get()
        row = conn.execute("SELECT meta FROM events WHERE id = ? AND kind = ?",
                           (event_id, kind)).fetchone()
        if row is None:
            return None
        conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
        conn.execute("INSERT OR REPLACE INTO deleted (id, at) VALUES (?, ?)",
                     (event_id, datetime.now().isoformat(timespec="seconds")))
        conn.commit()
    _peak_cache.clear()
    _day_cache.clear()
    return _meta(row["meta"])


def deleted_since(days=3):
    """Ids deleted by hand lately - sent on to the cloud and MongoDB."""
    since = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    with _lock:
        rows = _get().execute("SELECT id FROM deleted WHERE at >= ?", (since,)).fetchall()
    return [r["id"] for r in rows]


def drop_rows(ids):
    """The cloud side: forget records deleted on the office PC. Returns the
    metas of the ones it still had (to remove their photos)."""
    ids = [int(i) for i in ids or []]
    if not ids:
        return []
    marks = ",".join("?" * len(ids))
    with _lock:
        conn = _get()
        rows = conn.execute("SELECT meta FROM events WHERE id IN ({})".format(marks), ids).fetchall()
        if rows:
            conn.execute("DELETE FROM events WHERE id IN ({})".format(marks), ids)
            conn.commit()
    if rows:
        _peak_cache.clear()
        _day_cache.clear()
    return [_meta(r["meta"]) for r in rows]


def count_today(kind, with_photos=False):
    day = datetime.now().strftime("%Y-%m-%d")
    sql = "SELECT COUNT(*) AS n FROM events WHERE kind = ? AND day = ?"
    if with_photos:
        sql += " AND meta LIKE '%\"photos\"%'"
    with _lock:
        return _get().execute(sql, (kind, day)).fetchone()["n"]


def count_by_kind(day=None):
    day = day or datetime.now().strftime("%Y-%m-%d")
    with _lock:
        rows = _get().execute(
            "SELECT kind, COUNT(*) AS n FROM events WHERE day = ? GROUP BY kind",
            (day,)).fetchall()
    return {r["kind"]: r["n"] for r in rows}


def peak(kind, day=None):
    """Highest value each camera reached on a day - the busiest moment.
    Cached for a few seconds: the dashboard asks every couple of seconds."""
    day = day or datetime.now().strftime("%Y-%m-%d")
    key = (kind, day)
    hit = _peak_cache.get(key)
    if hit and time.time() - hit[0] < 5:
        return hit[1]
    with _lock:
        rows = _get().execute(
            "SELECT camera, MAX(value) AS top FROM events"
            " WHERE kind = ? AND day = ? GROUP BY camera", (kind, day)).fetchall()
    out = {r["camera"]: int(r["top"]) for r in rows}
    _peak_cache[key] = (time.time(), out)
    return out


_day_cache = {}


def people_by_hour(day=None, step=60):
    """People in view across all counting cameras through a day. Each camera
    logs its count only when it changes, so its last count is carried
    forward and the total is sampled every `step` seconds. Returns the
    average and the most for each hour, and the busiest moment of the day.
    Cached for a minute."""
    now = datetime.now()
    day = day or now.strftime("%Y-%m-%d")
    hit = _day_cache.get(day)
    if hit and time.time() - hit[0] < 60:
        return hit[1]
    with _lock:
        rows = _get().execute(
            "SELECT ts, camera, value FROM events"
            " WHERE kind = 'person_count' AND day = ? ORDER BY id", (day,)).fetchall()
    hours, peak = {}, {"value": 0, "time": None}
    if rows:
        start = datetime.fromisoformat(rows[0]["ts"])
        end = now if day == now.strftime("%Y-%m-%d") else \
            datetime.fromisoformat(day) + timedelta(days=1)
        current, i, t = {}, 0, start
        while t <= end:
            while i < len(rows) and rows[i]["ts"] <= t.isoformat(timespec="seconds"):
                current[rows[i]["camera"]] = rows[i]["value"]
                i += 1
            total = int(sum(current.values()))
            h = hours.setdefault(t.hour, [0, 0, 0])     # sum, samples, most
            h[0] += total
            h[1] += 1
            h[2] = max(h[2], total)
            if total > peak["value"]:
                peak = {"value": total, "time": t.strftime("%H:%M")}
            t += timedelta(seconds=step)
    out = {"day": day,
           "hours": [{"hour": k, "avg": round(v[0] / v[1], 1), "most": v[2]}
                     for k, v in sorted(hours.items())],
           "peak": peak}
    _day_cache[day] = (time.time(), out)
    return out


def add_uptime(rows):
    """Add to the hourly uptime counters. rows: (day, hour, camera, online s,
    offline s, drops, last_online iso or None)."""
    if not rows:
        return
    with _lock:
        conn = _get()
        conn.executemany(
            "INSERT INTO uptime (day, hour, camera, online, offline, drops, last_online)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(day, hour, camera) DO UPDATE SET"
            "  online = online + excluded.online,"
            "  offline = offline + excluded.offline,"
            "  drops = drops + excluded.drops,"
            "  last_online = COALESCE(excluded.last_online, last_online)", rows)
        conn.commit()


def uptime(day, days=1):
    """Per camera: seconds online and offline, drops and when it was last
    connected, over `days` days ending with `day`. A single day also gets
    its 24 hours, for the timeline."""
    end = datetime.strptime(day, "%Y-%m-%d")
    first = (end - timedelta(days=days - 1)).strftime("%Y-%m-%d")
    with _lock:
        rows = _get().execute(
            "SELECT day, hour, camera, online, offline, drops, last_online FROM uptime"
            " WHERE day BETWEEN ? AND ? ORDER BY day, hour", (first, day)).fetchall()
    out = {}
    for r in rows:
        c = out.setdefault(r["camera"], {"online": 0.0, "offline": 0.0, "drops": 0,
                                         "last_online": None, "hours": {}})
        c["online"] += r["online"]
        c["offline"] += r["offline"]
        c["drops"] += r["drops"]
        if r["last_online"] and (c["last_online"] or "") < r["last_online"]:
            c["last_online"] = r["last_online"]
        if days == 1:
            c["hours"][r["hour"]] = [round(r["online"]), round(r["offline"])]
    return {"first": first, "last": day, "cameras": out}


def close_days(today, names=None):
    """Save the daily record of every finished day (before `today`) that has
    hourly rows but no record yet. Returns the days written."""
    names = names or {}
    with _lock:
        conn = _get()
        days = [r["day"] for r in conn.execute(
            "SELECT DISTINCT day FROM uptime WHERE day < ?"
            " AND day NOT IN (SELECT DISTINCT day FROM camera_days)", (today,))]
        for day in days:
            rows = conn.execute(
                "SELECT camera, SUM(online) AS running, SUM(offline) AS off,"
                " SUM(drops) AS drops, MAX(last_online) AS last_online"
                " FROM uptime WHERE day = ? GROUP BY camera", (day,)).fetchall()
            conn.executemany(
                "INSERT OR REPLACE INTO camera_days (day, camera, name, running, off,"
                " not_watched, drops, last_online) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [(day, r["camera"], names.get(r["camera"], r["camera"]),
                  round(r["running"]), round(r["off"]),
                  max(0, 86400 - round(r["running"]) - round(r["off"])),
                  r["drops"], r["last_online"]) for r in rows])
        conn.commit()
    return days


def camera_days(first, last, camera=None):
    """Saved daily records from `first` to `last`, newest day first."""
    sql = "SELECT * FROM camera_days WHERE day BETWEEN ? AND ?"
    args = [first, last]
    if camera:
        sql += " AND camera = ?"
        args.append(camera)
    with _lock:
        rows = _get().execute(sql + " ORDER BY day DESC, camera", args).fetchall()
    return [dict(r) for r in rows]


def between(kinds, start, end, camera=None, label_in=None, limit=2000):
    """Rows of these kinds with start <= ts < end (ISO times), newest first."""
    sql = "SELECT * FROM events WHERE ts >= ? AND ts < ? AND kind IN ({})".format(
        ",".join("?" * len(kinds)))
    args = [start, end, *kinds]
    if camera:
        sql += " AND camera = ?"
        args.append(camera)
    if label_in:
        sql += " AND label IN ({})".format(",".join("?" * len(label_in)))
        args += list(label_in)
    sql += " ORDER BY ts DESC, id DESC LIMIT ?"
    args.append(int(limit))
    with _lock:
        rows = _get().execute(sql, args).fetchall()
    return [_row(r) for r in rows]


# ---------------------------------------------------------------- copying to the cloud view
# The office app sends its rows to the cloud copy (cloud.py), which stores
# them under the same ids, so every page there reads them as usual.

def max_id():
    with _lock:
        return _get().execute("SELECT MAX(id) AS id FROM events").fetchone()["id"] or 0


def first_id_since(day):
    with _lock:
        row = _get().execute("SELECT MIN(id) AS id FROM events WHERE day >= ?",
                             (day,)).fetchone()
    return row["id"]


def rows_after(after_id, limit=1000):
    with _lock:
        rows = _get().execute("SELECT * FROM events WHERE id > ? ORDER BY id LIMIT ?",
                              (after_id, limit)).fetchall()
    return [dict(r) for r in rows]


def rows_of_day(kinds, day):
    """Today's alerts and phone rows: they gain photos after they are first
    written, so they are sent again now and then."""
    with _lock:
        rows = _get().execute(
            "SELECT * FROM events WHERE day = ? AND kind IN ({})".format(",".join("?" * len(kinds))),
            [day, *kinds]).fetchall()
    return [dict(r) for r in rows]


def store_rows(rows):
    """Write rows exactly as given, ids included (the cloud side)."""
    if not rows:
        return
    with _lock:
        conn = _get()
        conn.executemany(
            "INSERT OR REPLACE INTO events (id, ts, day, camera, kind, label, value, meta)"
            " VALUES (:id, :ts, :day, :camera, :kind, :label, :value, :meta)", rows)
        conn.commit()
    _peak_cache.clear()
    _day_cache.clear()


def uptime_rows(first_day):
    with _lock:
        rows = _get().execute("SELECT * FROM uptime WHERE day >= ?", (first_day,)).fetchall()
    return [dict(r) for r in rows]


def store_uptime(rows, days):
    """Replace the hourly rows of whole days (the cloud side)."""
    with _lock:
        conn = _get()
        for day in days:
            conn.execute("DELETE FROM uptime WHERE day = ?", (day,))
        conn.executemany(
            "INSERT OR REPLACE INTO uptime (day, hour, camera, online, offline, drops, last_online)"
            " VALUES (:day, :hour, :camera, :online, :offline, :drops, :last_online)", rows)
        conn.commit()


def store_camera_days(rows):
    with _lock:
        conn = _get()
        conn.executemany(
            "INSERT OR REPLACE INTO camera_days (day, camera, name, running, off, not_watched,"
            " drops, last_online) VALUES (:day, :camera, :name, :running, :off, :not_watched,"
            " :drops, :last_online)", rows)
        conn.commit()


def purge_older_than(days=90):
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    with _lock:
        conn = _get()
        conn.execute("DELETE FROM events WHERE day < ?", (cutoff,))
        conn.execute("DELETE FROM uptime WHERE day < ?", (cutoff,))
        conn.commit()
