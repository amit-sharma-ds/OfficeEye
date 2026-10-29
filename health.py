"""Camera health: is every camera still giving a good picture?

Every CHECK_EVERY seconds each camera's newest frame is looked at - no
model, a few milliseconds a camera - with plain rules:

- offline: no picture from it for OFFLINE_AFTER seconds;
- stuck: pictures come, but not one pixel has changed for STUCK_AFTER s;
- covered: the picture is flat - one colour, no detail at all (a hand,
  cloth, tape or paint on the lens). In the dark after office hours a flat
  black picture is only a dark room, not counted;
- moved: the outlines in the picture (walls, doors, desks) no longer match
  the camera's normal view - light and people walking by hardly change
  them, a camera turned a little changes them a lot;
- blurry: far less sharp than the camera's own normal picture (dirt, a
  spider web, water on the lens, focus).

A problem counts once it lasts (CONFIRM checks, or the seconds above), then
it gets one record (camera_health) with a photo - for "moved" also the
normal view as it was - and a phone alert. The record closes by itself when
the picture is fine again.

The normal view and normal sharpness are learned per camera, one for day
pictures and one for infrared night pictures, and kept in data/health/.
After a camera was turned on purpose, "This is its normal view now" on the
Camera health page makes the picture of now its normal view.
"""

import json
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime, timedelta

import cv2
import numpy as np

import config
import events
import photos

log = logging.getLogger(__name__)

HEALTH_DIR = os.path.join(config.DATA_DIR, "health")
KIND = "camera_health"

# What each state means, in words anybody understands. (title, what to do)
STATES = {
    "ok":       ("Working fine", "Clear picture, pointing the usual way."),
    "offline":  ("Offline", "No video comes from this camera. Check its power, cable and network."),
    "stuck":    ("Picture stuck", "The picture has not changed for minutes. Restart the camera."),
    "covered":  ("Covered or blocked", "The picture is flat with no detail - something is in front of the lens."),
    "moved":    ("Moved or turned", "It no longer shows its normal view - somebody may have turned it."),
    "blurry":   ("Blurry", "The picture is not sharp any more. Clean the lens or fix its focus."),
    "dark":     ("Too dark to check", "The room is dark. Checked again when there is light."),
    "learning": ("Learning", "Learning what this camera normally shows - a few minutes."),
    "off":      ("No IP set", "This camera has no IP address yet."),
}
PROBLEMS = ("offline", "stuck", "covered", "moved", "blurry")


def office_open(office, when):
    hm = when.strftime("%H:%M")
    return (office.get("open") or "09:30") <= hm < (office.get("close") or "18:30")


def _grey(frame, width):
    h = max(1, int(frame.shape[0] * width / frame.shape[1]))
    return cv2.cvtColor(cv2.resize(frame, (width, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)


def _outlines(frame):
    """The picture's outlines, small and normalised: two of them match 0.75-1
    for the same view, below 0.4 once the camera is turned by a tenth."""
    g = cv2.GaussianBlur(cv2.resize(_grey(frame, 320), (160, 90), interpolation=cv2.INTER_AREA)
                         .astype(np.float32), (5, 5), 0)
    m = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    m -= m.mean()
    n = float(np.linalg.norm(m))
    return m / n if n > 1e-6 else None


def _night(frame):
    """An infrared picture: no colour in it at all."""
    hsv = cv2.cvtColor(cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean()) < 12


class Monitor:
    CHECK_EVERY = 30.0
    START_GRACE = 180.0        # after the app starts, cameras are still connecting
    OFFLINE_AFTER = 120.0      # no picture this long: offline
    STUCK_AFTER = 180.0        # not one pixel changed this long: stuck
    CONFIRM = {"covered": 2, "moved": 4, "blurry": 10}    # checks in a row
    CLEAR_AFTER = 2            # good checks in a row before a problem is over
    FLAT_STD = 12.0            # grey spread below this ...
    FLAT_SHARP = 40.0          # ... and hardly any edges: a flat picture (normal: 30+ and 1500+)
    DARK = 40.0                # mean grey below this after hours: just dark
    MOVED = 0.45               # outlines matching less than this: another view
    WELL = 0.8                 # matching this well: the normal view is refreshed with it
    REFRESH = 3 * 3600.0       # ... at most this often
    BLURRY = 0.25              # sharpness below this share of its normal: blurry
    SHARP_KEEP = 240           # normal sharpness: median of this many good checks
    SHARP_LEARN = 20           # ... known after this many

    def __init__(self):
        self._lock = threading.Lock()
        self._cams = {}            # camera -> state
        self._started = time.time()
        self._thread = None
        self._manager = None
        self._saved_at = 0.0

    # ---------- kept on disk ----------

    def _ref_path(self, camera, mode):
        return os.path.join(HEALTH_DIR, "{}_{}.jpg".format(camera, mode))

    def _load(self, camera):
        st = {"problem": None, "since": None, "event": None, "streak": {}, "good": 0,
              "down_since": None, "same_since": None, "tiny": None, "seq": None,
              "ref": {}, "ref_at": {}, "sharp": {"day": deque(maxlen=self.SHARP_KEEP),
                                                 "night": deque(maxlen=self.SHARP_KEEP)},
              "view": None, "checked": None, "match": None, "sharp_share": None, "mode": None}
        for mode in ("day", "night"):
            path = self._ref_path(camera, mode)
            img = cv2.imread(path) if os.path.exists(path) else None
            if img is not None:
                st["ref"][mode] = _outlines(img)
                st["ref_at"][mode] = os.path.getmtime(self._ref_path(camera, mode))
        saved = self._baselines().get(camera) or {}
        for mode in ("day", "night"):
            st["sharp"][mode].extend(saved.get(mode) or [])
        return st

    @staticmethod
    def _baselines():
        try:
            with open(os.path.join(HEALTH_DIR, "sharpness.json"), encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def _save_baselines(self):
        with self._lock:
            data = {cam: {m: [round(v, 1) for v in list(q)[-60:]] for m, q in st["sharp"].items()}
                    for cam, st in self._cams.items()}
        tmp = os.path.join(HEALTH_DIR, "sharpness.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, os.path.join(HEALTH_DIR, "sharpness.json"))

    def _set_ref(self, camera, st, mode, frame):
        out = _outlines(frame)
        if out is None:
            return
        small = frame if frame.shape[1] <= 960 else cv2.resize(
            frame, (960, int(frame.shape[0] * 960 / frame.shape[1])), interpolation=cv2.INTER_AREA)
        cv2.imwrite(self._ref_path(camera, mode), small, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        st["ref"][mode], st["ref_at"][mode] = out, time.time()

    # ---------- one look at one camera ----------

    def _look(self, camera, st, status, frame, seq, now, office):
        """What is wrong with the camera now (a problem key), or a calm state
        ("ok", "dark", "learning")."""
        up = status.get("state") == "connected" and frame is not None and \
            (status.get("frame_age") is None or status["frame_age"] < 60)
        if not up:
            st["down_since"] = st["down_since"] or now
            st["same_since"] = None
            return "offline" if now - st["down_since"] >= self.OFFLINE_AFTER else st["view"] or "learning"
        st["down_since"] = None

        tiny = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
        if st["tiny"] is not None and seq != st["seq"] and np.array_equal(tiny, st["tiny"]):
            st["same_since"] = st["same_since"] or now
        elif seq != st["seq"]:
            st["same_since"] = None
        st["tiny"], st["seq"] = tiny, seq
        if st["same_since"] and now - st["same_since"] >= self.STUCK_AFTER:
            return "stuck"

        g = _grey(frame, 320)
        mean, spread = float(g.mean()), float(g.std())
        sharp = float(cv2.Laplacian(g, cv2.CV_32F).var())
        mode = "night" if _night(frame) else "day"
        st["mode"] = mode
        if spread < self.FLAT_STD and sharp < self.FLAT_SHARP:
            if mean < self.DARK and not office_open(office, datetime.now()):
                return "dark"
            return "covered"
        if mean < self.DARK:
            return "dark"                        # a dim room: its outlines and sharpness say nothing

        ref = st["ref"].get(mode)
        out = _outlines(frame)
        if ref is None or out is None:
            if out is not None:
                self._set_ref(camera, st, mode, frame)       # its first picture: the normal view
            st["match"] = None
            return "learning"
        match = float((ref * out).sum())
        st["match"] = round(match, 2)
        if match < self.MOVED:
            return "moved"

        normal = st["sharp"][mode]
        share = sharp / float(np.median(normal)) if len(normal) >= self.SHARP_LEARN else None
        st["sharp_share"] = round(share, 2) if share is not None else None
        if share is not None and share < self.BLURRY:
            return "blurry"
        # A good picture: it teaches the normal sharpness and keeps the normal view up to date.
        normal.append(sharp)
        if match >= self.WELL and now - st["ref_at"].get(mode, 0) >= self.REFRESH:
            self._set_ref(camera, st, mode, frame)
        return "ok" if len(normal) >= self.SHARP_LEARN else "learning"

    def _settle(self, camera, st, found, frame, now):
        """Turn what one look found into the camera's state, with records:
        a problem counts once it lasts, and is over after good looks."""
        if found in ("offline", "stuck"):
            confirmed = found                    # these two already waited their time
            st["streak"] = {}
        elif found in self.CONFIRM:
            n = st["streak"].get(found, 0) + 1
            st["streak"] = {found: n}
            confirmed = found if n >= self.CONFIRM[found] else None
        else:
            st["streak"] = {}
            confirmed = None
        if confirmed and now - self._started < self.START_GRACE:
            confirmed = None                     # just started: cameras are still connecting

        if st["problem"]:
            if found == st["problem"]:
                st["good"] = 0                   # still there
            elif confirmed:
                self._close(st, now)             # another problem now
            else:
                st["good"] += 1
                if st["good"] >= self.CLEAR_AFTER:
                    self._close(st, now)
        if confirmed and not st["problem"]:
            self._open(camera, st, confirmed, frame, now)
        if st["problem"]:
            st["view"] = st["problem"]
        elif found in PROBLEMS:                  # not for sure yet: shown as before
            st["view"] = st["view"] if st["view"] in ("ok", "dark", "learning") else "learning"
        else:
            st["view"] = found
        st["checked"] = now

    def _open(self, camera, st, problem, frame, now):
        when = datetime.fromtimestamp(now)
        shots = []
        if frame is not None:
            shot = photos.save(camera, "health", frame, [], [], ts=when)
            if shot:
                shot["what"] = "now"
                shots.append(shot)
        if problem == "moved":
            path = self._ref_path(camera, st["mode"] or "day")
            before = cv2.imread(path) if os.path.exists(path) else None
            if before is not None:
                shot = photos.save(camera, "health", before, [], [], ts=when.replace(
                    microsecond=(when.microsecond // 1000 + 1) % 1000 * 1000))
                if shot:
                    shot["what"] = "normal view"
                    shots.append(shot)
        since = st["down_since"] if problem == "offline" else st["same_since"] if problem == "stuck" else now
        title, text = STATES[problem]
        meta = {"problem": problem, "title": title, "text": text, "photos": shots,
                "started": datetime.fromtimestamp(since or now).isoformat(timespec="seconds"),
                "ended": None}
        st["event"] = events.log(kind=KIND, camera=camera, label=problem, value=1, meta=meta)
        st["problem"], st["since"], st["good"] = problem, since or now, 0
        log.info("camera health: %s %s", camera, problem)

    @staticmethod
    def _close(st, now):
        if st["event"]:
            events.update_meta(st["event"], {"ended": datetime.fromtimestamp(now).isoformat(timespec="seconds")})
        st["problem"] = st["since"] = st["event"] = None
        st["good"] = 0

    def _resume(self):
        """After a restart, problems still open go on in their own records."""
        since = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        for r in events.between([KIND], since, "9999", limit=500):
            if r["meta"].get("ended"):
                continue
            st = self._cams.get(r["camera"])
            if st is None or st["event"]:
                events.update_meta(r["id"], {"ended": datetime.now().isoformat(timespec="seconds")})
                continue
            try:
                started = datetime.fromisoformat(r["meta"].get("started") or r["ts"]).timestamp()
            except ValueError:
                started = time.time()
            st.update(event=r["id"], problem=r["label"], since=started, view=r["label"])
            if r["label"] == "offline":
                st["down_since"] = started

    # ---------- the watch ----------

    def start(self, manager):
        if self._thread and self._thread.is_alive():
            return
        self._manager = manager
        self._started = time.time()
        self._thread = threading.Thread(target=self._run, name="camera-health", daemon=True)
        self._thread.start()

    def _run(self):
        os.makedirs(HEALTH_DIR, exist_ok=True)
        with self._lock:
            for cam in config.cameras():
                self._cams.setdefault(cam["id"], self._load(cam["id"]))
        try:
            self._resume()
        except Exception:                                   # noqa: BLE001
            log.exception("camera health: open records")
        while True:
            try:
                self.check_all()
                if time.time() - self._saved_at > 600:
                    self._save_baselines()
                    self._saved_at = time.time()
            except Exception:                               # noqa: BLE001
                log.exception("camera health")
            time.sleep(self.CHECK_EVERY)

    def check_all(self):
        now = time.time()
        office = config.get("office")
        status = self._manager.status()
        for cam_id, s in status.items():
            with self._lock:
                st = self._cams.get(cam_id)
                if st is None:
                    st = self._cams[cam_id] = self._load(cam_id)
            if s.get("state") == "off":
                if st["problem"]:
                    self._close(st, now)
                st["view"], st["checked"] = "off", now
                continue
            stream = self._manager.get(cam_id)
            frame, _, seq = stream.latest() if stream else (None, 0, None)
            found = self._look(cam_id, st, s, frame, seq, now, office)
            self._settle(cam_id, st, found, frame if found != "offline" else None, now)
        with self._lock:
            for cam_id in [c for c in self._cams if c not in status]:   # removed cameras
                st = self._cams.pop(cam_id)
                if st["problem"]:
                    self._close(st, now)

    # ---------- what the pages read ----------

    def report(self):
        """{camera: {state, title, text, since, checked, match, mode}}, and the
        counts of each state."""
        now = time.time()
        out, counts = {}, {}
        with self._lock:
            items = list(self._cams.items())
        for cam_id, st in items:
            state = st["view"] or "learning"
            title, text = STATES[state]
            since = st["since"] if st["problem"] else None
            out[cam_id] = {"state": state, "problem": state in PROBLEMS, "title": title, "text": text,
                           "since": datetime.fromtimestamp(since).isoformat(timespec="seconds") if since else None,
                           "for": int(now - since) if since else None,
                           "checked_ago": int(now - st["checked"]) if st["checked"] else None,
                           "match": st["match"], "sharp": st["sharp_share"], "mode": st["mode"],
                           "has_normal": bool(st["ref"]), "event": st["event"]}
            counts[state] = counts.get(state, 0) + 1
        return {"cameras": out, "counts": counts}

    def problems(self):
        with self._lock:
            return sum(1 for st in self._cams.values() if st["problem"])

    def normal_view(self, camera):
        """Path of the camera's normal view picture (day, or night if that
        is all there is), or None."""
        with self._lock:
            mode = (self._cams.get(camera) or {}).get("mode") or "day"
        for m in (mode, "day", "night"):
            p = self._ref_path(camera, m)
            if os.path.exists(p):
                return p
        return None

    def accept_view(self, camera):
        """The camera was turned on purpose: what it shows now is its normal view."""
        stream = self._manager.get(camera) if self._manager else None
        frame = stream.read() if stream else None
        if frame is None:
            return False
        mode = "night" if _night(frame) else "day"
        with self._lock:
            st = self._cams.get(camera)
        if st is None:
            return False
        self._set_ref(camera, st, mode, frame)
        st["streak"] = {}
        if st["problem"] == "moved":
            self._close(st, time.time())
            st["view"] = "ok"
        return True


monitor = Monitor()
