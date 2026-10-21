"""Washroom visits: who went in, when, and for how long - seen from a camera
in the corridor outside the washroom door. Nothing inside the washroom is
ever filmed; the camera must only see the corridor and the outside of the
door.

How a visit is told apart:

1. The door zone is drawn once on the Washroom page: the door and the floor
   right in front of it.
2. Every decoded frame, a cheap picture check (no model) watches the zone:
   when the door opens or closes, or somebody steps there, it changes.
3. While that lasts the model looks twice a second, and also about once a
   second whenever something moves near the door, so it knows who was
   walking up before the door opened.
4. When the door goes still again it is decided:
     somebody was walking up to the door just before   -> went IN
     the door opened with nobody outside               -> came OUT
     the door moved and nobody was seen at all          -> door only
   An OUT is paired with the oldest open IN, which gives the time inside.
5. Every IN and OUT keeps the clearest photo of the person at the door, so
   the people who used the washroom since it was last checked clean can be
   looked at when it is found dirty or damaged.
"""

import logging
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

import config
import events
import photos
from detection import Detector, pipeline

log = logging.getLogger(__name__)

BLUE = (220, 120, 40)          # BGR


class _Door:
    """Per-camera state, guarded by the detector's lock."""

    def __init__(self):
        self.zone = []             # [[x, y], ...] 0-1
        self.mask = None           # zone mask on the small grey picture
        self.rect = None           # (x1, y1, x2, y2) of the zone, pixels, padded
        self.near_rect = None      # the zone grown, for "walking up to the door"
        self.size = None           # frame (w, h) the rects are for
        self.bg = None             # how the zone looks when nothing happens
        self.prev_near = None
        self.calm_since = 0.0
        self.active_at = 0.0       # last time the zone changed
        self.moving_at = 0.0       # last time something moved near the door
        self.busy_since = 0.0      # start of the current change (re-learn if endless)
        self.level = 0.0
        self.episode = None        # the door is in use right now
        self.last_near = 0.0       # last check that saw somebody near the door
        self.recent = deque(maxlen=6)   # best looks at people near the door, last seconds
        self.inside = []           # [(in event id, ts)] - went in, not yet out
        self.last_visit = None     # (kind, ts)


class WashroomVisits(Detector):
    key = "washroom"
    title = "Washroom visits"
    groups = (config.WASHROOM,)

    CHANGE = 25                # grey-level change that counts
    ACTIVE_SHARE = 0.06        # share of the zone that must change: door moving
    MOVE_SHARE = 0.01          # share of the area near the door: somebody walking
    QUIET = 4.0                # the door must be still this long to end a use
    MAX_USE = 120.0            # a use never lasts longer than this
    RELEARN = 180.0            # zone changed this long: that is the new normal
    APPROACH = 6.0             # seen near the door this soon before it moved: going in
    INSIDE_MAX = 45 * 60.0     # nobody stays in longer - a missed "out"
    NEAR_GROW = 0.6            # near the door = the zone grown by this much each side

    def __init__(self):
        super().__init__()
        self._state = {}

    def _st(self, camera):
        st = self._state.get(camera)
        if st is None:
            st = self._state[camera] = _Door()
        return st

    # ---------- geometry ----------

    def _fit(self, st, camera, w, h):
        zone = (config.camera(camera) or {}).get("zone") or []
        if zone == st.zone and st.size == (w, h):
            return bool(zone)
        st.zone, st.size, st.bg, st.prev_near = zone, (w, h), None, None
        st.mask = st.rect = st.near_rect = None
        if len(zone) < 3:
            return False
        pts = np.array([[x * w, y * h] for x, y in zone], np.float32)
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        st.rect = (int(max(0, x1)), int(max(0, y1)), int(min(w, x2 + 1)), int(min(h, y2 + 1)))
        gx, gy = (x2 - x1) * self.NEAR_GROW, (y2 - y1) * self.NEAR_GROW
        st.near_rect = (int(max(0, x1 - gx)), int(max(0, y1 - gy)),
                        int(min(w, x2 + gx)), int(min(h, y2 + gy)))
        rx1, ry1, rx2, ry2 = st.rect
        sw = 64
        sh = max(8, int(sw * (ry2 - ry1) / max(1, rx2 - rx1)))
        mask = np.zeros((sh, sw), np.uint8)
        scaled = (pts - [rx1, ry1]) * [sw / max(1, rx2 - rx1), sh / max(1, ry2 - ry1)]
        cv2.fillPoly(mask, [scaled.astype(np.int32)], 255)
        st.mask = mask > 0
        return True

    @staticmethod
    def _grey(frame, rect, width=64):
        x1, y1, x2, y2 = rect
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        h = max(8, int(width * (y2 - y1) / max(1, x2 - x1)))
        small = cv2.resize(crop, (width, h), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)

    def _foot_in(self, st, box, grow=0.0):
        """Is the bottom middle of a box (the feet) in the zone, or near it?"""
        w, h = st.size
        foot = ((box[0] + box[2]) / 2.0, float(box[3]))
        if grow:
            x1, y1, x2, y2 = st.near_rect
            return x1 <= foot[0] <= x2 and y1 <= foot[1] <= y2
        poly = np.array([[x * w, y * h] for x, y in st.zone], np.float32)
        return cv2.pointPolygonTest(poly, foot, False) >= 0

    # ---------- every decoded frame (camera threads) ----------

    def on_frame(self, camera, frame, ts):
        h, w = frame.shape[:2]
        with self._lock:
            st = self._st(camera)
            if not self._fit(st, camera, w, h):
                return
            cur = self._grey(frame, st.rect)
            near = self._grey(frame, st.near_rect, 80)
            if cur is None or cur.shape != st.mask.shape:
                return
            if st.bg is None or st.bg.shape != cur.shape:
                st.bg, st.calm_since = cur, ts
            diff = np.abs(cur - st.bg) > self.CHANGE
            st.level = float(diff[st.mask].mean()) if st.mask.any() else 0.0
            if st.level >= self.ACTIVE_SHARE:
                st.active_at = ts
                st.busy_since = st.busy_since or ts
                if ts - st.busy_since > self.RELEARN:      # door left open, a bag put down
                    st.bg, st.busy_since = cur, 0.0
            else:
                st.busy_since = 0.0
                st.bg = st.bg * 0.97 + cur * 0.03            # light changes slowly
            if near is not None:
                if st.prev_near is not None and st.prev_near.shape == near.shape:
                    if float((np.abs(near - st.prev_near) > self.CHANGE).mean()) >= self.MOVE_SHARE:
                        st.moving_at = ts
                st.prev_near = near
            if st.episode is None and st.level >= self.ACTIVE_SHARE:
                # The door just started moving: a use begins.
                st.episode = self._begin(st, ts, frame)
            elif st.episode is not None and ts - st.episode["start"] > self.MAX_USE + 60:
                st.episode = None                            # no checks came: give up

    def _begin(self, st, ts, frame):
        """A door use starts. Looks at people from just before it count -
        the person walking up - older ones belong to somebody else."""
        recent = [c for c in st.recent if ts - c["ts"] < self.APPROACH]
        return {"start": ts, "approached": bool(recent), "obs": [],
                "cands": recent, "frame": frame.copy()}

    def schedule(self, camera):
        """(seconds between model checks, rank) for the pipeline."""
        now = time.time()
        with self._lock:
            st = self._state.get(camera)
            if st is None or not st.zone:
                return 60.0, 3
            if st.episode is not None or now - st.active_at < self.QUIET:
                return 0.5, 1
            if now - st.moving_at < 2.0:
                return 1.0, 1
        return 60.0, 3

    # ---------- after a model check (pipeline worker) ----------

    def process(self, camera, frame, ts, seen):
        h, w = frame.shape[:2]
        produced = []
        with self._lock:
            st = self._st(camera)
            if not self._fit(st, camera, w, h):
                return None
            at_door = [b for b in seen.people if self._foot_in(st, b)]
            near = [b for b in seen.people if self._foot_in(st, b, grow=True)]
            if near:
                st.last_near = ts
                best = max(near, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]) * b[4])
                look = {"ts": ts, "frame": frame.copy(), "box": best,
                        "others": [b for b in seen.people if b is not best],
                        "score": (best[2] - best[0]) * (best[3] - best[1]) * best[4]}
                st.recent.append(look)
            ep = st.episode
            if ep is None and at_door:
                # Somebody at the door although it did not move (it was open):
                # walked up = seen near it in an earlier check too.
                ep = st.episode = self._begin(st, ts, frame)
                ep["approached"] = any(ts - c["ts"] > 0.2 for c in ep["cands"])
            if ep is not None:
                ep["obs"].append((ts, len(at_door), len(near)))
                if near:
                    ep["cands"].append(st.recent[-1])
                    ep["cands"] = sorted(ep["cands"], key=lambda c: -c["score"])[:4]
                still = ts - st.active_at >= self.QUIET and not at_door
                if still or ts - ep["start"] > self.MAX_USE:
                    produced = self._finish(camera, st, ep, ts)
                    st.episode = None
            st.inside = [i for i in st.inside if ts - i[1] < self.INSIDE_MAX]
        for ev in produced:
            self._log(camera, ev)
        return None

    def _finish(self, camera, st, ep, ts):
        """Decide what a door use was. Returns events to write (outside the lock)."""
        obs = ep["obs"]
        seen_any = any(o[2] for o in obs) or bool(ep["cands"])
        end_near = bool(obs and obs[-1][2])
        cands = sorted(ep["cands"], key=lambda c: -c["score"])
        if ep["approached"] and not end_near:
            kind = "in"
        elif not ep["approached"] and seen_any:
            kind = "out"
        elif not seen_any:
            kind = "door"
        else:
            return []          # walked up and is still there, or walked past
        return [{"kind": kind, "start": ep["start"], "ts": ts,
                 "cands": cands[:2], "frame": ep["frame"]}]

    def _log(self, camera, ev):
        zone = (config.camera(camera) or {}).get("zone") or []
        when = datetime.fromtimestamp(ev["start"])
        shots = []
        for c in ev["cands"]:
            shots.append(photos.save(camera, "washroom", c["frame"], [c["box"]], c["others"],
                                     zone=zone, colour=BLUE, ts=datetime.fromtimestamp(c["ts"])))
        if not shots:
            shots.append(photos.save(camera, "washroom", ev["frame"], [], [], zone=zone,
                                     colour=BLUE, ts=when))
        meta = {"photos": shots, "at": when.isoformat(timespec="seconds")}
        kind = ev["kind"]
        if kind == "in":
            eid = self.emit(camera, "washroom_in", 1, "went in", meta)
            with self._lock:
                self._st(camera).inside.append((eid, ev["start"]))
                self._st(camera).last_visit = ("in", ev["start"])
        elif kind == "out":
            with self._lock:
                st = self._st(camera)
                pair = st.inside.pop(0) if st.inside else None
                st.last_visit = ("out", ev["start"])
            if pair:
                secs = int(ev["start"] - pair[1])
                meta.update(in_id=pair[0], seconds=secs)
            eid = self.emit(camera, "washroom_out", 1, "came out", meta)
            if pair:
                events.update_meta(pair[0], {"out_id": eid, "out_at": meta["at"],
                                             "seconds": meta["seconds"]})
        else:
            self.emit(camera, "washroom_door", 0, "door used, nobody seen", meta)

    # ---------- read-outs ----------

    def overlay(self, camera, frame):
        with self._lock:
            st = self._state.get(camera)
            zone = list(st.zone) if st else []
            active = bool(st and st.episode is not None)
        if zone:
            photos.draw_zone(frame, zone, (40, 40, 220) if active else BLUE)
        return frame

    def caption(self, camera):
        info = self.info(camera)
        if not info["zone"]:
            return ("washroom: draw the door zone", False)
        if info["in_use"]:
            return ("washroom in use since {}".format(
                datetime.fromtimestamp(info["since"]).strftime("%H:%M")), False)
        return ("door moving" if info["door"] else "washroom free", False)

    def busy(self, camera):
        with self._lock:
            st = self._state.get(camera)
            return bool(st and st.episode is not None)

    def info(self, camera):
        now = time.time()
        with self._lock:
            st = self._state.get(camera)
            if st is None:
                return {"zone": bool((config.camera(camera) or {}).get("zone")),
                        "in_use": False, "since": 0, "door": False, "inside": 0}
            inside = [i for i in st.inside if now - i[1] < self.INSIDE_MAX]
            return {"zone": bool(st.zone), "in_use": bool(inside),
                    "since": inside[0][1] if inside else 0,
                    "door": st.episode is not None or now - st.active_at < self.QUIET,
                    "inside": len(inside)}

    def forget(self, camera):
        with self._lock:
            self._state.pop(camera, None)


detector = WashroomVisits()
pipeline.add(detector)
