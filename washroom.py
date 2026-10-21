"""Washroom door log: a photo with the time of everybody at the washroom
door, seen from a camera in the corridor outside it. Nothing inside the
washroom is ever filmed; the camera must only see the corridor and the
outside of the door.

How a door use is caught:

1. The door zone is drawn once on the Washroom page: the door (or its
   edge) and the spot in front of it.
2. Every decoded frame a cheap picture check (no model) watches the zone,
   so the door starting to move is noticed at once.
3. A use starts when the door moves *or* the model sees a person at the
   door - whichever comes first. While it lasts the model looks twice a
   second, and about once a second whenever something moves near the door.
4. When the door has been still for a few seconds and nobody is at it, the
   use is written down: when it started, how long the door area was busy,
   and photos - the clearest look at each person seen at the door, or the
   door itself when nobody could be seen (somebody may have been missed).

When the washroom is found dirty or damaged, the page lists the uses since
it was last checked clean, photos included.
"""

import logging
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

import config
import photos
from detection import Detector, pipeline

log = logging.getLogger(__name__)

BLUE = (220, 120, 40)          # BGR
RED = (40, 40, 220)


class _Door:
    """Per-camera state, guarded by the detector's lock."""

    def __init__(self):
        self.zone = []             # [[x, y], ...] 0-1
        self.mask = None           # zone mask on the small grey picture
        self.rect = None           # (x1, y1, x2, y2) of the zone, pixels
        self.door_rect = None      # the zone grown a little: "at the door"
        self.near_rect = None      # grown more: "near the door"
        self.size = None           # frame (w, h) the rects are for
        self.bg = None             # how the zone looks when nothing happens
        self.relearn = False
        self.prev_near = None
        self.active_at = 0.0       # last time the zone changed
        self.moving_at = 0.0       # last time something moved near the door
        self.busy_since = 0.0      # start of the current change
        self.level = 0.0
        self.episode = None        # a door use going on right now
        self.at_door = 0           # people at the door in the last check
        self.recent = deque(maxlen=4)   # looks at people near the door, last seconds


def _overlaps(box, rect):
    x1, y1, x2, y2 = rect
    return box[0] < x2 and box[2] > x1 and box[1] < y2 and box[3] > y1


class WashroomVisits(Detector):
    key = "washroom"
    title = "Washroom door"
    groups = (config.WASHROOM,)

    CHANGE = 25                # grey-level change that counts
    ACTIVE_SHARE = 0.06        # share of the zone that must change: door moving
    MOVE_SHARE = 0.01          # share of the area near the door: somebody walking
    QUIET = 4.0                # door still this long, nobody at it: the use is over
    MAX_USE = 120.0            # one record never covers more than this
    RELEARN = 180.0            # zone changed this long: that is the new normal
    LOOK_BACK = 3.0            # a person seen near the door this soon before counts
    DOOR_GROW = 0.25           # "at the door" = the zone grown by this much each side
    NEAR_GROW = 0.8            # "near the door"

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
            return bool(st.mask is not None)
        st.zone, st.size, st.bg, st.prev_near = zone, (w, h), None, None
        st.mask = st.rect = st.door_rect = st.near_rect = None
        if len(zone) < 3:
            return False
        pts = np.array([[x * w, y * h] for x, y in zone], np.float32)
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        # A thin zone (the edge of a door) still gets a usable box around it.
        zw, zh = max(x2 - x1, 0.04 * w), max(y2 - y1, 0.04 * h)

        def grown(f):
            return (int(max(0, x1 - zw * f)), int(max(0, y1 - zh * f)),
                    int(min(w, x2 + zw * f)), int(min(h, y2 + zh * f)))

        st.rect = (int(max(0, x1)), int(max(0, y1)), int(min(w, x2 + 1)), int(min(h, y2 + 1)))
        st.door_rect, st.near_rect = grown(self.DOOR_GROW), grown(self.NEAR_GROW)
        rx1, ry1, rx2, ry2 = st.rect
        sw = 64
        sh = max(8, int(sw * (ry2 - ry1) / max(1, rx2 - rx1)))
        sh = min(sh, 256)
        mask = np.zeros((sh, sw), np.uint8)
        scaled = (pts - [rx1, ry1]) * [sw / max(1, rx2 - rx1), sh / max(1, ry2 - ry1)]
        cv2.fillPoly(mask, [scaled.astype(np.int32)], 255)
        st.mask = mask > 0
        return True

    @staticmethod
    def _grey(frame, rect, size):
        x1, y1, x2, y2 = rect
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        small = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)

    # ---------- every decoded frame (camera threads) ----------

    def on_frame(self, camera, frame, ts):
        h, w = frame.shape[:2]
        with self._lock:
            st = self._st(camera)
            if not self._fit(st, camera, w, h):
                return
            cur = self._grey(frame, st.rect, (st.mask.shape[1], st.mask.shape[0]))
            near = self._grey(frame, st.near_rect, (80, 60))
            if cur is None:
                return
            if st.bg is None or st.relearn:
                st.bg, st.relearn, st.busy_since = cur, False, 0.0
            diff = np.abs(cur - st.bg) > self.CHANGE
            st.level = float(diff[st.mask].mean()) if st.mask.any() else 0.0
            if st.level >= self.ACTIVE_SHARE:
                st.active_at = ts
                st.busy_since = st.busy_since or ts
                if ts - st.busy_since > self.RELEARN:      # door left open, a bag put down
                    st.relearn = True
            else:
                st.busy_since = 0.0
                st.bg = st.bg * 0.97 + cur * 0.03            # light changes slowly
            if near is not None:
                if st.prev_near is not None and \
                        float((np.abs(near - st.prev_near) > self.CHANGE).mean()) >= self.MOVE_SHARE:
                    st.moving_at = ts
                st.prev_near = near
            ep = st.episode
            if ep is None and st.level >= self.ACTIVE_SHARE:
                st.episode = self._begin(st, ts, frame, "door")
            elif ep is not None:
                if len(ep["frames"]) < 2 and ts - ep["start"] >= 1.0:
                    ep["frames"].append((ts, frame.copy()))      # the door a moment later
                if ts - ep["start"] > self.MAX_USE + 60:
                    st.episode = None                            # no checks came: give up

    def _begin(self, st, ts, frame, trigger):
        recent = [c for c in st.recent if ts - c["ts"] < self.LOOK_BACK]
        return {"start": ts, "trigger": trigger, "people": 0, "cands": recent,
                "frames": [(ts, frame.copy())]}

    def schedule(self, camera):
        """(seconds between model checks, rank) for the pipeline."""
        now = time.time()
        with self._lock:
            st = self._state.get(camera)
            if st is None or st.mask is None:
                return 60.0, 3
            if st.episode is not None or now - st.active_at < self.QUIET:
                return 0.5, 1
            if now - st.moving_at < 2.0:
                return 1.0, 1
        return 60.0, 3

    # ---------- after a model check (pipeline worker) ----------

    def process(self, camera, frame, ts, seen):
        h, w = frame.shape[:2]
        done = None
        with self._lock:
            st = self._st(camera)
            if not self._fit(st, camera, w, h):
                return None
            at_door = [b for b in seen.people if _overlaps(b, st.door_rect)]
            near = [b for b in seen.people if _overlaps(b, st.near_rect)]
            st.at_door = len(at_door)
            look = None
            if near:
                pick = at_door or near
                best = max(pick, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]) * b[4])
                look = {"ts": ts, "frame": frame.copy(), "box": best,
                        "others": [b for b in seen.people if b is not best],
                        "score": b_area(best) * best[4] * (2 if at_door else 1)}
                st.recent.append(look)
            ep = st.episode
            if ep is None and at_door:
                ep = st.episode = self._begin(st, ts, frame, "person")
            if ep is not None:
                ep["people"] = max(ep["people"], len(at_door))
                if look is not None and all(c is not look for c in ep["cands"]):
                    ep["cands"].append(look)
                    # Keep the best looks, spread in time: one per second at most.
                    ep["cands"] = sorted(ep["cands"], key=lambda c: -c["score"])
                    kept = []
                    for c in ep["cands"]:
                        if all(abs(c["ts"] - k["ts"]) >= 1.0 for k in kept):
                            kept.append(c)
                    ep["cands"] = kept[:3]
                over = ts - st.active_at >= self.QUIET and not at_door
                if over or ts - ep["start"] > self.MAX_USE:
                    if not over:
                        st.relearn = True        # door stays open: that is normal now
                    done = (ep, ts)
                    st.episode = None
        if done:
            self._log(camera, *done)
        return None

    def _log(self, camera, ep, end):
        zone = (config.camera(camera) or {}).get("zone") or []
        shots = []
        for c in sorted(ep["cands"], key=lambda c: c["ts"]):
            shots.append(photos.save(camera, "washroom", c["frame"], [c["box"]], c["others"],
                                     zone=zone, colour=BLUE, ts=datetime.fromtimestamp(c["ts"])))
        if not shots:                    # nobody seen: the door, as it moved
            for fts, f in ep["frames"]:
                shots.append(photos.save(camera, "washroom", f, [], [], zone=zone,
                                         colour=BLUE, ts=datetime.fromtimestamp(fts)))
        people = ep["people"] or (1 if ep["cands"] else 0)
        meta = {"photos": shots, "people": people, "trigger": ep["trigger"],
                "at": datetime.fromtimestamp(ep["start"]).isoformat(timespec="seconds"),
                "end": datetime.fromtimestamp(end).isoformat(timespec="seconds"),
                "seconds": int(end - ep["start"])}
        label = ("{} at the door".format("1 person" if people == 1 else "{} people".format(people))
                 if people else "door moved, nobody seen")
        self.emit(camera, "washroom_visit", people, label, meta)

    # ---------- read-outs ----------

    def overlay(self, camera, frame):
        with self._lock:
            st = self._state.get(camera)
            zone = list(st.zone) if st else []
            active = bool(st and st.episode is not None)
            rect = st.door_rect if st else None
        if zone:
            photos.draw_zone(frame, zone, RED if active else BLUE)
        if rect and active:
            cv2.rectangle(frame, rect[:2], rect[2:], RED, 1)
        return frame

    def caption(self, camera):
        info = self.info(camera)
        if not info["zone"]:
            return ("washroom: draw the door zone", False)
        if info["in_use"]:
            return ("somebody at the washroom door", False)
        return ("door moving" if info["door"] else "washroom door quiet", False)

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
            ep = st.episode
            return {"zone": st.mask is not None,
                    "in_use": bool(ep is not None and st.at_door),
                    "since": ep["start"] if ep else 0,
                    "door": ep is not None or now - st.active_at < self.QUIET,
                    "inside": st.at_door if ep is not None else 0}

    def forget(self, camera):
        with self._lock:
            self._state.pop(camera, None)


def b_area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


detector = WashroomVisits()
pipeline.add(detector)
