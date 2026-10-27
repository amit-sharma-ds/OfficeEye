"""Night watch: after the office has closed, the counting cameras and the
washroom-door camera watch for anybody in the building.

Three things are recorded, each with photos:

1. A person - the model sees somebody in two checks a few seconds apart.
   One record per visit: when it started, when the person was last seen,
   and up to four photos.
2. Movement - the picture changes where nothing should move, even when the
   model cannot make out a person (a dark corner, an infrared picture). At
   most one record per camera every few minutes, with the moving part marked.
3. Lights - the whole picture suddenly gets much brighter or darker: a light
   was switched on or off.

Most IP cameras switch to infrared (black and white) when the lights go off,
so people stay visible; a camera without infrared sees nothing in the dark.
The hours (they run past midnight) and the sensitivity are set on the Night
vision page. Areas marked "ignore" on a camera (a TV, a fan) never count as
movement.
"""

import json
import logging
import time
from datetime import datetime, timedelta

import cv2
import numpy as np

import config
import events
import photos
from detection import Detector, _iso, _label, pipeline

log = logging.getLogger(__name__)

NIGHT = (230, 90, 60)          # BGR: the night-watch blue


def night_state(rules, office, now):
    """Is the night watch on at this moment? The night runs past midnight:
    20:00 to 08:00 means 20:00-23:59 and 00:00-08:00."""
    if not rules.get("enabled", True):
        return {"state": "off", "text": "Night watch is switched off"}
    start = rules.get("watch_from") or office.get("close") or "20:00"
    end = rules.get("watch_to") or office.get("open") or "08:00"
    hm = now.strftime("%H:%M")
    on = (start <= hm or hm < end) if start > end else (start <= hm < end)
    if on:
        return {"state": "on", "text": "Watching until {}".format(end)}
    return {"state": "off", "text": "Starts at {} (until {} next morning)".format(start, end)}


def night_of(rules, office, now):
    """The date of the watch this moment belongs to: after midnight in a
    20:00-08:00 watch, the evening before. Kept with every record, so that
    changing the hours later never moves (or hides) what was recorded."""
    start = rules.get("watch_from") or office.get("close") or "20:00"
    end = rules.get("watch_to") or office.get("open") or "08:00"
    back = start > end and now.strftime("%H:%M") < end
    return (now - timedelta(days=1 if back else 0)).strftime("%Y-%m-%d")


class NightWatch(Detector):
    key = "night"
    title = "Night watch"
    groups = (config.COUNT, config.WASHROOM)

    SMALL_W = 160              # the picture compared for movement, pixels across
    LEVEL = 25                 # grey-level change that counts as movement
    SHARE = {"low": 0.02, "medium": 0.01, "high": 0.005}   # of the picture
    LIGHT_JUMP = 35.0          # average brightness change: a light went on or off
    LIGHT_EVERY = 60.0         # one lights record per camera at most this often (s)
    CONFIRM = 2                # a person seen in this many checks ...
    CONFIRM_WITHIN = 10.0      # ... this close together starts a visit
    GAP = 30.0                 # nobody seen this long: the visit is over
    PHOTO_EVERY = 5.0
    MAX_PHOTOS = 4
    BG_TIME = 180.0            # how slowly the empty room is learnt (s)
    STILL_MATCH = 0.9          # a weak "person" this like the room is a thing in it
    STILL_SCORE = 0.9          # ... unless the model is this sure it is a person

    def __init__(self):
        super().__init__()
        self._state = {}
        self._rules, self._rules_at = ({}, {}), 0.0

    def rules(self):
        """(night rules, office hours) - read again every few seconds."""
        if time.time() - self._rules_at > 5:
            self._rules, self._rules_at = (config.get("night"), config.get("office")), time.time()
        return self._rules

    def _small(self, frame):
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h = max(1, int(grey.shape[0] * self.SMALL_W / grey.shape[1]))
        return cv2.GaussianBlur(cv2.resize(grey, (self.SMALL_W, h)), (5, 5), 0)

    def _ignore_mask(self, camera, shape, st):
        """Where movement is not counted: the camera's ignore areas."""
        areas = (config.camera(camera) or {}).get("ignore") or []
        key = (json.dumps(areas), shape)
        if st.get("ignore_key") != key:
            mask = None
            if areas:
                h, w = shape
                mask = np.zeros(shape, np.uint8)
                for poly in areas:
                    if len(poly) >= 3:
                        pts = np.array([[x * w, y * h] for x, y in poly], dtype=np.int32)
                        cv2.fillPoly(mask, [pts], 1)
            st["ignore_key"], st["ignore"] = key, mask
        return st["ignore"]

    @staticmethod
    def _blank():
        return {"prev": None, "bright": None, "hits": [], "visit": None,
                "motion_at": 0.0, "light_at": 0.0, "people": [], "bg": None, "bg_at": 0.0}

    def _part_of_room(self, box, small, bg, shape):
        """A weak "person" that looks just like the learnt empty room there -
        a coat on a chair, a cloth on a sofa - is a thing, not a visitor."""
        if bg is None or box[4] >= self.STILL_SCORE:
            return False
        s = small.shape[1] / float(shape[1])
        x1, y1, x2, y2 = (int(round(v * s)) for v in box[:4])
        a = small[max(0, y1):y2, max(0, x1):x2].astype(np.float32)
        b = bg[max(0, y1):y2, max(0, x1):x2]
        if a.shape[0] < 6 or a.shape[1] < 6:
            return False
        a, b = a - a.mean(), b - b.mean()
        d = float(np.sqrt((a * a).sum() * (b * b).sum()))
        return d > 1e-6 and float((a * b).sum()) / d >= self.STILL_MATCH

    def process(self, camera, frame, ts, seen):
        rules, office = self.rules()
        now = datetime.fromtimestamp(ts)
        watching = (night_state(rules, office, now)["state"] == "on" and
                    camera not in (rules.get("cameras_off") or []))   # switched off for the night
        small = self._small(frame)
        bright = float(small.mean())
        with self._lock:
            st = self._state.setdefault(camera, self._blank())
            prev, before, bg = st["prev"], st["bright"], st["bg"]
            st["prev"], st["bright"] = small, bright
            # Learn the empty room slowly - somebody walking through is not learnt.
            if st["bg"] is None or st["bg"].shape != small.shape:
                st["bg"] = small.astype(np.float32)
            else:
                a = min(1.0, max(0.0, ts - st["bg_at"]) / self.BG_TIME)
                st["bg"] = st["bg"] * (1.0 - a) + small * a
            st["bg_at"] = ts
        ends = []
        if not watching:
            with self._lock:
                st["people"], st["hits"] = [], []
                if st["visit"] is not None:
                    ends.append(st["visit"])
                    st["visit"] = None
            self._finish(ends)
            return None

        people = [b for b in seen.people if not self._part_of_room(b, small, bg, frame.shape)]
        light = None
        if before is not None and abs(bright - before) >= self.LIGHT_JUMP:
            light = "lights on" if bright > before else "lights off"
        moving = None
        if light is None and prev is not None and prev.shape == small.shape:
            changed = cv2.absdiff(prev, small) > self.LEVEL
            with self._lock:
                ignore = self._ignore_mask(camera, small.shape, st)
            if ignore is not None:
                changed &= ignore == 0
            share = float(np.count_nonzero(changed)) / changed.size
            if share >= self.SHARE.get(rules.get("sensitivity"), self.SHARE["medium"]):
                ys, xs = np.nonzero(changed)
                s = frame.shape[1] / float(small.shape[1])
                moving = (int(xs.min() * s), int(ys.min() * s),
                          int((xs.max() + 1) * s), int((ys.max() + 1) * s), 1.0)

        shots, new_visit, records = [], False, []
        with self._lock:
            st["people"] = people
            if people:
                st["hits"] = [t for t in st["hits"] if ts - t <= self.CONFIRM_WITHIN] + [ts]
            v = st["visit"]
            if people and (v is not None or len(st["hits"]) >= self.CONFIRM):
                if v is None:
                    v = st["visit"] = {"event": None, "started": ts, "last": ts,
                                       "photos": [], "photo_at": 0.0, "peak": len(people)}
                    new_visit = True
                v["last"], v["peak"] = ts, max(v["peak"], len(people))
                if len(v["photos"]) < self.MAX_PHOTOS and ts - v["photo_at"] >= self.PHOTO_EVERY:
                    v["photo_at"] = ts
                    shots.append("visit")
            elif v is not None and ts - v["last"] > self.GAP:
                ends.append(v)
                st["visit"] = None
            if light and ts - st["light_at"] >= self.LIGHT_EVERY:
                st["light_at"] = ts
                records.append((light, []))
            every = max(1.0, float(rules.get("motion_every", 5))) * 60
            if moving and not people and st["visit"] is None and ts - st["motion_at"] >= every:
                st["motion_at"] = ts
                records.append(("movement", [moving]))
            visit = st["visit"]

        if shots and visit is not None:
            shot = photos.save(camera, "night", frame, people, [], colour=NIGHT, ts=now)
            if shot:
                with self._lock:
                    visit["photos"].append(shot)
                    all_shots = list(visit["photos"])
            else:
                all_shots = list(visit["photos"])
            n = len(people)
            if new_visit:
                eid = self.emit(camera, "night_activity", n, "person",
                                {"type": "person", "people": n, "started": _iso(ts),
                                 "last_seen": _iso(ts), "seconds": 0, "photos": all_shots,
                                 "night": night_of(rules, office, now)})
                with self._lock:
                    visit["event"] = eid
            elif visit["event"]:
                events.update_meta(visit["event"], {"photos": all_shots, "people": visit["peak"],
                                                    "last_seen": _iso(ts),
                                                    "seconds": int(ts - visit["started"])})
        elif visit is not None and visit["event"] and not new_visit:
            events.update_meta(visit["event"], {"last_seen": _iso(visit["last"]),
                                                "seconds": int(visit["last"] - visit["started"])})
        for kind, targets in records:
            shot = photos.save(camera, "night", frame, targets, people, colour=NIGHT, ts=now)
            self.emit(camera, "night_activity", 0, kind,
                      {"type": kind, "started": _iso(ts), "photos": [shot] if shot else [],
                       "night": night_of(rules, office, now)})
        self._finish(ends)
        return None

    @staticmethod
    def _finish(ends):
        """Visits that are over get their end time."""
        for v in ends:
            if v.get("event"):
                events.update_meta(v["event"], {"ended": _iso(v["last"]), "last_seen": _iso(v["last"]),
                                                "seconds": int(v["last"] - v["started"]),
                                                "people": v["peak"]})

    def overlay(self, camera, frame):
        with self._lock:
            st = self._state.get(camera)
            if not st or st["visit"] is None:
                return frame
            people = list(st["people"])
        for x1, y1, x2, y2, _ in people:
            cv2.rectangle(frame, (x1, y1), (x2, y2), NIGHT, 2)
            _label(frame, "night: person", x1, y1, NIGHT)
        return frame

    def caption(self, camera):
        with self._lock:
            st = self._state.get(camera)
            if st and st["visit"] is not None:
                return ("NIGHT - somebody is here", True)
        return None

    def busy(self, camera):
        """Somebody seen at night: keep looking, so the visit is followed."""
        with self._lock:
            st = self._state.get(camera)
            return bool(st and (st["visit"] is not None or
                                any(time.time() - t <= self.CONFIRM_WITHIN for t in st["hits"])))

    def forget(self, camera):
        with self._lock:
            self._state.pop(camera, None)


detector = NightWatch()
pipeline.add(detector)
