"""Night watch: after the office has closed, the counting cameras and the
washroom-door camera are guarded the way a restricted area is, all night.

While the watch is on, a camera whose picture changes is looked at before
every other counting camera (only restricted cameras go first), and while
somebody is there it is looked at again and again.

What is recorded, with photos:

1. A person - one the model is sure of starts a record at once, with a
   photo of that very moment; a weaker one needs a second sighting within
   a few seconds. Something moving where the model saw nobody (a dark
   corner, a far-off figure) is looked at again closer: that part of the
   picture alone, brightened when dark, at a lower cutoff, and kept only
   when the pose model finds a body in it. One record per visit: when it
   started, when the person was last seen, and up to four photos.

Only when switched on under Rules & timing ("movement"):

2. Movement - something bigger than a speck moves in two checks in a row and
   no person is found in it. The camera's own exposure changes, the switch
   to and from infrared, and moving parts that are mostly glaring white (an
   insect lit up right at the lens, car headlights) are not movement. At
   most one record per camera every few minutes, with the moving part marked.
3. Lights - the whole picture suddenly gets much brighter or darker: a light
   was switched on or off.

Most IP cameras switch to infrared (black and white) when the lights go off,
so people stay visible; a camera without infrared sees nothing in the dark -
the page shows which ones. Photos of a dark picture are brightened. The
hours (they run past midnight) and the sensitivity are set on the Night
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
AMBER = (0, 170, 255)


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

    SURE = 0.5                 # a person this certain starts a record at once
    CONFIRM = 2                # a weaker one must be seen this many times ...
    CONFIRM_WITHIN = 15.0      # ... this close together (s)
    GAP = 30.0                 # nobody seen this long: the visit is over
    PHOTO_EVERY = 4.0
    MAX_PHOTOS = 4
    BG_TIME = 180.0            # how slowly the empty room is learnt (s)
    STILL_MATCH = 0.9          # a weak "person" this like the room is a thing in it
    STILL_SCORE = 0.9          # ... unless the model is this sure it is a person

    SMALL_W = 160              # the picture compared for movement, pixels across
    LEVEL = 25                 # grey-level change that counts as movement
    SHARE = {"low": 0.02, "medium": 0.01, "high": 0.005}   # of the picture, in one moving part
    GLARE = 225                # a moving part mostly this bright is an insect at the lens or headlights
    GLARE_SHARE = 0.5
    MOVE_CONFIRM = 2           # movement in this many checks ...
    MOVE_WITHIN = 10.0         # ... this close together before it is recorded
    CLOSER_EVERY = 4.0         # the moving part is looked at closer at most this often (s)
    CLOSER_CONF = 0.25
    LIGHT_JUMP = 35.0          # average brightness change: a light went on or off
    LIGHT_EVERY = 60.0         # one lights record per camera at most this often (s)
    INFRARED_SAT = 12          # average colour below this: a black-and-white (infrared) picture
    DARK = 60                  # average brightness below this: photos are brightened

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

    def _infrared(self, frame):
        tiny = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
        return float(cv2.cvtColor(tiny, cv2.COLOR_BGR2HSV)[..., 1].mean()) < self.INFRARED_SAT

    @staticmethod
    def _brighten(frame):
        """A dark picture made readable: lifted shadows, then local contrast."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lift = np.array([255.0 * (i / 255.0) ** 0.6 for i in range(256)], np.uint8)
        lab[..., 0] = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(cv2.LUT(lab[..., 0], lift))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

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
        return {"prev": None, "bright": None, "ir": None, "hits": [], "visit": None,
                "motion_at": 0.0, "light_at": 0.0, "people": [], "bg": None, "bg_at": 0.0,
                "watching": False, "checked": 0.0, "moving": None, "moved_at": 0.0,
                "moves": [], "closer_at": 0.0, "known": None}

    def _learn(self, st, small, people, shape, ts):
        """Learn the empty room slowly - never where somebody is seen, so a
        person sitting still all evening does not become part of the room.
        Where nobody has ever been out of the way yet, the room is unknown."""
        s = small.shape[1] / float(shape[1])
        here = np.zeros(small.shape, bool)
        for x1, y1, x2, y2, _ in people:
            here[max(0, int(y1 * s)):int(y2 * s) + 1, max(0, int(x1 * s)):int(x2 * s) + 1] = True
        if st["bg"] is None or st["bg"].shape != small.shape:
            st["bg"], st["bg_at"], st["known"] = small.astype(np.float32), ts, ~here
            return
        a = min(1.0, max(0.0, ts - st["bg_at"]) / self.BG_TIME)
        fresh = ~here & ~st["known"]                  # seen empty for the first time: taken as is
        bg = np.where(here, st["bg"], np.where(fresh, small, st["bg"] * (1.0 - a) + small * a))
        st["bg"], st["bg_at"], st["known"] = bg.astype(np.float32), ts, st["known"] | ~here

    def _part_of_room(self, box, small, bg, shape, known=None):
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
        if known is not None and known[max(0, y1):y2, max(0, x1):x2].mean() < 0.8:
            return False                               # never seen empty there: count it
        a, b = a - a.mean(), b - b.mean()
        d = float(np.sqrt((a * a).sum() * (b * b).sum()))
        return d > 1e-6 and float((a * b).sum()) / d >= self.STILL_MATCH

    def _moving(self, prev, small, ignore, sensitivity, shape):
        """The biggest part of the picture that moved since the last check, as
        a box on the frame - or None. The whole picture getting brighter or
        darker (the camera's own exposure) is taken out first, specks are
        dropped, and a part that is mostly glaring white is not movement."""
        a = small.astype(np.int16) - int(small.mean())
        b = prev.astype(np.int16) - int(prev.mean())
        changed = (np.abs(a - b) > self.LEVEL).astype(np.uint8)
        if ignore is not None:
            changed[ignore > 0] = 0
        changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        if not changed.any():
            return None
        n, parts, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(changed, np.ones((5, 5), np.uint8)))
        need = self.SHARE.get(sensitivity, self.SHARE["medium"]) * changed.size
        glare = np.maximum(small, prev) >= self.GLARE
        best = None
        for i in range(1, n):
            part = (parts == i) & (changed > 0)
            size = int(np.count_nonzero(part))
            if size < need or np.count_nonzero(part & glare) >= self.GLARE_SHARE * size:
                continue
            if best is None or size > best[0]:
                best = (size,) + tuple(int(v) for v in stats[i][:4])
        if best is None:
            return None
        _, x, y, w, h = best
        s = shape[1] / float(small.shape[1])
        return (int(x * s), int(y * s), int((x + w) * s), int((y + h) * s), 1.0)

    def _closer(self, frame, box, dark):
        """People in the part of the picture around a moving box, looked at
        on their own (brightened when dark), so a small or dim figure the
        whole-picture check passed over is found."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box[:4]
        half = max(x2 - x1, y2 - y1, 0.4 * min(w, h)) * 0.8
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        ox, oy = int(max(0, cx - half)), int(max(0, cy - half))
        crop = frame[oy:int(min(h, cy + half)), ox:int(min(w, cx + half))]
        if crop.shape[0] < 32 or crop.shape[1] < 32:
            return []
        try:
            found = pipeline.look_closer(self._brighten(crop) if dark else crop, self.CLOSER_CONF)
        except Exception:                             # noqa: BLE001
            log.exception("night: closer look failed")
            return []
        return [(a + ox, b + oy, c + ox, d + oy, s) for a, b, c, d, s in found]

    def process(self, camera, frame, ts, seen):
        rules, office = self.rules()
        now = datetime.fromtimestamp(ts)
        watching = (night_state(rules, office, now)["state"] == "on" and
                    camera not in (rules.get("cameras_off") or []))   # switched off for the night
        small = self._small(frame)
        bright = float(small.mean())
        ir = self._infrared(frame)
        with self._lock:
            st = self._state.setdefault(camera, self._blank())
            prev, before, was_ir, bg, known = st["prev"], st["bright"], st["ir"], st["bg"], st["known"]
            st.update(prev=small, bright=bright, ir=ir, checked=ts, watching=watching)
            self._learn(st, small, seen.people, frame.shape, ts)
        ends = []
        if not watching:
            with self._lock:
                st["people"], st["hits"], st["moves"], st["moving"] = [], [], [], None
                if st["visit"] is not None:
                    ends.append(st["visit"])
                    st["visit"] = None
            self._finish(ends)
            return None

        dark = bright < self.DARK
        people = [b for b in seen.people if not self._part_of_room(b, small, bg, frame.shape, known)]
        moves = bool(rules.get("movement", False))       # movement and lights recorded too?
        switched = was_ir is not None and was_ir != ir    # went infrared, or back to colour
        light = None
        if moves and not switched and before is not None and abs(bright - before) >= self.LIGHT_JUMP:
            light = "lights on" if bright > before else "lights off"
        moving = None
        if not switched and light is None and prev is not None and prev.shape == small.shape:
            with self._lock:
                ignore = self._ignore_mask(camera, small.shape, st)
            moving = self._moving(prev, small, ignore, rules.get("sensitivity"), frame.shape)
        if moving and not people and ts - st["closer_at"] >= self.CLOSER_EVERY:
            with self._lock:
                st["closer_at"] = ts
            people = [b for b in self._closer(frame, moving, dark)
                      if not self._part_of_room(b, small, bg, frame.shape, known)]

        shots, new_visit, records = [], False, []
        with self._lock:
            st["people"] = people
            if moving:
                st["moving"], st["moved_at"] = moving, ts
                st["moves"] = [t for t in st["moves"] if ts - t <= self.MOVE_WITHIN] + [ts]
            if people:
                st["hits"] = [t for t in st["hits"] if ts - t <= self.CONFIRM_WITHIN] + [ts]
            v = st["visit"]
            sure = any(b[4] >= self.SURE for b in people)
            if people and (v is not None or sure or len(st["hits"]) >= self.CONFIRM):
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
            if (moves and moving and len(st["moves"]) >= self.MOVE_CONFIRM and not people and
                    st["visit"] is None and ts - st["motion_at"] >= every):
                st["motion_at"] = ts
                records.append(("movement", [moving]))
            visit = st["visit"]

        picture = self._brighten(frame) if dark and (shots or records) else frame
        if shots and visit is not None:
            shot = photos.save(camera, "night", picture, people, [], colour=NIGHT, ts=now)
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
            shot = photos.save(camera, "night", picture, targets, people, colour=NIGHT, ts=now)
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
            if not st or not st["watching"]:
                return frame
            people = list(st["people"]) if st["visit"] is not None else []
            moving = st["moving"] if time.time() - st["moved_at"] <= 3 and not people else None
        for x1, y1, x2, y2, score in people:
            cv2.rectangle(frame, (x1, y1), (x2, y2), NIGHT, 2)
            _label(frame, "night: person {:.0%}".format(score), x1, y1, NIGHT)
        if moving:
            cv2.rectangle(frame, moving[:2], moving[2:4], AMBER, 1)
            _label(frame, "movement", moving[0], moving[1], AMBER)
        return frame

    def caption(self, camera):
        with self._lock:
            st = self._state.get(camera)
            if st and st["visit"] is not None:
                return ("NIGHT - somebody is here", True)
        return None

    def busy(self, camera):
        """Somebody seen, or something moving, at night: keep looking."""
        now = time.time()
        with self._lock:
            st = self._state.get(camera)
            return bool(st and st["watching"] and (
                st["visit"] is not None or
                any(now - t <= self.CONFIRM_WITHIN for t in st["hits"]) or
                any(now - t <= self.MOVE_WITHIN for t in st["moves"])))

    def urgent(self, camera):
        """Guarded like a restricted area for as long as the watch is on."""
        with self._lock:
            st = self._state.get(camera)
            return bool(st and st["watching"])

    def cameras(self):
        """Each camera as the night watch last saw it, for the page."""
        now = time.time()
        with self._lock:
            return {cam: {"watching": st["watching"],
                          "checked_ago": round(now - st["checked"], 1) if st["checked"] else None,
                          "someone": st["visit"] is not None,
                          "people": len(st["people"]) if st["visit"] is not None else 0,
                          "moved_ago": round(now - st["moved_at"]) if st["moved_at"] else None,
                          "bright": round(st["bright"] or 0), "infrared": bool(st["ir"])}
                    for cam, st in self._state.items()}

    def forget(self, camera):
        with self._lock:
            self._state.pop(camera, None)


detector = NightWatch()
pipeline.add(detector)
