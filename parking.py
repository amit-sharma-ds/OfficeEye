"""Parking: the cars and bikes on the cameras that look at the parking.

No model of its own - the counting cameras' check looks for vehicles in the
same pass as people (detection.Vision), so the parking costs nothing more.
On the cameras ticked under Parking -> Rules & cameras:

1. Parked, not passing: a vehicle seen at one spot (its box hardly moves)
   for the minutes set is parked. One that drives past, turns or looks for
   a place never stands still that long, so it is never recorded.
2. One record per vehicle per stay: when it came, kept up to date while it
   stands there, when it left and how long it stood.
3. Photos only of what happens - never again and again of a vehicle just
   standing there:
   - came: the moment it stopped at its spot (a bike's rider is still on
     it), with a close-up for the number plate;
   - getting out: nobody can be seen inside a car or truck as it parks, so
     the first time somebody shows up next to it in the minutes after it
     stopped, a photo with a close-up of them;
   - gone: the moment it was first missing at its spot.
   A vehicle already there when the watch began has no photo of its coming.
4. Missed now and then (somebody standing in front of it, the evening
   light) is not gone: only after LEAVE_CHECKS checks and LEAVE_AFTER
   seconds without it. One "gone" that is back at the same spot, looking
   the same, within REOPEN seconds never left: its record is opened again
   and the photo of it going dropped - no new record, no new photos.
5. After a restart the vehicles still standing take up their own records
   again rather than getting new ones; records whose vehicle went meanwhile
   get the time it was last seen as their end.

Marked on a record: "after hours" - it came while the office was closed;
"was there" - it already stood there when the watch began, so it came
before its first time shows.
"""

import logging
import time
from collections import Counter
from datetime import datetime, timedelta

import cv2
import numpy as np

import config
import events
import photos
from detection import Detector, _iou, _iso, _label, _overlap, pipeline

log = logging.getLogger(__name__)

TEAL = (170, 160, 20)          # BGR: the parking colour
KINDS = {"car": "car", "truck": "car", "bus": "car", "motorcycle": "bike", "bicycle": "cycle"}


def office_open(office, when):
    """Was the office open at this moment, by its hours on Settings?"""
    hm = when.strftime("%H:%M")
    return (office.get("open") or "09:30") <= hm < (office.get("close") or "18:30")


def _short(sec):
    sec = int(max(0, sec))
    if sec < 3600:
        return "{} min".format(sec // 60)
    return "{} h {:02d} min".format(sec // 3600, sec % 3600 // 60)


class Parking(Detector):
    key = "parking"
    title = "Parking"
    groups = (config.COUNT,)

    SAME_SPOT = 0.45           # box overlap with the last check: the same vehicle
    STILL = 0.5                # ... and with its first box: it has not moved on
    LEAVE_CHECKS = 3           # a parked vehicle missed this many checks in a row ...
    LEAVE_AFTER = 180.0        # ... and for this long (s) has left
    FORGET_AFTER = 60.0        # one only passing by is forgotten after 2 checks and this long unseen
    ONE_VEHICLE = 0.6          # two boxes of two types overlapping this much: one vehicle (car + truck)
    PART_OF = 0.8              # a car box this much inside a surer car box is part of it
    MIN_SIDE = 12              # pixels: anything smaller is too far off to tell
    WAS_THERE = 90.0           # first seen this soon after the watch began: it was already there
    RESUME = 600.0             # after a restart, open records are taken up this long
    SAVE_EVERY = 60.0          # "last seen" written at most this often
    SNAP_QUALITY = 90          # a moment kept for a photo, until it is known to be needed
    OUT_WINDOW = 300.0         # people getting out of a car that came are looked for this long (s)
    NEAR = 0.5                 # ... within this share of the car's size around it
    REOPEN = 600.0             # back at the same spot this soon after "leaving": it never left
    SAME_LOOK = 0.7            # ... if it looks the same (grey picture match, -1 to 1)

    def __init__(self):
        super().__init__()
        self._state = {}
        self._rules, self._rules_at = ({}, {}), 0.0
        self._shot_n = 0

    def rules(self):
        """(parking rules, office hours) - read again every few seconds."""
        if time.time() - self._rules_at > 5:
            self._rules, self._rules_at = (config.get("parking"), config.get("office")), time.time()
        return self._rules

    @classmethod
    def _vehicles(cls, seen):
        """Vehicles big enough to tell, each one once. The model already keeps
        one box per vehicle of each type, so bikes standing close in a row all
        count; only a vehicle found as two types (a car also taken for a
        truck) and a car's windscreen found as a car of its own are dropped.
        A bike in front of a car is a vehicle of its own."""
        out = []
        for v in sorted(seen.vehicles, key=lambda b: -b[4]):
            if v[2] - v[0] < cls.MIN_SIDE or v[3] - v[1] < cls.MIN_SIDE:
                continue
            car = KINDS.get(v[5]) == "car"
            if any((k[5] != v[5] and _iou(v, k) >= cls.ONE_VEHICLE) or
                   (car and KINDS.get(k[5]) == "car" and _overlap(v, k)[1] >= cls.PART_OF)
                   for k in out):
                continue
            out.append(v)
        return out

    @staticmethod
    def _track(v, ts):
        # came / gone / out: a moment kept for a photo; out is "done" once taken
        return {"box": v, "first_box": v, "first": ts, "last": ts, "hits": 1, "misses": 0,
                "kinds": Counter([v[5]]), "event": None, "photos": [], "saved": 0.0,
                "was_there": False, "came": None, "gone": None, "out": None, "look": None}

    @staticmethod
    def _look(frame, box):
        """A tiny grey picture of a vehicle, to tell it from another one at
        the same spot later."""
        x1, y1, x2, y2 = (int(v) for v in box[:4])
        crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if crop.size == 0:
            return None
        g = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (16, 16),
                       interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        g -= g.mean()
        n = float(np.linalg.norm(g))
        return g / n if n > 1e-6 else None

    @classmethod
    def _near(cls, car, people):
        """People next to a car - getting out of it, or just out."""
        x1, y1, x2, y2 = car[:4]
        mx, my = (x2 - x1) * cls.NEAR, (y2 - y1) * cls.NEAR
        return [p for p in people
                if p[0] < x2 + mx and p[2] > x1 - mx and p[1] < y2 + my and p[3] > y1 - my]

    def _stamp(self, at):
        """A photo's time. Its milliseconds count up, so two photos from one
        moment never get the same file name."""
        self._shot_n = (self._shot_n + 1) % 1000
        return datetime.fromtimestamp(at).replace(microsecond=self._shot_n * 1000)

    def _photo(self, camera, moment, what):
        """The photo of a moment kept earlier: the vehicle marked when it came,
        the people next to it when they got out, its spot when it went."""
        img = cv2.imdecode(np.frombuffer(moment["jpg"], np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        box, at = moment["box"], self._stamp(moment["at"])
        if what == "came":
            shot = photos.save(camera, "parking", img, [box], [], colour=TEAL, ts=at)
        elif what == "out":
            shot = photos.save(camera, "parking", img, moment["people"], [box], colour=TEAL, ts=at)
        else:
            h, w = img.shape[:2]
            x1, y1, x2, y2 = box[:4]
            spot = [[x1 / w, y1 / h], [x2 / w, y1 / h], [x2 / w, y2 / h], [x1 / w, y2 / h]]
            shot = photos.save(camera, "parking", img, [], [], zone=spot, colour=TEAL, ts=at)
        if shot:
            shot["what"] = what
        return shot

    @staticmethod
    def _open_records(camera):
        """This camera's records still open - the app was restarted meanwhile."""
        since = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
        rows = events.between(["parking"], since, "9999", camera=camera, limit=500)
        return [r for r in rows if not r["meta"].get("ended")]

    @staticmethod
    def _box_of(row, w, h):
        b = row["meta"].get("box") or []
        if len(b) != 4:
            return None
        return (int(b[0] * w), int(b[1] * h), int(b[2] * w), int(b[3] * h), 1.0)

    @staticmethod
    def _ts_of(iso):
        try:
            return datetime.fromisoformat(iso).timestamp()
        except (TypeError, ValueError):
            return None

    def process(self, camera, frame, ts, seen):
        rules, office = self.rules()
        if not (rules.get("enabled", True) and camera in (rules.get("cameras") or [])):
            with self._lock:
                st = self._state.pop(camera, None)
            if st:                      # taken off the parking cameras: close its records
                self._finish([t for t in st["tracks"] if t["event"] is not None])
            return None

        with self._lock:
            known = camera in self._state
        orphans = [] if known else self._open_records(camera)
        h, w = frame.shape[:2]
        need = max(0.5, float(rules.get("park_minutes", 2))) * 60
        new, keep, ends, lost, snaps, back, outs = [], [], [], [], [], [], []
        with self._lock:
            st = self._state.setdefault(camera, {"tracks": [], "since": ts, "orphans": orphans,
                                                 "ended": []})
            st["ended"] = [e for e in st["ended"] if ts - e["at"] <= self.REOPEN]
            tracks, found = st["tracks"], self._vehicles(seen)
            # Best overlaps first, over all vehicles at once: in a tight row of
            # bikes each one keeps its own place rather than a neighbour's.
            pairs = sorted(((_iou(t["box"], v), i, j) for i, t in enumerate(tracks)
                            for j, v in enumerate(found)), reverse=True)
            got_t, got_v = set(), set()
            for iou, i, j in pairs:
                if iou < self.SAME_SPOT:
                    break
                if i in got_t or j in got_v:
                    continue
                got_t.add(i)
                got_v.add(j)
                t, v = tracks[i], found[j]
                t.update(box=v, last=ts, misses=0, hits=t["hits"] + 1, gone=None)
                t["kinds"][v[5]] += 1
                if t["event"] is not None:
                    t["look"] = self._look(frame, v)
                if t["event"] is None and _iou(t["first_box"], v) < self.STILL:
                    t.update(first=ts, first_box=v, hits=1)      # still moving: the clock starts again
                    if not t["was_there"]:
                        snaps.append((t, "came"))                # it settles now: the driver gets off
            for i, t in enumerate(tracks):
                if i not in got_t:
                    t["misses"] += 1
                    if t["event"] is not None and t["misses"] == 1:
                        snaps.append((t, "gone"))                # first missing: being driven off?
            for j, v in enumerate(found):
                if j in got_v:
                    continue
                t = self._track(v, ts)
                t["was_there"] = ts - st["since"] <= self.WAS_THERE
                for r in st["orphans"]:          # a vehicle whose record is still open?
                    box = self._box_of(r, w, h)
                    if box and _iou(box, v) >= self.SAME_SPOT:
                        st["orphans"].remove(r)
                        m = r["meta"]
                        t.update(event=r["id"], photos=list(m.get("photos") or []), saved=ts,
                                 was_there=bool(m.get("was_there")), out="done",
                                 first=self._ts_of(m.get("started")) or ts, look=self._look(frame, v))
                        keep.append((t["event"], t["first"], ts))   # seen again just now
                        break
                if t["event"] is None:
                    # Back at a spot it "left" a few minutes ago, looking the same:
                    # it never left (missed a while) - its record goes on.
                    look = self._look(frame, v)
                    for e in st["ended"]:
                        if KINDS.get(e["kind"]) == KINDS.get(v[5]) and _iou(e["box"], v) >= self.SAME_SPOT \
                                and look is not None and e["look"] is not None \
                                and float(np.dot(look, e["look"])) >= self.SAME_LOOK:
                            st["ended"].remove(e)
                            t.update(event=e["event"], first=e["first"], photos=e["photos"],
                                     was_there=e["was_there"], saved=ts, hits=2, look=look, out="done")
                            t["kinds"] = e["kinds"]
                            back.append((t, e["gone"]))
                            break
                if t["event"] is None and not t["was_there"]:
                    snaps.append((t, "came"))
                tracks.append(t)
            if st["orphans"] and ts - st["since"] > self.RESUME:
                lost, st["orphans"] = st["orphans"], []
            for t in list(tracks):
                if t["event"] is not None:
                    if t["misses"] >= self.LEAVE_CHECKS and ts - t["last"] >= self.LEAVE_AFTER:
                        tracks.remove(t)
                        ends.append(t)
                elif t["misses"] >= 2 and ts - t["last"] >= self.FORGET_AFTER:
                    tracks.remove(t)                     # only drove past
            for t in tracks:
                if t["event"] is None and not t["misses"] and t["hits"] >= 2 and ts - t["first"] >= need:
                    t["event"] = "new"
                    new.append(t)
                elif t["event"] not in (None, "new") and not t["misses"] and ts - t["saved"] >= self.SAVE_EVERY:
                    t["saved"] = ts
                    keep.append((t["event"], t["first"], t["last"]))
                # A car or truck that came and stands still: the first people next
                # to it in the minutes after are the ones getting out.
                if t["out"] is None and not t["was_there"] and not t["misses"] and t["hits"] >= 2 \
                        and KINDS.get(t["kinds"].most_common(1)[0][0]) == "car" \
                        and ts - t["first"] <= self.OUT_WINDOW:
                    near = self._near(t["box"], seen.people)
                    if near:
                        t["out"] = "taking"
                        snaps.append((t, "out", near))
        if snaps:
            # The moment is kept (as a jpg, small) - a photo is made of it only
            # if the vehicle turns out to have parked, or to have gone.
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.SNAP_QUALITY])
            for t, what, *people in snaps:
                t[what] = {"jpg": buf.tobytes(), "at": ts, "box": t["box"],
                           "people": people[0] if people else []} if ok else None
                if what == "out":
                    outs.append(t)

        for t in new:
            box = t["box"]
            shots = [self._photo(camera, t["came"], "came") if t["came"] else None]
            if isinstance(t["out"], dict):          # got out before it counted as parked
                shots.append(self._photo(camera, t["out"], "out"))
                t["out"] = "done"
            t["came"] = None
            kind = t["kinds"].most_common(1)[0][0]
            meta = {"type": KINDS.get(kind, "car"), "kind": kind,
                    "started": _iso(t["first"]), "last_seen": _iso(t["last"]),
                    "seconds": int(t["last"] - t["first"]), "photos": [s for s in shots if s],
                    "box": [round(box[0] / w, 4), round(box[1] / h, 4),
                            round(box[2] / w, 4), round(box[3] / h, 4)],
                    "after_hours": not office_open(office, datetime.fromtimestamp(t["first"])),
                    "was_there": t["was_there"]}
            eid = self.emit(camera, "parking", 1, meta["type"], meta)
            with self._lock:
                t["event"], t["photos"], t["saved"] = eid, meta["photos"], ts
        for t in outs:                              # got out of a car parked already
            if t["event"] not in (None, "new") and isinstance(t["out"], dict):
                shot = self._photo(camera, t["out"], "out")
                t["out"] = "done"
                if shot:
                    t["photos"] = t["photos"] + [shot]
                    events.update_meta(t["event"], {"photos": t["photos"]})
        for eid, first, last in keep:
            events.update_meta(eid, {"last_seen": _iso(last), "seconds": int(last - first)})
        for t, gone in back:                        # never left: open again, drop its "gone" photo
            events.update_meta(t["event"], {"ended": None, "photos": t["photos"],
                                            "last_seen": _iso(ts), "seconds": int(ts - t["first"])})
            if gone:
                photos.remove([gone])
        for t in ends:
            # The moment it was first missing at its spot.
            before = list(t["photos"])
            shot = self._photo(camera, t["gone"], "gone") if t["gone"] else None
            t["gone"] = None
            t["photos"] = before + ([shot] if shot else [])
            if t["event"] not in (None, "new"):
                with self._lock:            # kept a while, in case it shows up there again
                    st["ended"].append({"event": t["event"], "box": t["box"], "at": ts,
                                        "kind": t["kinds"].most_common(1)[0][0], "kinds": t["kinds"],
                                        "look": t["look"], "first": t["first"], "photos": before,
                                        "gone": shot, "was_there": t["was_there"]})
        self._finish(ends)
        for r in lost:                 # gone while the app was not running
            m = r["meta"]
            events.update_meta(r["id"], {"ended": m.get("last_seen") or m.get("started") or r["ts"]})
        return None

    @staticmethod
    def _finish(ends):
        """Vehicles that left get their end time (and the photo of them going)."""
        for t in ends:
            if t["event"] in (None, "new"):
                continue
            patch = {"ended": _iso(t["last"]), "last_seen": _iso(t["last"]),
                     "seconds": int(t["last"] - t["first"])}
            if t["photos"]:
                patch["photos"] = t["photos"]
            events.update_meta(t["event"], patch)

    def overlay(self, camera, frame):
        now = time.time()
        with self._lock:
            st = self._state.get(camera)
            if not st:
                return frame
            parked = [(t["box"], t["kinds"].most_common(1)[0][0], t["first"])
                      for t in st["tracks"] if t["event"] is not None]
        for (x1, y1, x2, y2, *_), kind, first in parked:
            cv2.rectangle(frame, (x1, y1), (x2, y2), TEAL, 2)
            _label(frame, "{} {}".format(KINDS.get(kind, kind), _short(now - first)), x1, y1, TEAL)
        return frame

    def caption(self, camera):
        n = self.parked(camera)
        return ("{} parked".format(n), False) if n else None

    def waiting(self, camera):
        """A vehicle not parked yet, or a parked one not seen just now: the
        camera's routine re-check (every 30 s or so) settles it, even on a
        busy PC where quiet cameras are otherwise looked at last."""
        with self._lock:
            st = self._state.get(camera)
            return bool(st and any(t["event"] is None or t["misses"] for t in st["tracks"]))

    def parked(self, camera):
        with self._lock:
            st = self._state.get(camera)
            return sum(1 for t in st["tracks"] if t["event"] is not None) if st else 0

    def forget(self, camera):
        with self._lock:
            self._state.pop(camera, None)


detector = Parking()
pipeline.add(detector)
