"""Detection pipeline.

Every camera hands its newest frame to the pipeline. One worker thread then
decides which cameras to look at next, runs the model once over a small
batch of frames, and passes what it saw to the detections registered below.

Why one worker and not one per camera: the model already uses every core,
so thirty threads queueing for it only fight. One worker that picks well gets
more done. It picks in this order:

1. restricted-area cameras - whenever anything moves there, and constantly
   while an alert is open, so it can tell when the area is clear again;
2. the camera somebody is watching live;
3. counting cameras whose picture changed since they were last checked;
4. quiet counting cameras, at least every `count_refresh` seconds, so a count
   never goes stale.

A picture that has not changed is not looked at again, so an empty corridor
costs nothing. On a machine with an NVIDIA GPU the same code runs there, with
bigger batches.

To add a detection:

    @register
    class HelmetDetector(Detector):
        key = "helmet"
        title = "Helmet check"
        groups = (config.COUNT,)          # which cameras it runs on

        def process(self, camera, frame, ts, seen):
            # seen.people / seen.phones: [(x1, y1, x2, y2, score), ...]
            return [{"kind": "no_helmet", "label": "gate", "meta": {"conf": 0.82}}]

Anything process() returns is written to the event log, and whatever
overlay() draws shows up on the live view.
"""

import json
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

import config
import events
import photos

log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REGISTRY = {}


def register(cls):
    REGISTRY[cls.key] = cls
    return cls


class Seen:
    """What the model found in one frame."""
    __slots__ = ("people", "phones", "vehicles", "objects")

    def __init__(self, people=None, phones=None, vehicles=None, objects=None):
        self.people = people or []      # [(x1, y1, x2, y2, score), ...]
        self.phones = phones or []
        self.vehicles = vehicles or []  # [(x1, y1, x2, y2, score, "car"), ...] - parking.py
        self.objects = objects or []    # everything else the model knows, same shape - search.py


class Detector:
    """Base class. process() runs on the pipeline's worker thread; overlay(),
    caption() and the read-outs are called from web requests, so guard any
    state they share with self._lock."""

    key = "base"
    title = "Detector"
    groups = (config.COUNT, config.RESTRICTED)

    def __init__(self):
        self._lock = threading.Lock()

    def process(self, camera, frame, ts, seen):
        """Return a list of events, or None. An event is a dict with
        kind / label / value / meta - all optional except kind."""
        return None

    def overlay(self, camera, frame):
        """Draw boxes for the live view."""
        return frame

    def caption(self, camera):
        """(text, is_alert) for the banner on the live view, or None."""
        return None

    def busy(self, camera):
        """True while something is going on at this camera that has to be
        looked at again soon, even if the picture barely changes."""
        return False

    def waiting(self, camera):
        """True while something here waits for a second look that the
        routine re-check (every count_refresh seconds) can give - so that
        re-check is not put behind every camera with movement."""
        return False

    def stats(self):
        return {}

    def forget(self, camera):
        """A camera was removed - drop whatever was kept for it."""

    @staticmethod
    def emit(camera, kind, value=0, label="", meta=None):
        """Log an event straight away and return its id, for detectors that
        need to add to it later (photos of an ongoing alert)."""
        return events.log(kind=kind, camera=camera, label=label, value=value,
                          meta=meta)


# ---------------------------------------------------------------- the model

class Vision:
    """The one YOLO model every camera shares. Finds people and phones in a
    single pass - filtering to two classes costs the same as one."""

    PERSON = 0     # COCO: person      - other models are looked up by name
    PHONE = 67     # COCO: cell phone
    VEHICLES = ("car", "truck", "bus", "motorcycle", "bicycle")   # for the parking
    VEHICLE_CONF = 0.25        # bikes close in a row score low; parking.py waits for a still one
    OBJECT_CONF = 0.3          # a bottle, a book, a laptop... for the Search page

    def __init__(self, cfg):
        import torch
        from ultralytics import YOLO          # imported late: slow to load

        want = cfg.get("device", "auto")
        gpu = want in ("auto", "cuda") and torch.cuda.is_available()
        self.device = "cuda:0" if gpu else "cpu"
        self.half = gpu
        self.batch = 16 if gpu else 4
        self.device_name = torch.cuda.get_device_name(0) if gpu else "CPU"
        self.model_name = cfg.get("model", "yolo11n.pt")
        path = self.model_name
        if not os.path.isabs(path):
            path = os.path.join(BASE_DIR, path)
        self.model = YOLO(path if os.path.exists(path) else self.model_name)
        # Models trained on other datasets (Objects365, our own) number their
        # classes differently: find person and phone by name.
        ids = {str(v).lower(): int(k) for k, v in self.model.names.items()}
        self.person = ids.get("person", self.PERSON)
        self.phone = ids.get("cell phone", ids.get("phone", self.PHONE))
        self.vehicles = {ids[n]: n for n in self.VEHICLES if n in ids}
        self.names = {int(k): str(v) for k, v in self.model.names.items()}
        self.configure(cfg)
        self.detect([np.zeros((480, 640, 3), np.uint8)])       # warm up

    def same_model(self, cfg):
        return (cfg.get("model", "yolo11n.pt") == self.model_name and
                cfg.get("device", "auto") in (("auto", "cuda") if self.half
                                              else ("auto", "cpu")))

    # Phones seen from a ceiling score far below people, so they get their own
    # cutoff (Phone use -> Rules & timing -> How sensitive). People keep theirs.
    PHONE_CONF = {"low": 0.30, "medium": 0.15, "high": 0.10}

    def configure(self, cfg):
        """Settings that do not need the model reloaded."""
        self.imgsz = int(cfg.get("imgsz", 480))
        self.conf = float(cfg.get("confidence", 0.4))
        self.want_phone = bool(cfg.get("phone", True))
        # Every object the model knows, in the same pass as people: the
        # parking needs vehicles, the Search page any of the 80 objects, and
        # leaving classes out saves no model time.
        self.classes = None
        sens = config.get("phone").get("sensitivity", "medium")
        self.phone_conf = self.PHONE_CONF.get(sens, self.PHONE_CONF["medium"])

    def detect(self, frames):
        extra = {"half": True} if self.half else {}
        results = self.model.predict(frames, imgsz=self.imgsz,
                                     conf=min(self.conf, self.phone_conf),
                                     classes=self.classes, device=self.device,
                                     verbose=False, **extra)
        out = []
        for r in results:
            seen = Seen()
            if r.boxes is not None and len(r.boxes):
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy()
                for (a, b, c, d), s, k in zip(xyxy, confs, clss):
                    box = (int(a), int(b), int(c), int(d), float(s))
                    if int(k) == self.person:
                        if s >= self.conf:          # the people cutoff, as before
                            seen.people.append(box)
                    elif int(k) in self.vehicles:
                        if s >= self.VEHICLE_CONF:
                            seen.vehicles.append(box + (self.vehicles[int(k)],))
                    elif int(k) == self.phone and self.want_phone:
                        if s >= self.phone_conf:
                            seen.phones.append(box)
                    elif s >= self.OBJECT_CONF:
                        seen.objects.append(box + (self.names.get(int(k), str(int(k))),))
            out.append(seen)
        return out


def _overlap(a, b):
    """(IoU, share of the smaller box that lies inside the other)."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    small = min(area_a, area_b)
    return (inter / union if union > 0 else 0.0,
            inter / small if small > 0 else 0.0)


def _same_person(a, b):
    iou, inside = _overlap(a, b)
    return iou >= 0.6 or inside >= 0.8


# ---------------------------------------------------------------- detections

GREEN = (80, 220, 80)      # BGR
AMBER = (0, 170, 255)
RED = (60, 60, 235)
GREY = (170, 170, 170)


def _label(frame, text, x, y, colour):
    cv2.putText(frame, text, (x, max(y - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, colour, 1, cv2.LINE_AA)


@register
class PeopleCount(Detector):
    """How many people a counting camera can see right now - not a doorway
    in/out counter, it describes this moment."""

    key = "people"
    title = "People count"
    groups = (config.COUNT,)
    MIN_LOG_GAP = 2.0       # seconds, so a flickering box cannot spam the log

    def __init__(self):
        super().__init__()
        self._state = {}

    def process(self, camera, frame, ts, seen):
        count = len(seen.people)
        with self._lock:
            st = self._state.setdefault(camera, {"boxes": [], "logged": None,
                                                 "logged_at": 0.0})
            st["boxes"] = list(seen.people)
            if st["logged"] == count or ts - st["logged_at"] < self.MIN_LOG_GAP:
                return None
            st["logged"], st["logged_at"] = count, ts
        return [{"kind": "person_count", "value": count,
                 "label": "{} in view".format(count)}]

    def overlay(self, camera, frame):
        with self._lock:
            boxes = list((self._state.get(camera) or {}).get("boxes", []))
        for x1, y1, x2, y2, score in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
            _label(frame, "{:.0%}".format(score), x1, y1, GREEN)
        return frame

    def caption(self, camera):
        with self._lock:
            n = len((self._state.get(camera) or {}).get("boxes", []))
        return ("{} person{}".format(n, "" if n == 1 else "s"), False)

    def forget(self, camera):
        with self._lock:
            self._state.pop(camera, None)


def _iso(ts):
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def phone_watch_state(rules, office, now):
    """Is phone use checked at this moment? Outside the watch hours and in a
    break it is not recorded. Times are "HH:MM" strings, same day."""
    hm = now.strftime("%H:%M")
    start = rules.get("watch_from") or office.get("open") or "00:00"
    end = rules.get("watch_to") or office.get("close") or "23:59"
    if not start <= hm < end:
        return {"state": "off", "text": "Not watching now - phone use is checked {} to {}".format(start, end)}
    for a, b in rules.get("breaks") or []:
        if a <= hm < b:
            return {"state": "break", "text": "Break until {} - phone use is not recorded".format(b)}
    return {"state": "on", "text": "Watching until {}".format(end)}


@register
class PhoneUse(Detector):
    """Who is *using* a phone, not just near one.

    A phone counts for a person when it lies well inside their outline - in
    the hands in front of them, at the face or the ear - not at its edge,
    where a phone lies on the desk next to them. Each person is followed from
    check to check, and the phone has to be seen with them again and again
    for the time set under Phone use -> Rules & timing: a hand resting on a
    phone or a quick look stays below it. Seen from a ceiling the phone can
    be anywhere on the body, so there is no "top half" rule.

    Outside the watch hours and in breaks nothing is recorded. When somebody
    starts counting, one photo is taken - at most one per camera every few
    minutes, and only once per person per use. A use that starts inside that
    gap gets its photo once the gap is over, if it is still going on."""

    key = "phone"
    title = "Phone use"
    groups = (config.COUNT,)
    INSIDE = 0.10           # phone centre at least this far inside the person's box
    MATCH_IOU = 0.3         # the same person as in the last check
    GAP = 20.0              # not seen with the phone this long: that use is over
    PENDING = 60.0          # look again soon while a use is building up, this long
    MIN_HITS = 2

    MODEL_INFO = os.path.join(BASE_DIR, "data", "phone", "model.json")
    PAD = 0.15              # cut-out margin, as in tools/phone_data.py

    def __init__(self):
        super().__init__()
        self._state = {}
        self._rules, self._rules_at = ({}, {}), 0.0
        self._cls, self._cls_cut, self._cls_mtime, self._cls_at = None, 0.5, None, 0.0

    @staticmethod
    def _blank():
        return {"tracks": [], "phones": [], "using": [], "on_phone": 0, "photo_at": 0.0}

    def _classifier(self):
        """The office-trained phone model (tools/train_phone.py), once it has
        passed its test - looked for once a minute, so no restart is needed."""
        if time.time() - self._cls_at < 60:
            return self._cls
        self._cls_at = time.time()
        try:
            mtime = os.path.getmtime(self.MODEL_INFO)
        except OSError:
            self._cls = None
            return None
        if mtime != self._cls_mtime:
            try:
                info = json.load(open(self.MODEL_INFO, encoding="utf-8"))
                from ultralytics import YOLO
                self._cls = YOLO(os.path.join(BASE_DIR, info["model"]), task="classify")
                self._cls_cut, self._cls_mtime = float(info["cutoff"]), mtime
                log.info("phone model in use: %s, cutoff %s", info["model"], info["cutoff"])
            except Exception:                             # noqa: BLE001
                log.exception("could not load the phone model")
                self._cls, self._cls_mtime = None, mtime
        return self._cls

    def _looks(self, frame, people):
        """{person index: phone probability} from the office-trained model."""
        cls = self._classifier()
        if cls is None or not people:
            return {}
        h, w = frame.shape[:2]
        crops, idx = [], []
        for i, (x1, y1, x2, y2, _) in enumerate(people):
            if y2 - y1 < 60:
                continue                            # too far away to judge
            px, py = (x2 - x1) * self.PAD, (y2 - y1) * self.PAD
            crop = frame[int(max(0, y1 - py)):int(min(h, y2 + py)), int(max(0, x1 - px)):int(min(w, x2 + px))]
            if crop.size:
                crops.append(crop)
                idx.append(i)
        if not crops:
            return {}
        out = {}
        for i, r in zip(idx, cls.predict(crops, imgsz=224, verbose=False)):
            names = {v: k for k, v in r.names.items()}
            out[i] = float(r.probs.data[names["phone"]])
        return out

    def rules(self):
        """(phone rules, office hours) - read again every few seconds."""
        if time.time() - self._rules_at > 5:
            self._rules, self._rules_at = (config.get("phone"), config.get("office")), time.time()
        return self._rules

    def _holder(self, phone, people):
        """Index of the person holding this phone, or None."""
        cx, cy = (phone[0] + phone[2]) / 2.0, (phone[1] + phone[3]) / 2.0
        best, best_area = None, None
        for i, (x1, y1, x2, y2, _) in enumerate(people):
            mx, my = (x2 - x1) * self.INSIDE, (y2 - y1) * self.INSIDE
            if not (x1 + mx <= cx <= x2 - mx and y1 + my <= cy <= y2 - my):
                continue                            # outside, or at the edge: on the desk
            area = (x2 - x1) * (y2 - y1)
            if best_area is None or area < best_area:
                best, best_area = i, area            # nearest (smallest) person wins
        return best

    def process(self, camera, frame, ts, seen):
        """Every use of a phone is its own record: written when it starts to
        count (with a photo when the camera's photo gap allows), kept up to
        date while it lasts, and given its end time when the phone is put
        away, the person leaves, or a break / the end of watch hours comes."""
        rules, office = self.rules()
        watching = phone_watch_state(rules, office, datetime.fromtimestamp(ts))["state"] == "on"
        min_s = float(rules.get("min_seconds", 10))
        need = self.MIN_HITS if min_s > 0 else 1
        holders = {}
        if watching:
            for ph in seen.phones:
                i = self._holder(ph, seen.people)
                if i is not None and (i not in holders or ph[4] > holders[i][4]):
                    holders[i] = ph
            # The office-trained model looks at each person as a whole (hands,
            # head, phone) - it sees uses the phone detector cannot.
            try:
                for i, p in self._looks(frame, seen.people).items():
                    if p >= self._cls_cut and i not in holders:
                        x1, y1, x2, y2, _ = seen.people[i]
                        holders[i] = (x1, y1, x2, y2, p)
            except Exception:                             # noqa: BLE001
                log.exception("phone model failed")
        ends, keep = [], []
        with self._lock:
            st = self._state.setdefault(camera, self._blank())
            old, tracks = list(st["tracks"]), []
            for i, person in enumerate(seen.people):
                tr, best = None, self.MATCH_IOU
                for t in old:
                    iou = _iou(t["box"], person)
                    if iou >= best:
                        tr, best = t, iou
                if tr is not None:
                    old.remove(tr)
                else:
                    tr = {"first": None, "last": 0.0, "hits": 0, "phone": None,
                          "event": None, "saved": 0.0, "shot": False}
                tr["box"] = person
                if i in holders:
                    if tr["first"] is None or ts - tr["last"] > self.GAP:
                        self._close(tr, ends)               # an earlier use is over
                        tr.update(first=ts, hits=0)
                    tr["last"], tr["phone"] = ts, holders[i]
                    tr["hits"] += 1
                elif tr["first"] is not None and ts - tr["last"] > self.GAP:
                    self._close(tr, ends)
                tracks.append(tr)
            for t in old:                                    # people who left
                self._close(t, ends)
            if not watching:                                 # a break, or after hours
                for t in tracks:
                    self._close(t, ends)
            st["tracks"] = tracks
            using = [t for t in tracks if t["first"] is not None and ts - t["last"] <= self.GAP
                     and t["hits"] >= need and t["last"] - t["first"] >= min_s]
            st["using"] = [t["box"] for t in using]
            st["phones"] = [t["phone"] for t in tracks if t["phone"] and ts - t["last"] <= self.GAP]
            st["on_phone"] = len(using)
            new = [t for t in using if t["event"] is None]
            for t in new:
                t["event"], t["saved"] = "writing", ts
            # Uses already recorded without a photo (they started inside the gap),
            # pictured only in a check that sees the phone with them.
            late = [t for t in using if t["event"] not in (None, "writing")
                    and not t["shot"] and t["last"] == ts]
            shoot = bool(new or late) and ts - st["photo_at"] >= float(rules.get("photo_every", 2)) * 60
            if shoot:
                st["photo_at"] = ts
                for t in new + late:
                    t["shot"] = True
            else:
                late = []
            for t in using:
                if t["event"] not in (None, "writing") and ts - t["saved"] >= 30:
                    t["saved"] = ts
                    keep.append((t["event"], t["first"], t["last"]))
            starts = [(t, t["first"]) for t in new]
            late_ids = [t["event"] for t in late]
            shown = new + late
            targets = [t["box"] for t in shown]
            others = [t["box"] for t in tracks if all(t is not u for u in shown)]
            phones_now = list(st["phones"])

        shot = None
        if shoot:
            shot = photos.save(camera, "phone", frame, targets, others + phones_now,
                               colour=AMBER, ts=datetime.fromtimestamp(ts))
        if shot:
            for eid in late_ids:
                events.update_meta(eid, {"photos": [shot]})
        for t, first in starts:
            meta = {"started": _iso(first), "last_seen": _iso(ts), "seconds": int(ts - first)}
            if shot:
                meta["photos"] = [shot]
            eid = self.emit(camera, "phone_use", 1, "using a phone", meta)
            with self._lock:
                if t["event"] == "writing":
                    t["event"] = eid
                else:                                         # it ended meanwhile
                    ends.append((eid, first, t["last"] or ts))
        for eid, first, last in keep:
            events.update_meta(eid, {"last_seen": _iso(last), "seconds": int(last - first)})
        for eid, first, last in ends:
            events.update_meta(eid, {"ended": _iso(last), "last_seen": _iso(last),
                                     "seconds": int(last - first)})
        return None

    @staticmethod
    def _close(t, ends):
        """A use is over: note its record for the end time, forget it."""
        if t["event"] not in (None, "writing"):
            ends.append((t["event"], t["first"], t["last"]))
        t.update(first=None, hits=0, phone=None, event=None, shot=False)

    def overlay(self, camera, frame):
        with self._lock:
            st = self._state.get(camera)
            if not st:
                return frame
            phones, using = list(st["phones"]), list(st["using"])
        for x1, y1, x2, y2, _ in phones:                 # phones in somebody's hands, faint
            cv2.rectangle(frame, (x1, y1), (x2, y2), AMBER, 1)
        for x1, y1, x2, y2, _ in using:
            cv2.rectangle(frame, (x1, y1), (x2, y2), AMBER, 2)
            _label(frame, "on phone", x1, y1, AMBER)
        return frame

    def caption(self, camera):
        n = self.on_phone(camera)
        return ("{} on phone".format(n), False) if n else None

    def busy(self, camera):
        """A use is building up: look again soon so it is confirmed in time."""
        now = time.time()
        with self._lock:
            st = self._state.get(camera)
            return bool(st and any(t["first"] is not None and t["hits"] < 50 and
                                   now - t["first"] < self.PENDING and now - t["last"] < self.GAP
                                   for t in st["tracks"]))

    def on_phone(self, camera):
        with self._lock:
            st = self._state.get(camera)
            return st["on_phone"] if st else 0

    def using(self, camera):
        """The people on a phone right now (their boxes) - for the Search page."""
        with self._lock:
            st = self._state.get(camera)
            return list(st["using"]) if st else []

    def forget(self, camera):
        with self._lock:
            self._state.pop(camera, None)


@register
class RestrictedArea(Detector):
    """Nobody belongs here. Anybody whose feet are inside the zone drawn for
    the camera (or anywhere in view, when no zone is drawn) raises an alert
    with photos, and it clears once the zone has looked empty for a few
    seconds. While the alert is open a few more photos are taken, so there is
    a clear picture of who it was.

    Two guards against false alerts. Somebody the model is sure of (SURE)
    raises the alert at once; a weak sighting needs a second one within
    CONFIRM_WITHIN seconds - long enough for a busy PC that checks the camera
    only every few seconds. And each camera slowly learns how the empty room
    looks: a weak "person" that matches that picture - a cloth on a sofa, a
    coat on a chair - is part of the room, not an intruder. Anybody the model
    is very sure of always counts, still or not."""

    key = "restricted"
    title = "Restricted area"
    groups = (config.RESTRICTED,)
    ALERT_GAP = 15.0        # coming back within this continues the same alert
    CLEAR_AFTER = 5.0       # zone must look empty this long before it clears
    PHOTO_EVERY = 4.0
    MAX_PHOTOS = 4
    SURE = 0.5              # a person this certain raises the alert at once
    CONFIRM = 2             # a weaker one must be seen in this many checks ...
    CONFIRM_WITHIN = 15.0   # ... within this many seconds before an alert opens
    BG_WIDTH = 320          # the learnt room picture, pixels across
    BG_TIME = 180.0         # how slowly it learns (s): somebody walking in is not learnt
    STILL_MATCH = 0.9       # a box this like the room picture is part of the room ...
    STILL_SCORE = 0.9       # ... unless the model is at least this sure it is a person

    def __init__(self):
        super().__init__()
        self._state = {}

    @staticmethod
    def _blank():
        return {"inside": [], "outside": [], "zone": [], "occupied": False,
                "since": 0.0, "started": 0.0, "last_seen": 0.0, "cleared_at": 0.0,
                "event": None, "photos": [], "photo_at": 0.0, "peak": 0,
                "hits": [], "bg": None, "bg_at": 0.0}

    def _small(self, frame):
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h = int(grey.shape[0] * self.BG_WIDTH / grey.shape[1])
        return cv2.resize(grey, (self.BG_WIDTH, h)).astype(np.float32)

    def _part_of_room(self, box, small, bg, shape):
        """Does this weak box look just like the learnt empty room there?"""
        if bg is None or box[4] >= self.STILL_SCORE:
            return False
        s = small.shape[1] / float(shape[1])
        x1, y1, x2, y2 = (int(round(v * s)) for v in box[:4])
        a = small[max(0, y1):y2, max(0, x1):x2]
        b = bg[max(0, y1):y2, max(0, x1):x2]
        if a.shape[0] < 8 or a.shape[1] < 8:
            return False                    # too small to judge: count it
        a, b = a - a.mean(), b - b.mean()
        d = float(np.sqrt((a * a).sum() * (b * b).sum()))
        return d > 1e-6 and float((a * b).sum()) / d >= self.STILL_MATCH

    @staticmethod
    def split(people, zone, shape):
        """(inside, outside). A person is inside when the bottom middle of
        their box - their feet - is inside the zone."""
        if not zone or len(zone) < 3:
            return list(people), []
        h, w = shape[:2]
        poly = np.array([[x * w, y * h] for x, y in zone], dtype=np.float32)
        inside, outside = [], []
        for box in people:
            foot = ((box[0] + box[2]) / 2.0, float(box[3]))
            (inside if cv2.pointPolygonTest(poly, foot, False) >= 0
             else outside).append(box)
        return inside, outside

    def process(self, camera, frame, ts, seen):
        zone = (config.camera(camera) or {}).get("zone") or []
        small = self._small(frame)
        with self._lock:
            st = self._state.setdefault(camera, self._blank())
            bg = st["bg"]
        room = [b for b in seen.people if self._part_of_room(b, small, bg, frame.shape)]
        people = [b for b in seen.people if all(b is not r for r in room)]
        inside, outside = self.split(people, zone, frame.shape)
        outside = outside + room                    # shown grey, never an alert
        n = len(inside)
        when = datetime.fromtimestamp(ts)
        shoot = new_alert = cleared = False

        reopened = False
        with self._lock:
            # Learn the room slowly - somebody walking through is not learnt.
            if st["bg"] is None or small.shape != st["bg"].shape:
                st["bg"] = small
            else:
                a = min(1.0, max(0.0, ts - st["bg_at"]) / self.BG_TIME)
                st["bg"] = st["bg"] * (1.0 - a) + small * a
            st["bg_at"] = ts
            st["inside"], st["outside"], st["zone"] = inside, outside, zone
            if n:
                st["hits"] = [t for t in st["hits"] if ts - t <= self.CONFIRM_WITHIN] + [ts]
            sure = any(b[4] >= self.SURE for b in inside)
            if n and (st["occupied"] or sure or len(st["hits"]) >= self.CONFIRM):
                if not st["occupied"]:
                    st["occupied"], st["since"] = True, ts
                    if st["event"] and ts - st["cleared_at"] < self.ALERT_GAP:
                        reopened = True             # back again: same alert
                    else:
                        new_alert = shoot = True
                        st.update(started=ts, photos=[], peak=n, event=None)
                st["last_seen"] = ts
                st["peak"] = max(st["peak"], n)
                if (not new_alert and st["event"] and
                        len(st["photos"]) < self.MAX_PHOTOS and
                        ts - st["photo_at"] >= self.PHOTO_EVERY):
                    shoot = True
                if shoot:
                    st["photo_at"] = ts
            elif st["occupied"] and ts - st["last_seen"] >= self.CLEAR_AFTER:
                st["occupied"], st["since"] = False, 0.0
                st["cleared_at"] = ts
                cleared = True
            event_id, peak = st["event"], st["peak"]
            started, last_seen = st["started"], st["last_seen"]

        if reopened and event_id:
            events.update_meta(event_id, {"ended": None, "seconds": None})
        produced = []
        if shoot:
            shot = photos.save(camera, "alert", frame, inside, outside,
                               zone=zone, ts=when)
            with self._lock:
                st["photos"].append(shot)
                shots = list(st["photos"])
            if new_alert:
                event_id = self.emit(
                    camera, "restricted_alert", n,
                    "{} person{} in a restricted area".format(n, "" if n == 1 else "s"),
                    {"photos": shots, "people": n,
                     "started": when.isoformat(timespec="seconds")})
                with self._lock:
                    st["event"] = event_id
            elif event_id:
                events.update_meta(event_id, {"photos": shots, "people": peak})
        if cleared:
            if event_id:
                events.update_meta(event_id, {
                    "people": peak,
                    "ended": datetime.fromtimestamp(last_seen).isoformat(timespec="seconds"),
                    "seconds": round(last_seen - started)})
            produced.append({"kind": "restricted_clear", "value": 0,
                             "label": "area clear"})
        return produced or None

    def overlay(self, camera, frame):
        zone = (config.camera(camera) or {}).get("zone") or []
        with self._lock:
            st = self._state.get(camera) or self._blank()
            inside, outside = list(st["inside"]), list(st["outside"])
        if zone:
            photos.draw_zone(frame, zone, RED)
        for x1, y1, x2, y2, _ in outside:
            cv2.rectangle(frame, (x1, y1), (x2, y2), GREY, 1)
        for x1, y1, x2, y2, score in inside:
            cv2.rectangle(frame, (x1, y1), (x2, y2), RED, 2)
            _label(frame, "intruder {:.0%}".format(score), x1, y1, RED)
        return frame

    def caption(self, camera):
        with self._lock:
            st = self._state.get(camera)
            if st and st["occupied"]:
                n = max(len(st["inside"]), 1)
                return ("ALERT - {} in restricted area".format(n), True)
        return ("area clear", False)

    def busy(self, camera):
        with self._lock:
            st = self._state.get(camera)
            return bool(st and (st["occupied"] or st["inside"]))

    def alert(self, camera):
        with self._lock:
            st = self._state.get(camera)
            if not st:
                return {"active": False, "count": 0, "since": 0, "event": None}
            return {"active": st["occupied"], "count": len(st["inside"]),
                    "since": st["since"], "event": st["event"]}

    def forget(self, camera):
        with self._lock:
            self._state.pop(camera, None)


# ---------------------------------------------------------------- steadier counts

def _iou(a, b):
    return _overlap(a, b)[0]


class Smoother:
    """Keeps a counting camera's number steady. Seen from a ceiling, somebody
    sitting still is found in one check and missed in the next, so the count
    would blink. Each person is remembered across checks instead:

    - a person missed in a check stays counted until they have been missed
      KEEP_MISSES times in a row *and* for KEEP_SECONDS;
    - a weak first sighting (below CONFIRM_SCORE) only counts once it is seen
      a second time, so a one-off false box does not either.
    """

    KEEP_MISSES = 2
    KEEP_SECONDS = 4.0
    CONFIRM_SCORE = 0.45
    MATCH_IOU = 0.3

    def __init__(self):
        self._lock = threading.Lock()
        self._people = {}       # camera -> [{"box", "best", "hits", "misses", "last"}]

    def update(self, camera, seen, ts):
        """Return `seen` with people replaced by everybody who counts now."""
        with self._lock:
            tracks = self._people.setdefault(camera, [])
            free = list(range(len(seen.people)))
            for tr in sorted(tracks, key=lambda t: -t["hits"]):
                best, best_iou = None, self.MATCH_IOU
                for i in free:
                    iou = _iou(tr["box"], seen.people[i])
                    if iou >= best_iou:
                        best, best_iou = i, iou
                if best is None:
                    tr["misses"] += 1
                    continue
                free.remove(best)
                box = seen.people[best]
                tr.update(box=box, best=max(tr["best"], box[4]), hits=tr["hits"] + 1,
                          misses=0, last=ts)
            for i in free:
                box = seen.people[i]
                tracks.append({"box": box, "best": box[4], "hits": 1, "misses": 0,
                               "last": ts})
            tracks[:] = [t for t in tracks if not (
                t["misses"] >= self.KEEP_MISSES and ts - t["last"] >= self.KEEP_SECONDS)]
            # A remembered person lying on top of somebody seen right now is the
            # same person seen differently (whole, then half) - drop the memory.
            # Two people seen at the same time are never merged: from a ceiling
            # camera, two people sitting one behind the other have nested boxes.
            live = [t for t in tracks if not t["misses"]]
            tracks[:] = live + [t for t in tracks if t["misses"] and
                                not any(_same_person(t["box"], l["box"]) for l in live)]
            counted = [t["box"] for t in tracks if self._counts(t)]
        return Seen(counted, seen.phones, seen.vehicles, seen.objects)

    def _counts(self, t):
        return t["hits"] >= 2 or t["best"] >= self.CONFIRM_SCORE

    def busy(self, camera):
        """Somebody is waiting to be confirmed or to be let go - look again
        soon rather than at the next routine refresh."""
        with self._lock:
            return any(t["misses"] or not self._counts(t)
                       for t in self._people.get(camera, []))

    def forget(self, camera):
        with self._lock:
            self._people.pop(camera, None)


# ---------------------------------------------------------------- pipeline

class Track:
    """What the pipeline remembers about one camera."""
    __slots__ = ("seq", "ts", "frame_ts", "small", "seen", "ver", "busy", "waiting", "size", "jpg")

    def __init__(self):
        self.seq = -1           # frame last looked at
        self.ts = 0.0           # when
        self.frame_ts = 0.0     # when that frame was taken
        self.small = None       # tiny grey copy, to spot changes
        self.seen = None
        self.ver = 0            # bumps on every result, for image caches
        self.busy = False
        self.waiting = False    # a detection waits on the routine re-check
        self.size = None        # (width, height) of the picture looked at
        self.jpg = None         # that picture, small - the Search shows it and reads colours off it


def _small(frame):
    tiny = cv2.resize(frame, (80, 60), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(cv2.cvtColor(tiny, cv2.COLOR_BGR2GRAY), (5, 5), 0)


THUMB_W = 640               # the picture kept from each check, for the Search


def _thumb(frame):
    h, w = frame.shape[:2]
    if w > THUMB_W:
        frame = cv2.resize(frame, (THUMB_W, max(1, int(h * THUMB_W / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buf.tobytes() if ok else None


class Pipeline:
    HEARTBEAT = 10.0        # look at a quiet restricted camera at least this often
    COUNT_GAP = 1.5         # never re-check a counting camera faster than this...
    WATCHED_GAP = 0.5       # ...unless somebody is watching it live
    RESTRICTED_WAIT = 2.0   # longest a restricted camera waits behind counting (s)
    STALE = 6.0             # a frame this old means the camera stopped
    MOTION_LEVEL = 20       # grey-level change that counts as different
    MOTION_SHARE = {config.COUNT: 0.004,          # share of the picture that
                    config.RESTRICTED: 0.002}     # has to change

    def __init__(self):
        self._lock = threading.Lock()
        self._inbox = {}            # camera -> (frame, ts, seq)
        self._tracks = {}           # camera -> Track
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._reload = threading.Event()
        self._reload.set()          # the first load happens on the worker
        self._thread = None
        self._vision = None
        self._detectors = [cls() for cls in REGISTRY.values()]
        self._cfg = config.get("detection")
        self._state = "loading" if self._cfg.get("enabled") else "off"
        self._error = ""
        self._watched = lambda camera: False
        self._groups, self._groups_at = {}, 0.0
        self._ignore = {}           # camera -> [polygon as float32 0-1 points]
        self._smooth = Smoother()
        self._runs = deque(maxlen=400)        # (end, frames, seconds)
        self._lag = {config.COUNT: None, config.RESTRICTED: None}
        self._running_since = 0.0
        self._look_all_at = 0.0     # "Check now" on the Search page
        self._jobs = deque()        # run_soon(): the AI search, one picture each

    # ---------- wiring ----------

    def attach(self, watched):
        """watched(camera) -> True while somebody has its live video open."""
        self._watched = watched

    def add(self, detector):
        """A detection from another module (washroom.py), added after start-up."""
        self._detectors.append(detector)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="detect",
                                        daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._wake.set()

    def reload(self):
        """Settings changed. The worker picks them up between two batches."""
        cfg = config.get("detection")
        with self._lock:
            self._cfg = cfg
            if cfg.get("enabled") and self._vision is None:
                self._state = "loading"
        self._reload.set()
        self._wake.set()

    def on_frame(self, camera, frame, ts, seq):
        """Called by the camera threads with every decoded frame. Only the
        newest one per camera is kept - the count should describe now, not
        a queue from a minute ago."""
        with self._lock:
            self._inbox[camera] = (frame, ts, seq)
        if self._group_map().get(camera) == config.WASHROOM:
            # The door watch looks at every frame (cheap, no model) - a door
            # opens and shuts between two model checks.
            wash = self._det("washroom")
            if wash is not None:
                try:
                    wash.on_frame(camera, frame, ts)
                except Exception:                         # noqa: BLE001
                    log.exception("washroom door watch failed")
        self._wake.set()

    def forget(self, camera):
        with self._lock:
            self._inbox.pop(camera, None)
            self._tracks.pop(camera, None)
        self._smooth.forget(camera)
        for det in self._detectors:
            det.forget(camera)

    # ---------- the worker ----------

    def _run(self):
        while not self._stop.is_set():
            if self._reload.is_set():
                self._load()
            if self._state != "running":
                self._wake.wait(0.5)
                self._wake.clear()
                continue
            picks = self._pick()
            with self._lock:
                # A search job goes before counting, never before a restricted check.
                job = self._jobs.popleft() if self._jobs and not (picks and picks[0][0] == 0) else None
            if job is not None:
                try:
                    job()
                except Exception:                         # noqa: BLE001
                    log.exception("search job failed")
                continue
            if not picks:
                self._wake.wait(0.2)
                self._wake.clear()
                continue
            self._analyse(picks)

    def _load(self):
        self._reload.clear()
        cfg = config.get("detection")
        with self._lock:
            self._cfg = cfg
        if not cfg.get("enabled"):
            with self._lock:
                self._state, self._error = "off", ""
            return
        try:
            vision = self._vision
            if vision is None or not vision.same_model(cfg):
                with self._lock:
                    self._state = "loading"
                vision = Vision(cfg)
                log.info("model ready: %s on %s, batch %s", vision.model_name,
                         vision.device_name, vision.batch)
            vision.configure(cfg)
            with self._lock:
                self._vision = vision
                self._state, self._error = "running", ""
                self._running_since = time.time()
                self._runs.clear()
        except Exception as exc:                          # noqa: BLE001
            log.exception("could not load the detection model")
            with self._lock:
                self._state, self._error = "error", str(exc)[:200]

    def _group_map(self):
        now = time.time()
        if now - self._groups_at > 1.0:
            cams = config.cameras()
            self._groups = {c["id"]: c.get("group", config.COUNT) for c in cams}
            self._ignore = {c["id"]: [np.array(p, dtype=np.float32)
                                      for p in c.get("ignore") or [] if len(p) >= 3]
                            for c in cams}
            self._groups_at = now
        return self._groups

    def _drop_ignored(self, camera, seen, shape):
        """Leave out anything whose middle is inside one of the camera's
        ignore areas - a poster, a TV, a mannequin that looks like a person."""
        polys = self._ignore.get(camera)
        if not polys:
            return seen
        h, w = shape[:2]

        def keep(box):
            mid = ((box[0] + box[2]) / 2.0 / w, (box[1] + box[3]) / 2.0 / h)
            return all(cv2.pointPolygonTest(p, mid, False) < 0 for p in polys)

        return Seen([b for b in seen.people if keep(b)],
                    [b for b in seen.phones if keep(b)],
                    [b for b in seen.vehicles if keep(b)],
                    [b for b in seen.objects if keep(b)])

    def group(self, camera):
        return self._group_map().get(camera, config.COUNT)

    def _moved(self, before, now_small, group):
        if before is None:
            return True
        diff = cv2.absdiff(before, now_small)
        share = float(np.count_nonzero(diff > self.MOTION_LEVEL)) / diff.size
        return share >= self.MOTION_SHARE.get(group, 0.004)

    def _pick(self):
        """Which cameras to look at next, most urgent first."""
        now = time.time()
        groups = self._group_map()
        cfg = self._cfg
        restricted_gap = 1.0 / max(0.5, float(cfg.get("restricted_fps", 4)))
        refresh = max(5.0, float(cfg.get("count_refresh", 30)))
        with self._lock:
            inbox = list(self._inbox.items())
        wash = self._det("washroom")

        due = []
        for cam, (frame, fts, seq) in inbox:
            group = groups.get(cam)
            if group is None:
                continue                                    # camera was removed
            tr = self._tracks.get(cam)
            if tr is None:
                tr = self._tracks[cam] = Track()
            if seq == tr.seq or now - fts > self.STALE:
                continue
            age = now - tr.ts if tr.ts else 1e9
            forced = tr.ts < self._look_all_at          # asked for a fresh look (Search page)
            if group == config.WASHROOM:
                # Its door watch says how often: twice a second while the door
                # is in use, about once a second while somebody walks near it,
                # else once a minute - a busy corridor elsewhere costs nothing.
                gap, rank = wash.schedule(cam) if wash else (60.0, 3)
                if age >= gap or forced:
                    due.append((min(rank, 2) if forced else rank, -age, cam, frame, fts, seq, group, None))
                continue
            restricted = group == config.RESTRICTED
            watched = self._watched(cam)
            if restricted:
                if age < restricted_gap:
                    continue
                rank = 0
            else:
                if age < (self.WATCHED_GAP if watched else self.COUNT_GAP):
                    continue
                rank = 1 if watched else 2

            small = None
            if forced and not restricted:
                pass                                        # looked at now, moving or not
            elif tr.seen is not None and not tr.busy and \
                    age < (self.HEARTBEAT if restricted else refresh):
                small = _small(frame)
                if not self._moved(tr.small, small, group):
                    continue                                # nothing changed
            elif tr.seen is not None and not tr.busy and not restricted:
                rank = 2 if tr.waiting else 3               # routine refresh: last
            due.append((rank, -age, cam, frame, fts, seq, group, small))

        if not due:
            return []
        due.sort(key=lambda d: (d[0], d[1]))
        if due[0][0] == 0:
            # A restricted camera never waits behind a batch of counting ones.
            due = [d for d in due if d[0] == 0]
            return due[:self._vision.batch]
        # ...nor long behind one already running: while there is a restricted
        # camera, a counting batch is kept short enough to finish within
        # RESTRICTED_WAIT seconds on this PC.
        return due[:self._count_batch(groups)]

    def _count_batch(self, groups):
        batch = self._vision.batch
        if config.RESTRICTED not in groups.values():
            return batch
        runs = list(self._runs)[-20:]
        frames = sum(r[1] for r in runs)
        per = sum(r[2] for r in runs) / frames if frames else 0.5
        return max(1, min(batch, int(self.RESTRICTED_WAIT / max(per, 0.05))))

    def _analyse(self, picks):
        frames = [p[3] for p in picks]
        started = time.time()
        try:
            seens = self._vision.detect(frames)
        except Exception as exc:                          # noqa: BLE001
            log.exception("detection failed")
            with self._lock:
                self._error = str(exc)[:200]
            time.sleep(1.0)
            return
        done = time.time()
        self._runs.append((done, len(frames), done - started))

        for (_, _, cam, frame, fts, seq, group, small), seen in zip(picks, seens):
            lag = done - fts
            old = self._lag.get(group)
            self._lag[group] = lag if old is None else old * 0.9 + lag * 0.1
            seen = self._drop_ignored(cam, seen, frame.shape)
            # Counting cameras get a steady number; restricted cameras use every
            # raw sighting - an alert must not wait for a second look.
            if group == config.COUNT:
                seen = self._smooth.update(cam, seen, fts)
            jpg = _thumb(frame)
            with self._lock:
                tr = self._tracks.get(cam)
                if tr is None:
                    continue                                # removed meanwhile
                tr.seq, tr.ts, tr.frame_ts, tr.seen = seq, done, fts, seen
                tr.size, tr.jpg = (frame.shape[1], frame.shape[0]), jpg
                tr.small = small if small is not None else _small(frame)
                tr.ver += 1
            busy = group == config.COUNT and self._smooth.busy(cam)
            waiting = False
            for det in self._detectors:
                if group not in det.groups:
                    continue
                try:
                    produced = det.process(cam, frame, fts, seen) or []
                except Exception as exc:                  # noqa: BLE001
                    log.exception("detector %s failed", det.key)
                    with self._lock:
                        self._error = "{}: {}".format(det.key, exc)[:200]
                    continue
                for ev in produced:
                    events.log(kind=ev.get("kind", det.key), camera=cam,
                               label=ev.get("label", ""), value=ev.get("value", 0),
                               meta=ev.get("meta"))
                busy = busy or det.busy(cam)
                waiting = waiting or det.waiting(cam)
            tr.busy, tr.waiting = busy, waiting

    # ---------- what the rest of the app reads ----------

    def _det(self, key):
        for det in self._detectors:
            if det.key == key:
                return det
        return None

    def overlay(self, camera, frame, caption=True):
        """Boxes, the zone and a caption on top of a frame (drawn in place).
        The camera wall shows the numbers itself, so it asks for no caption."""
        group = self.group(camera)
        with self._lock:
            tr = self._tracks.get(camera)
            seen = tr.seen if tr else None
            state = self._state
        if caption:                         # live view: show what is ignored
            for poly in self._ignore.get(camera) or []:
                photos.draw_zone(frame, poly.tolist(), GREY)
                h, w = frame.shape[:2]
                x, y = int(poly[:, 0].min() * w) + 4, int(poly[:, 1].min() * h) + 16
                cv2.putText(frame, "ignored", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (230, 230, 230), 1, cv2.LINE_AA)
        dets = [d for d in self._detectors if group in d.groups]
        if seen is not None and state == "running":
            for det in dets:
                try:
                    frame = det.overlay(camera, frame)
                except Exception:                         # noqa: BLE001
                    log.exception("overlay %s failed", det.key)
            parts = [c for c in (d.caption(camera) for d in dets) if c]
            alert = any(a for _, a in parts)
            text = " - ".join(t for t, a in parts if a == alert)
        else:
            alert = False
            text = {"off": "detection off", "loading": "loading the model...",
                    "error": "detection error"}.get(state, "waiting for detection")
        if group == config.RESTRICTED and seen is None:
            zone = (config.camera(camera) or {}).get("zone") or []
            if zone:
                photos.draw_zone(frame, zone, RED)
        if caption:
            _caption(frame, text, alert)
        return frame

    def version(self, camera):
        with self._lock:
            tr = self._tracks.get(camera)
            return tr.ver if tr else 0

    def look_all(self):
        """Look at every camera again soon, moving or not (Search -> Check now).
        Restricted cameras still come first; this PC needs a minute or so."""
        with self._lock:
            self._look_all_at = time.time()
        self._wake.set()

    def class_names(self):
        """Everything the model can find, by name (the Search page)."""
        vision = self._vision
        return sorted(vision.names.values()) if vision else []

    def detector(self, key):
        return self._det(key)

    def latest(self):
        """What the last check of each camera found, for the Search page:
        {camera: {"at": when, "seen": Seen, "size": (w, h), "live": frames
        coming now}}, and since when a "Check now" is waiting (0 = none)."""
        now = time.time()
        with self._lock:
            out = {cam: {"at": tr.ts, "seen": tr.seen, "size": tr.size, "jpg": tr.jpg,
                         "live": now - self._inbox.get(cam, (None, 0.0))[1] < 15}
                   for cam, tr in self._tracks.items() if tr.seen is not None and tr.size}
            return out, self._look_all_at

    def run_soon(self, job):
        """Run job() on the detection worker - after any restricted-area check
        that is due, before counting checks. The AI search does its pictures
        this way, one at a time, so an alert never waits behind it."""
        with self._lock:
            self._jobs.append(job)
        self._wake.set()

    def drop_jobs(self):
        with self._lock:
            self._jobs.clear()

    def live(self):
        """Per-camera numbers for the dashboard, in one go."""
        now = time.time()
        refresh = float(self._cfg.get("count_refresh", 30))
        phone = self._det("phone")
        area = self._det("restricted")
        wash = self._det("washroom")
        out = {}
        with self._lock:
            tracks = {cam: (tr.ts, tr.seen, tr.ver) for cam, tr in self._tracks.items()}
            inbox = {cam: v[1] for cam, v in self._inbox.items()}
            running = self._state == "running"
        groups = self._group_map()
        for cam, group in groups.items():
            ts, seen, ver = tracks.get(cam, (0.0, None, 0))
            fresh = (running and seen is not None and
                     now - inbox.get(cam, 0) < 15 and now - ts < refresh + 30)
            item = {"checked_ago": round(now - ts, 1) if ts else None,
                    "fresh": fresh, "ver": ver}
            if group == config.COUNT:
                item["count"] = len(seen.people) if fresh else None
                item["on_phone"] = phone.on_phone(cam) if (fresh and phone) else None
            elif group == config.WASHROOM:
                # Checked only when the door is in use, so "fresh" is about
                # the camera's frames, not the last model check.
                info = wash.info(cam) if wash else {}
                item["fresh"] = running and now - inbox.get(cam, 0) < 15
                item["washroom"] = dict(info, watching=item["fresh"])
            else:
                a = area.alert(cam) if area else {}
                item["alert"] = {"active": bool(fresh and a.get("active")),
                                 "count": a.get("count", 0) if fresh else 0,
                                 "since": a.get("since", 0),
                                 "watching": fresh}
            out[cam] = item
        return out

    def status(self):
        now = time.time()
        with self._lock:
            runs = [r for r in self._runs if now - r[0] < 30]
            vision = self._vision
            state, error = self._state, self._error
            since = self._running_since
        frames = sum(r[1] for r in runs)
        busy = sum(r[2] for r in runs)
        window = min(30.0, now - since) if since else 0
        return {
            "enabled": bool(self._cfg.get("enabled")),
            "state": state,
            "error": error,
            "device": vision.device_name if vision else "",
            "batch": vision.batch if vision else 0,
            "available": [{"key": d.key, "title": d.title} for d in self._detectors],
            "perf": {
                "per_second": round(frames / window, 2) if window > 1 else 0,
                "ms_per_frame": round(1000 * busy / frames) if frames else 0,
                "load": round(min(1.0, busy / window), 2) if window > 1 else 0,
                "lag_count": _round(self._lag.get(config.COUNT)),
                "lag_restricted": _round(self._lag.get(config.RESTRICTED)),
            },
        }


def _round(x):
    return round(x, 1) if x is not None else None


def _caption(frame, text, alert):
    if not text:
        return
    if alert:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 46), (40, 40, 200), -1)
        cv2.putText(frame, text, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (8, 8), (24 + tw, 40), (20, 20, 20), -1)
    cv2.putText(frame, text, (16, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)


pipeline = Pipeline()
