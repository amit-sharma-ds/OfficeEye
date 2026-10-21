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
    __slots__ = ("people", "phones")

    def __init__(self, people=None, phones=None):
        self.people = people or []      # [(x1, y1, x2, y2, score), ...]
        self.phones = phones or []


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
        self.configure(cfg)
        self.detect([np.zeros((480, 640, 3), np.uint8)])       # warm up

    def same_model(self, cfg):
        return (cfg.get("model", "yolo11n.pt") == self.model_name and
                cfg.get("device", "auto") in (("auto", "cuda") if self.half
                                              else ("auto", "cpu")))

    def configure(self, cfg):
        """Settings that do not need the model reloaded."""
        self.imgsz = int(cfg.get("imgsz", 480))
        self.conf = float(cfg.get("confidence", 0.4))
        self.classes = [self.person, self.phone] if cfg.get("phone", True) \
            else [self.person]

    def detect(self, frames):
        extra = {"half": True} if self.half else {}
        results = self.model.predict(frames, imgsz=self.imgsz, conf=self.conf,
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
                    (seen.people if int(k) == self.person else seen.phones).append(box)
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


@register
class PhoneUse(Detector):
    """Who is *using* a phone, not just near one. A phone only counts when it
    is raised into the top half of a person - held up to read, or to the ear.
    A phone down by the hand, or lying on a desk somebody touches, is ignored.

    It has to be seen twice in a row before it counts, and then one photo is
    taken - at most one per camera per minute."""

    key = "phone"
    title = "Phone use"
    groups = (config.COUNT,)
    USING_ZONE = 0.50       # fraction of the person's height, from the top
    CONFIRM = 2             # checks in a row before it counts
    PHOTO_GAP = 60.0
    MIN_LOG_GAP = 2.0

    def __init__(self):
        super().__init__()
        self._state = {}

    @staticmethod
    def _blank():
        return {"people": [], "phones": [], "using": set(), "streak": 0,
                "on_phone": 0, "logged": None, "logged_at": 0.0, "photo_at": 0.0}

    def _using_phone(self, people, phones):
        """Indices into `people` of everybody using a phone."""
        using = set()
        for (fx1, fy1, fx2, fy2, _) in phones:
            cx, cy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
            best, best_area = None, None
            for i, (x1, y1, x2, y2, _) in enumerate(people):
                if not (x1 <= cx <= x2 and y1 <= cy <= y2):
                    continue                        # phone is not on this person
                if cy > y1 + self.USING_ZONE * (y2 - y1):
                    continue                        # phone is low: holding, not using
                area = (x2 - x1) * (y2 - y1)
                if best_area is None or area < best_area:
                    best, best_area = i, area        # nearest (smallest) person wins
            if best is not None:
                using.add(best)
        return using

    def process(self, camera, frame, ts, seen):
        using = self._using_phone(seen.people, seen.phones)
        produced = []
        with self._lock:
            st = self._state.setdefault(camera, self._blank())
            st["people"], st["phones"] = list(seen.people), list(seen.phones)
            st["streak"] = st["streak"] + 1 if using else 0
            st["using"] = using if st["streak"] >= self.CONFIRM else set()
            prev, n = st["on_phone"], len(st["using"])
            st["on_phone"] = n
            shoot = n > prev and ts - st["photo_at"] >= self.PHOTO_GAP
            if shoot:
                st["photo_at"] = ts
            log_it = (shoot or (st["logged"] != n and
                                ts - st["logged_at"] >= self.MIN_LOG_GAP))
            if log_it:
                quiet = not n and st["logged"] in (None, 0)
                st["logged"], st["logged_at"] = n, ts
                if quiet:
                    log_it = False                  # nothing to say about a steady zero

        if shoot:
            targets = [p for i, p in enumerate(seen.people) if i in using]
            others = [p for i, p in enumerate(seen.people) if i not in using]
            shot = photos.save(camera, "phone", frame, targets,
                               others + list(seen.phones), colour=AMBER,
                               ts=datetime.fromtimestamp(ts))
            produced.append({"kind": "phone_use", "value": n,
                             "label": "{} using a phone".format(n),
                             "meta": {"photos": [shot]}})
        elif log_it:
            produced.append({"kind": "phone_use", "value": n,
                             "label": ("{} using a phone".format(n) if n
                                       else "nobody on a phone")})
        return produced or None

    def overlay(self, camera, frame):
        with self._lock:
            st = self._state.get(camera)
            if not st:
                return frame
            people, phones, using = list(st["people"]), list(st["phones"]), set(st["using"])
        for x1, y1, x2, y2, _ in phones:                 # every phone, faint
            cv2.rectangle(frame, (x1, y1), (x2, y2), AMBER, 1)
        for i in using:
            if i < len(people):
                x1, y1, x2, y2, _ = people[i]
                cv2.rectangle(frame, (x1, y1), (x2, y2), AMBER, 2)
                _label(frame, "on phone", x1, y1, AMBER)
        return frame

    def caption(self, camera):
        n = self.on_phone(camera)
        return ("{} on phone".format(n), False) if n else None

    def busy(self, camera):
        with self._lock:
            st = self._state.get(camera)
            return bool(st and 0 < st["streak"] < self.CONFIRM)   # confirm it soon

    def on_phone(self, camera):
        with self._lock:
            st = self._state.get(camera)
            return st["on_phone"] if st else 0

    def forget(self, camera):
        with self._lock:
            self._state.pop(camera, None)


@register
class RestrictedArea(Detector):
    """Nobody belongs here. Anybody whose feet are inside the zone drawn for
    the camera (or anywhere in view, when no zone is drawn) raises an alert
    with photos, and it clears once the zone has looked empty for a few
    seconds. While the alert is open a few more photos are taken, so there is
    a clear picture of who it was."""

    key = "restricted"
    title = "Restricted area"
    groups = (config.RESTRICTED,)
    ALERT_GAP = 15.0        # coming back within this continues the same alert
    CLEAR_AFTER = 5.0       # zone must look empty this long before it clears
    PHOTO_EVERY = 4.0
    MAX_PHOTOS = 4

    def __init__(self):
        super().__init__()
        self._state = {}

    @staticmethod
    def _blank():
        return {"inside": [], "outside": [], "zone": [], "occupied": False,
                "since": 0.0, "started": 0.0, "last_seen": 0.0, "cleared_at": 0.0,
                "event": None, "photos": [], "photo_at": 0.0, "peak": 0}

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
        inside, outside = self.split(seen.people, zone, frame.shape)
        n = len(inside)
        when = datetime.fromtimestamp(ts)
        shoot = new_alert = cleared = False

        reopened = False
        with self._lock:
            st = self._state.setdefault(camera, self._blank())
            st["inside"], st["outside"], st["zone"] = inside, outside, zone
            if n:
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
        return Seen(counted, seen.phones)

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
    __slots__ = ("seq", "ts", "frame_ts", "small", "seen", "ver", "busy")

    def __init__(self):
        self.seq = -1           # frame last looked at
        self.ts = 0.0           # when
        self.frame_ts = 0.0     # when that frame was taken
        self.small = None       # tiny grey copy, to spot changes
        self.seen = None
        self.ver = 0            # bumps on every result, for image caches
        self.busy = False


def _small(frame):
    tiny = cv2.resize(frame, (80, 60), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(cv2.cvtColor(tiny, cv2.COLOR_BGR2GRAY), (5, 5), 0)


class Pipeline:
    HEARTBEAT = 10.0        # look at a quiet restricted camera at least this often
    COUNT_GAP = 1.5         # never re-check a counting camera faster than this...
    WATCHED_GAP = 0.5       # ...unless somebody is watching it live
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

    # ---------- wiring ----------

    def attach(self, watched):
        """watched(camera) -> True while somebody has its live video open."""
        self._watched = watched

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
                    [b for b in seen.phones if keep(b)])

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
            if tr.seen is not None and not tr.busy and \
                    age < (self.HEARTBEAT if restricted else refresh):
                small = _small(frame)
                if not self._moved(tr.small, small, group):
                    continue                                # nothing changed
            elif tr.seen is not None and not tr.busy and not restricted:
                rank = 3                                    # routine refresh: last
            due.append((rank, -age, cam, frame, fts, seq, group, small))

        if not due:
            return []
        due.sort(key=lambda d: (d[0], d[1]))
        if due[0][0] == 0:
            # A restricted camera never waits behind a batch of counting ones.
            due = [d for d in due if d[0] == 0]
        return due[:self._vision.batch]

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
            with self._lock:
                tr = self._tracks.get(cam)
                if tr is None:
                    continue                                # removed meanwhile
                tr.seq, tr.ts, tr.frame_ts, tr.seen = seq, done, fts, seen
                tr.small = small if small is not None else _small(frame)
                tr.ver += 1
            busy = group == config.COUNT and self._smooth.busy(cam)
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
            tr.busy = busy

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

    def live(self):
        """Per-camera numbers for the dashboard, in one go."""
        now = time.time()
        refresh = float(self._cfg.get("count_refresh", 30))
        phone = self._det("phone")
        area = self._det("restricted")
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
