"""Head down / sleeping at the desk, on the counting cameras.

How it is caught, cheapest check first:

1. Stillness (no model): every check, each person's cut-out is compared
   with the one from the check before. Somebody typing, talking or moving
   about changes it; somebody asleep barely does.
2. Head down (pose model): only people who have been still for a minute
   are looked at by a small pose model (yolo26n-pose), at most once every
   POSE_GAP seconds each, so it costs little. The head counts as down when
   it is no higher than the shoulders - on the desk, on the arms, hanging
   forward. Somebody seen from behind, with no head points, gives no
   verdict either way.
3. Still and head down for the minutes set under Sleeping -> Rules &
   timing: one record with a photo, kept up to date while it lasts, and
   given its end time when the person looks up, gets moving or leaves.

Outside the watch hours and in breaks nothing is recorded. Ceiling cameras
see heads from above, so the pose is weaker there than face-on; the minutes
rule keeps a quick look down at a desk out.
"""

import logging
import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np

import config
import events
import photos
from detection import BASE_DIR, Detector, _iou, _iso, _label, pipeline

log = logging.getLogger(__name__)

PURPLE = (200, 80, 170)        # BGR


def watch_state(rules, office, now):
    """Is head down checked at this moment? Times are "HH:MM", same day."""
    hm = now.strftime("%H:%M")
    start = rules.get("watch_from") or office.get("open") or "00:00"
    end = rules.get("watch_to") or office.get("close") or "23:59"
    if not start <= hm < end:
        return {"state": "off", "text": "Not watching now - head down is checked {} to {}".format(start, end)}
    for a, b in rules.get("breaks") or []:
        if a <= hm < b:
            return {"state": "break", "text": "Break until {} - head down is not recorded".format(b)}
    return {"state": "on", "text": "Watching until {}".format(end)}


def is_head_down(kp, width):
    """True / False from one person's 17 pose points [(x, y, conf), ...] in
    cut-out pixels, or None when the shoulders cannot be seen."""
    c = kp[:, 2]
    sh = [i for i in (5, 6) if c[i] >= HeadDown.SEEN]
    if not sh:
        return None
    head = [i for i in (0, 1, 2, 3, 4) if c[i] >= HeadDown.SEEN]
    if not head:
        return None                     # seen from behind, or not clear: no verdict
    sw = abs(kp[5, 0] - kp[6, 0]) if len(sh) == 2 else width * 0.35
    sw = max(sw, width * 0.15)
    rise = (np.mean(kp[sh, 1]) - np.mean(kp[head, 1])) / sw
    return bool(rise < HeadDown.RISE)


class HeadDown(Detector):
    key = "sleeping"
    title = "Head down"
    groups = (config.COUNT,)

    MODEL = "yolo26n-pose.pt"
    MATCH_IOU = 0.3            # the same person as in the last check
    STILL_IOU = 0.7            # the box hardly moved
    STILL_LEVEL = 25           # grey change that counts as movement
    STILL_SHARE = 0.08         # this much of the cut-out may change: still
    STILL_FIRST = 60.0         # still this long before the pose model looks
    POSE_GAP = 30.0            # one pose look per person at most this often
    POSE_SZ = 256
    PAD = 0.15
    MISSES = 2                 # "head up" looks in a row that end it
    MOVES = 3                  # checks in a row with movement that end it
    SEEN = 0.5                 # pose point confidence
    RISE = 0.25                # head centre less than this x shoulder width above them
    MAX_LOOKS = 1              # pose looks per check at most, so counting never waits long
    LOOK_EVERY = 2.0           # and one pose look in the whole app at most this often (s)

    def __init__(self):
        super().__init__()
        self._state = {}
        self._pose, self._pose_tried, self._device = None, 0.0, "cpu"
        self._pose_lock = threading.Lock()
        self._rules, self._rules_at = ({}, {}), 0.0
        self._looked_at = 0.0

    def rules(self):
        """(head-down rules, office hours) - read again every few seconds."""
        if time.time() - self._rules_at > 5:
            self._rules, self._rules_at = (config.get("sleeping"), config.get("office")), time.time()
        return self._rules

    def _model(self):
        """The pose model, loaded on first need (and tried again every few
        minutes if that failed)."""
        with self._pose_lock:
            if self._pose is None and time.time() - self._pose_tried > 300:
                self._pose_tried = time.time()
                try:
                    import torch
                    from ultralytics import YOLO
                    path = os.path.join(BASE_DIR, self.MODEL)
                    self._pose = YOLO(path if os.path.exists(path) else self.MODEL)
                    self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
                    log.info("head-down pose model ready: %s", self.MODEL)
                except Exception:                         # noqa: BLE001
                    log.exception("could not load the pose model")
            return self._pose

    def _small(self, grey, box):
        x1, y1, x2, y2 = (int(v) for v in box[:4])
        cut = grey[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
        if cut.size == 0:
            return None
        return cv2.GaussianBlur(cv2.resize(cut, (32, 48)), (3, 3), 0)

    def _still(self, a, b):
        if a is None or b is None:
            return False
        diff = cv2.absdiff(a, b)
        return float(np.count_nonzero(diff > self.STILL_LEVEL)) / diff.size <= self.STILL_SHARE

    def _looks(self, frame, boxes):
        """[True / False / None per box]: is the head down?"""
        model = self._model()
        if model is None or not boxes:
            return [None] * len(boxes)
        h, w = frame.shape[:2]
        out = []
        for x1, y1, x2, y2, *_ in boxes:
            px, py = int((x2 - x1) * self.PAD), int((y2 - y1) * self.PAD)
            crop = frame[max(0, y1 - py):min(h, y2 + py), max(0, x1 - px):min(w, x2 + px)]
            if crop.size == 0 or y2 - y1 < 60:
                out.append(None)                    # too small to judge
                continue
            r = model.predict(crop, imgsz=self.POSE_SZ, conf=0.25, verbose=False,
                              device=self._device)[0]
            if r.keypoints is None or not len(r.boxes):
                out.append(None)
                continue
            ch, cw = crop.shape[:2]
            mids = r.boxes.xyxy.cpu().numpy()
            # The pose found nearest the middle of the cut-out is this person.
            best = int(np.argmin([abs((m[0] + m[2]) / 2 - cw / 2) + abs((m[1] + m[3]) / 2 - ch / 2)
                                  for m in mids]))
            out.append(is_head_down(r.keypoints.data[best].cpu().numpy(), cw))
        return out

    @staticmethod
    def _new(person):
        return {"box": person, "small": None, "still_box": person, "still_since": None,
                "moves": 0, "pose_at": 0.0, "down_since": None, "last_down": 0.0,
                "ups": 0, "event": None, "saved": 0.0}

    def process(self, camera, frame, ts, seen):
        """Every head-down spell is its own record: written once it has
        lasted the set minutes (with a photo), kept up to date while it
        lasts, and given its end time when it is over."""
        rules, office = self.rules()
        watching = watch_state(rules, office, datetime.fromtimestamp(ts))["state"] == "on"
        need = max(0.5, float(rules.get("min_minutes", 5))) * 60
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ends = []
        with self._lock:
            st = self._state.setdefault(camera, {"tracks": [], "down": 0})
            old, tracks = list(st["tracks"]), []
            for person in seen.people:
                tr, best = None, self.MATCH_IOU
                for t in old:
                    iou = _iou(t["box"], person)
                    if iou >= best:
                        tr, best = t, iou
                if tr is not None:
                    old.remove(tr)
                else:
                    tr = self._new(person)
                small = self._small(grey, person)
                if tr["still_since"] is not None and _iou(tr["still_box"], person) >= self.STILL_IOU \
                        and self._still(tr["small"], small):
                    tr["moves"] = 0
                else:
                    tr["still_since"], tr["still_box"] = ts, person
                    tr["moves"] += 1
                    if tr["down_since"] is not None and tr["moves"] >= self.MOVES:
                        self._close(tr, ends)               # up and about again
                tr["box"], tr["small"] = person, small
                tracks.append(tr)
            for t in old:                                    # people who left
                self._close(t, ends)
            if not watching:                                 # a break, or after hours
                for t in tracks:
                    self._close(t, ends)
            st["tracks"] = tracks
            look = [t for t in tracks if watching and ts - t["pose_at"] >= self.POSE_GAP and
                    (t["down_since"] is not None or ts - t["still_since"] >= self.STILL_FIRST)]
            look = sorted(look, key=lambda t: t["pose_at"])[:self.MAX_LOOKS]   # longest waiting first
            if time.time() - self._looked_at < self.LOOK_EVERY:
                look = []                   # the pose model had its turn just now
            elif look:
                self._looked_at = time.time()
            for t in look:
                t["pose_at"] = ts
            boxes = [t["box"] for t in look]

        downs = self._looks(frame, boxes) if boxes else []

        new, keep = [], []
        with self._lock:
            for t, down in zip(look, downs):
                if down:
                    if t["down_since"] is None:
                        t["down_since"] = t["still_since"]  # still in this pose since then
                    t["last_down"], t["ups"] = ts, 0
                elif down is False and t["down_since"] is not None:
                    t["ups"] += 1
                    if t["ups"] >= self.MISSES:
                        self._close(t, ends)                # looked up
            for t in st["tracks"]:
                if t["down_since"] is None:
                    continue
                if t["event"] is None and t["last_down"] - t["down_since"] >= need:
                    t["event"], t["saved"] = "writing", ts
                    new.append(t)
                elif t["event"] not in (None, "writing") and ts - t["saved"] >= 30:
                    t["saved"] = ts
                    keep.append((t["event"], t["down_since"], t["last_down"]))
            st["down"] = sum(1 for t in st["tracks"] if t["event"] is not None)
            targets = [t["box"] for t in new]
            others = [t["box"] for t in st["tracks"] if all(t is not u for u in new)]

        shot = None
        if new:
            shot = photos.save(camera, "sleep", frame, targets, others,
                               colour=PURPLE, ts=datetime.fromtimestamp(ts))
        for t in new:
            first, last = t["down_since"], t["last_down"]
            meta = {"started": _iso(first), "last_seen": _iso(last), "seconds": int(last - first)}
            if shot:
                meta["photos"] = [shot]
            eid = self.emit(camera, "head_down", 1, "head down", meta)
            with self._lock:
                if t["event"] == "writing":
                    t["event"] = eid
                else:                                         # it ended meanwhile
                    ends.append((eid, first, t["last_down"] or last))
        for eid, first, last in keep:
            events.update_meta(eid, {"last_seen": _iso(last), "seconds": int(last - first)})
        for eid, first, last in ends:
            events.update_meta(eid, {"ended": _iso(last), "last_seen": _iso(last),
                                     "seconds": int(last - first)})
        return None

    @staticmethod
    def _close(t, ends):
        """A head-down spell is over: note its record for the end time."""
        if t["event"] not in (None, "writing"):
            ends.append((t["event"], t["down_since"], t["last_down"]))
        t.update(down_since=None, last_down=0.0, ups=0, event=None)

    def overlay(self, camera, frame):
        with self._lock:
            st = self._state.get(camera)
            if not st:
                return frame
            down = [t["box"] for t in st["tracks"] if t["event"] is not None]
        for x1, y1, x2, y2, _ in down:
            cv2.rectangle(frame, (x1, y1), (x2, y2), PURPLE, 2)
            _label(frame, "head down", x1, y1, PURPLE)
        return frame

    def caption(self, camera):
        n = self.head_down(camera)
        return ("{} head down".format(n), False) if n else None

    def head_down(self, camera):
        with self._lock:
            st = self._state.get(camera)
            return st["down"] if st else 0

    def forget(self, camera):
        with self._lock:
            self._state.pop(camera, None)


detector = HeadDown()
pipeline.add(detector)
