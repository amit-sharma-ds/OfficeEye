"""Fire and smoke watch: every camera is looked at about once a minute by a
model trained only for fire and smoke (YOLOv8n on the D-Fire set: 21,000
pictures of real fires and smoke, indoors and out).

A model alone sees "fire" in an orange lamp, a red shirt or the sun on a
wall. So nothing is recorded until all of these hold:

1. The model is sure enough (Rules: sensitivity) and the spot is not tiny.
2. Fire looks like fire: bright red, orange or yellow (or, in a black and
   white night picture, very bright). Smoke is grey or white, not a colour.
3. It is seen again at the same spot in 3 checks, a few seconds apart.
4. It moves: flames flicker and smoke drifts between those checks. A thing
   that stays exactly the same is a lamp or a poster - that spot is learnt
   as "not fire" for that camera and never alerts again.
5. It is not in an area marked "ignore" on the camera (a TV, a window), and
   not a "fire" on a person (a red shirt) unless the model is very sure.

Then one record is made with photos, a phone alert and a note on every page.
"Not fire" on a record teaches that spot too, and keeps its picture in
data/fire/not_fire - pictures to train the model on later.
"""

import hashlib
import json
import logging
import os
import shutil
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

KIND = "fire_smoke"
MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fire_smoke_yolov8n.pt")
MODEL_URL = "https://huggingface.co/rabahdev/fire-smoke-yolov8n/resolve/main/best.pt"
# The file was checked (only torch and ultralytics parts inside) - a file that
# is not exactly this one is never loaded.
MODEL_SHA256 = "b91633799ceb052c814b4f8b77a37efc9a40f002d528df97d74463585fa4f28f"
FIRE_DIR = os.path.join(config.DATA_DIR, "fire")
SPOTS_FILE = os.path.join(FIRE_DIR, "not_fire_spots.json")
RED = (0, 0, 255)

# Model score needed, by sensitivity: (fire, smoke).
SURE = {"low": (0.55, 0.6), "medium": (0.45, 0.5), "high": (0.35, 0.4)}


def _iso(ts):
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _same_spot(a, b):
    """Two boxes (0-1 of the picture) on one fire: they overlap, or the middle
    of one is inside the other - flames change shape, smoke drifts."""
    if _iou(a, b) >= 0.15:
        return True
    for p, q in ((a, b), (b, a)):
        cx, cy = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
        if q[0] <= cx <= q[2] and q[1] <= cy <= q[3]:
            return True
    return False


def _inside(a, b):
    """Share of box a that lies inside box b."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    area = (a[2] - a[0]) * (a[3] - a[1])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) / area if area > 0 else 0.0


class FireWatch:
    SHARE = 0.3            # most of the detection worker's time spent on fire
    ROUTINE_SIZE = 480     # picture size for the routine look (640 for a suspect)
    HURRY_SCORE = 0.3      # a weaker guess than this is not looked at again sooner
    MIN_GAP = 0.3          # (s) between two fire checks, however quick
    STALE = 30.0           # a picture this old is not looked at
    FLOOR = 0.2            # model score worth following up at all
    MIN_AREA = 0.0015      # of the picture: smaller is a reflection far away
    CONFIRM = 3            # seen in this many checks ...
    WITHIN = 60.0          # ... within this many seconds
    HURRY = 3.0            # a camera with a suspect is looked at again this soon
    FOLLOW = 10.0          # ... and every this often while a fire goes on
    CHANGE = 4.0           # grey change in the spot between checks: it flickers
    FIRE_SHARE = 0.05      # of the spot: fire-coloured or flame-bright pixels
    SMOKE_SAT = 110        # smoke is greyish: average saturation below this
    SMOKE_MAX = 0.9        # of the picture: "smoke" this big is the camera's haze
    ON_PERSON = 0.6        # a "fire" this much inside a person ...
    ON_PERSON_SURE = 0.7   # ... needs this score (a red shirt)
    VERY_SURE = 0.85       # beats a learnt "not fire" spot
    GAP = 90.0             # not seen this long: the fire record is over
    PHOTO_EVERY = 15.0
    MAX_PHOTOS = 4
    MAX_SPOTS = 40         # learnt "not fire" spots per camera
    BG_SIZE = (320, 180)   # the learnt room (each camera), compared with the spot
    BG_RATE = 0.1          # how fast it learns, per check with nothing suspect
    VEIL = 0.8             # smoke: outlines behind it this much fainter than in the room
    PLAIN = 6.0            # ... unless the room there is plain (a bare wall)
    SMOKE_CHANGED = 0.35   # smoke: this share of the spot differs from the room
    SAT_UP = 8             # smoke adds no colour (a pink shirt does)
    NEW_FLAME = 0.025      # fire: this share of the spot is flame-coloured now and was not before
    SMOKE_CONFIRM = 4      # smoke needs more checks than fire ...
    SMOKE_SPAN = 20.0      # ... over at least this long (s): real smoke keeps coming
    PEOPLE_AGE = 120.0     # people seen by the main check this recently count
    SMOKE_ON_PERSON = 0.15 # smoke this much over a person is not trusted

    def __init__(self):
        self._lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._model = None
        self._state, self._error = "off", ""
        self._cams = {}
        self._spots = {}
        self._manager = self._pipeline = None
        self._thread = None
        self._cost = 0.45
        self._today, self._stopped, self._checks = "", {}, 0

    # ---------- the model ----------

    def _load_model(self):
        if not os.path.exists(MODEL):
            import requests
            log.info("fire: downloading the fire and smoke model")
            r = requests.get(MODEL_URL, timeout=120)
            r.raise_for_status()
            with open(MODEL + ".part", "wb") as f:
                f.write(r.content)
            os.replace(MODEL + ".part", MODEL)
        with open(MODEL, "rb") as f:
            if hashlib.sha256(f.read()).hexdigest() != MODEL_SHA256:
                raise RuntimeError("the fire model file is not the one that was checked - not loaded")
        from ultralytics import YOLO
        model = YOLO(MODEL, task="detect")
        self._names = {int(k): str(v).lower() for k, v in model.names.items()}
        return model

    def _predict(self, frame, size=640):
        """[(kind, (x1, y1, x2, y2) 0-1, score)] - on the detection worker when
        it runs (a restricted-area check still goes first), else right here."""
        out = {}
        done = threading.Event()

        def job():
            try:
                t0 = time.time()
                with self._model_lock:
                    r = self._model.predict(frame, imgsz=size, conf=self.FLOOR, verbose=False)[0]
                self._cost = 0.8 * self._cost + 0.2 * (time.time() - t0)
                h, w = frame.shape[:2]
                out["boxes"] = [(self._names.get(int(c), "fire"),
                                 (x1 / w, y1 / h, x2 / w, y2 / h), float(s))
                                for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.tolist(),
                                                                  r.boxes.cls.tolist(),
                                                                  r.boxes.conf.tolist())]
            finally:
                done.set()

        pipe = self._pipeline
        if pipe is not None and pipe.status().get("state") == "running":
            pipe.run_soon(job)
            done.wait(60)
        else:
            job()
        return out.get("boxes", [])

    # ---------- the checks ----------

    @staticmethod
    def _crop(frame, box):
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(box[0] * w)), max(0, int(box[1] * h))
        x2, y2 = min(w, int(box[2] * w) + 1), min(h, int(box[3] * h) + 1)
        return frame[y1:y2, x1:x2]

    @staticmethod
    def _grey(frame):
        return cv2.GaussianBlur(cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY),
                                (3, 3), 0)

    def _looks_right(self, kind, crop, colour):
        """Fire: bright red/orange/yellow (night: very bright). Smoke: no
        strong colour. False = the model's guess does not look like it."""
        if crop.size == 0:
            return False
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        if kind == "smoke":
            return not colour or float(s.mean()) < self.SMOKE_SAT
        if not colour:                       # infrared: flames are the brightest thing
            return np.count_nonzero(v >= 220) / v.size >= self.FIRE_SHARE
        flame = ((h <= 35) | (h >= 170)) & (s >= 60) & (v >= 170)
        core = (v >= 240) & (s < 60)         # the middle of a flame is nearly white
        return (np.count_nonzero(flame) / v.size >= self.FIRE_SHARE / 2 and
                np.count_nonzero(flame | core) / v.size >= self.FIRE_SHARE)

    @staticmethod
    def _ignored(camera, box):
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        for poly in (config.camera(camera) or {}).get("ignore") or []:
            if len(poly) >= 3 and cv2.pointPolygonTest(
                    np.array(poly, np.float32), (float(cx), float(cy)), False) >= 0:
                return True
        return False

    def _people(self, camera, frame):
        """People the main check saw here in the last few seconds (0-1 boxes)."""
        pipe = self._pipeline
        if pipe is None:
            return []
        try:
            latest, _ = pipe.latest()
        except Exception:                              # noqa: BLE001
            return []
        t = latest.get(camera)
        if not t or time.time() - t["at"] > self.PEOPLE_AGE or not t["size"]:
            return []
        w, h = t["size"]
        return [(b[0] / w, b[1] / h, b[2] / w, b[3] / h) for b in t["seen"].people]

    def _stop(self, why):
        day = datetime.now().strftime("%Y-%m-%d")
        if day != self._today:
            self._today, self._stopped, self._checks = day, {}, 0
        self._stopped[why] = self._stopped.get(why, 0) + 1

    def _room(self, frame):
        return cv2.resize(frame, self.BG_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)

    @staticmethod
    def _part(img, box):
        h, w = img.shape[:2]
        x1, y1 = int(box[0] * w), int(box[1] * h)
        x2, y2 = max(x1 + 2, int(np.ceil(box[2] * w))), max(y1 + 2, int(np.ceil(box[3] * h)))
        return img[y1:y2, x1:x2].astype(np.uint8)

    @staticmethod
    def _outlines(bgr):
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return float(np.mean(np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1))))

    @staticmethod
    def _flames(bgr, colour):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        if not colour:
            return v >= 220
        return (((h <= 35) | (h >= 170)) & (s >= 60) & (v >= 170)) | ((v >= 240) & (s < 60))

    def _veiled(self, room, frame, box):
        """Smoke lies over the room like a veil: what is behind it gets fainter,
        much of the spot changes and it adds no colour. Floor glare (there
        before) and people (more outlines, more colour) fail this."""
        now, then = self._part(self._room(frame), box), self._part(room, box)
        if now.size == 0:
            return False
        e_now, e_then = self._outlines(now), self._outlines(then)
        changed = float(np.mean(np.abs(now.astype(np.int16) - then).max(axis=2) > 12))
        sat = lambda x: float(cv2.cvtColor(x, cv2.COLOR_BGR2HSV)[..., 1].mean())
        fainter = e_now <= self.VEIL * e_then or (e_then < self.PLAIN and e_now < self.PLAIN * 1.5)
        return fainter and changed >= self.SMOKE_CHANGED and sat(now) <= sat(then) + self.SAT_UP

    def _new_flame(self, room, frame, box, colour):
        """Flames that were not there before - an orange bag or a lamp was."""
        now, then = self._part(self._room(frame), box), self._part(room, box)
        if now.size == 0:
            return False
        return float(np.mean(self._flames(now, colour) & ~self._flames(then, colour))) >= self.NEW_FLAME

    def judge(self, camera, frame, boxes, rules, room=None):
        """The model's boxes that pass checks 1, 2 and 5, and why the rest did not.
        With a camera, each spot is also compared with the learnt room."""
        sure_fire, sure_smoke = SURE.get(rules.get("sensitivity"), SURE["medium"])
        small = cv2.resize(frame, (64, 36))
        colour = float(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[..., 1].mean()) >= 12
        people = self._people(camera, frame) if camera else []
        spots = self._spots.get(camera, []) if camera else []
        keep, why = [], []
        for kind, box, score in boxes:
            area = (box[2] - box[0]) * (box[3] - box[1])
            reason = None
            if kind == "smoke" and not rules.get("smoke", True):
                reason = "smoke is switched off"
            elif score < (sure_smoke if kind == "smoke" else sure_fire):
                reason = "model not sure"
            elif area < self.MIN_AREA:
                reason = "too small"
            elif kind == "smoke" and area > self.SMOKE_MAX:
                reason = "whole picture"
            elif camera and self._ignored(camera, box):
                reason = "ignore area"
            elif not self._looks_right(kind, self._crop(frame, box), colour):
                reason = "wrong colour"
            elif kind == "fire" and score < self.ON_PERSON_SURE and \
                    any(_inside(box, p) >= self.ON_PERSON for p in people):
                reason = "on a person"
            elif kind == "smoke" and any(_inside(p, box) >= 0.5 or _inside(box, p) >= self.SMOKE_ON_PERSON
                                         for p in people):
                reason = "on a person"
            elif score < self.VERY_SURE and any(s["kind"] == kind and _iou(box, s["box"]) >= 0.4
                                                 for s in spots):
                reason = "learnt not fire"
            elif camera and room is None:
                reason = "learning the room"
            elif camera and kind == "smoke" and not self._veiled(room, frame, box):
                reason = "not like smoke"
            elif camera and kind == "fire" and not self._new_flame(room, frame, box, colour):
                reason = "there before"
            if reason:
                why.append((kind, box, score, reason))
            else:
                keep.append((kind, box, score))
        return keep, why

    def _change(self, a, b, box):
        """How much the spot changed between two pictures, above the change of
        the whole picture (the camera adjusting its brightness)."""
        d = cv2.absdiff(a, b).astype(np.float32)
        h, w = d.shape
        x1, y1 = int(box[0] * w), int(box[1] * h)
        x2, y2 = max(x1 + 1, int(np.ceil(box[2] * w))), max(y1 + 1, int(np.ceil(box[3] * h)))
        return float(d[y1:y2, x1:x2].mean()) - float(np.median(d))

    def check(self, camera, frame, ts, rules):
        with self._lock:
            st = self._cams[camera]
            close = st["incident"] is not None or st["hurry_until"] > ts
        # The routine look is at 480 px (this PC is slow); a suspect is looked
        # at in full, 640 px.
        boxes = self._predict(frame, 640 if close else self.ROUTINE_SIZE)
        keep, why = self.judge(camera, frame, boxes, rules, st.get("room"))
        grey = self._grey(frame)
        room = self._room(frame)
        with self._lock:
            st = self._cams[camera]
            st["checked"], st["boxes"] = ts, keep
            if not keep:                      # learn the room, slowly - never from a suspect
                old = st.get("room")
                st["room"] = room if old is None or old.shape != room.shape                     else old * (1.0 - self.BG_RATE) + room * self.BG_RATE
            self._checks += 1
            for _, _, _, reason in why:
                if reason != "model not sure":
                    self._stop(reason)
            if keep or any(score >= self.HURRY_SCORE for _, _, score in boxes):
                st["hurry_until"] = ts + self.WITHIN / 2     # a closer look at 640 px soon
            for kind, box, score in keep:
                st["hits"].append({"ts": ts, "kind": kind, "box": box, "score": score, "grey": grey})
            while st["hits"] and ts - st["hits"][0]["ts"] > self.WITHIN:
                st["hits"].popleft()
            found = None
            for kind, box, score in keep:
                same = [h for h in st["hits"] if h["kind"] == kind and _same_spot(h["box"], box)]
                seen = {h["ts"] for h in same}
                if len(seen) < (self.SMOKE_CONFIRM if kind == "smoke" else self.CONFIRM) or                         (kind == "smoke" and max(seen) - min(seen) < self.SMOKE_SPAN):
                    continue
                spot = (min(h["box"][0] for h in same), min(h["box"][1] for h in same),
                        max(h["box"][2] for h in same), max(h["box"][3] for h in same))
                # Most of the earlier checks must differ - one person walking past
                # a lamp is not a flicker.
                moved = float(np.median([self._change(h["grey"], grey, spot) for h in same if h["ts"] < ts]))
                if moved >= self.CHANGE:
                    if found is None or kind == "fire":
                        found = (kind, box, max(h["score"] for h in same))
                else:
                    # The same picture in every check: a lamp, a poster.
                    self._learn(camera, kind, spot, "never changed")
                    gone = {id(h) for h in same}
                    st["hits"] = deque(h for h in st["hits"] if id(h) not in gone)
                    self._stop("never changed")
            inc = st["incident"]
        if found:
            self._seen(camera, st, frame, ts, found)
        elif inc is not None and ts - inc["last"] > self.GAP:
            self._end(st)

    # ---------- records ----------

    def _seen(self, camera, st, frame, ts, found):
        kind, box, score = found
        h, w = frame.shape[:2]
        target = [(int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h), score)]
        now = datetime.fromtimestamp(ts)
        with self._lock:
            inc = st["incident"]
            new = inc is None
            if new:
                inc = st["incident"] = {"event": None, "started": ts, "last": ts, "photos": [],
                                        "photo_at": 0.0, "kind": kind, "score": score, "box": box}
            inc["last"], inc["box"] = ts, box
            inc["score"] = max(inc["score"], score)
            upgraded = kind == "fire" and inc["kind"] != "fire"
            if upgraded:
                inc["kind"] = "fire"
            shoot = len(inc["photos"]) < self.MAX_PHOTOS and ts - inc["photo_at"] >= self.PHOTO_EVERY
            if shoot:
                inc["photo_at"] = ts
        shot = photos.save(camera, "fire", frame, target, [], colour=RED, ts=now) if shoot else None
        with self._lock:
            if shot:
                inc["photos"].append(shot)
            meta = {"type": inc["kind"], "score": round(inc["score"], 2), "box": [round(v, 4) for v in box],
                    "started": _iso(inc["started"]), "last_seen": _iso(ts),
                    "seconds": int(ts - inc["started"]), "photos": list(inc["photos"])}
        if new:
            meta["ended"] = None
            eid = events.log(kind=KIND, camera=camera, label=kind, value=int(score * 100), meta=meta)
            with self._lock:
                inc["event"] = eid
            try:                              # the picture as the camera saw it, for training later
                os.makedirs(os.path.join(FIRE_DIR, "raw"), exist_ok=True)
                cv2.imwrite(os.path.join(FIRE_DIR, "raw", "{}.jpg".format(eid)), frame)
            except Exception:                 # noqa: BLE001
                log.exception("fire: raw picture not saved")
            log.warning("fire: %s at %s (score %.2f)", kind, camera, score)
        elif inc["event"]:
            events.update_meta(inc["event"], meta)     # "type" says fire once smoke turns to fire

    def _end(self, st):
        with self._lock:
            inc, st["incident"] = st["incident"], None
        if inc and inc["event"]:
            events.update_meta(inc["event"], {"ended": _iso(inc["last"]), "last_seen": _iso(inc["last"]),
                                              "seconds": int(inc["last"] - inc["started"])})

    def _learn(self, camera, kind, box, why):
        """Never alert for this spot again (unless the model is very sure)."""
        spots = self._spots.setdefault(camera, [])
        spots.append({"kind": kind, "box": [round(v, 4) for v in box], "why": why,
                      "at": datetime.now().isoformat(timespec="seconds")})
        del spots[:-self.MAX_SPOTS]
        self._save_spots()

    def _save_spots(self):
        try:
            os.makedirs(FIRE_DIR, exist_ok=True)
            with open(SPOTS_FILE + ".tmp", "w", encoding="utf-8") as f:
                json.dump(self._spots, f)
            os.replace(SPOTS_FILE + ".tmp", SPOTS_FILE)
        except OSError:
            log.exception("fire: spots not saved")

    def not_fire(self, event_id):
        """Somebody says a record was not a fire: learn its spot, keep its
        picture for training, and end it."""
        row = events.get(event_id)
        if not row or row.get("kind") != KIND:
            return False
        meta = row["meta"] if isinstance(row["meta"], dict) else json.loads(row["meta"] or "{}")
        with self._lock:
            if meta.get("box"):
                self._learn(row["camera"], meta.get("type") or row["label"], meta["box"], "marked not fire")
            st = self._cams.get(row["camera"])
            if st and st["incident"] and st["incident"]["event"] == event_id:
                st["incident"], st["hits"] = None, deque()
        raw = os.path.join(FIRE_DIR, "raw", "{}.jpg".format(event_id))
        if os.path.exists(raw):
            os.makedirs(os.path.join(FIRE_DIR, "not_fire"), exist_ok=True)
            shutil.move(raw, os.path.join(FIRE_DIR, "not_fire", "{}.jpg".format(event_id)))
        events.update_meta(event_id, {"not_fire": True, "ended": meta.get("ended") or meta.get("last_seen")})
        return True

    def forget_spots(self, camera=None):
        with self._lock:
            n = sum(len(v) for k, v in self._spots.items() if camera in (None, k))
            if camera:
                self._spots.pop(camera, None)
            else:
                self._spots = {}
            self._save_spots()
        return n

    # ---------- the loop ----------

    def start(self, manager, pipeline):
        if self._thread and self._thread.is_alive():
            return
        self._manager, self._pipeline = manager, pipeline
        try:
            with open(SPOTS_FILE, encoding="utf-8") as f:
                self._spots = json.load(f)
        except (OSError, ValueError):
            self._spots = {}
        self._thread = threading.Thread(target=self._run, name="fire", daemon=True)
        self._thread.start()

    def _next(self, rules, now):
        """The camera to look at now: one with a suspect or a fire first, else
        the one looked at longest ago."""
        off = set(rules.get("cameras_off") or [])
        status = self._manager.status()
        best, hurry = None, None
        with self._lock:
            for cam in status:
                st = self._cams.setdefault(cam, {"checked": 0.0, "seq": None, "hits": deque(),
                                                 "hurry_until": 0.0, "incident": None, "boxes": [], "room": None})
                if cam in off or status[cam].get("state") == "off":
                    continue
                gap = self.FOLLOW if st["incident"] else self.HURRY
                if (st["incident"] or st["hurry_until"] > now) and now - st["checked"] >= gap:
                    if hurry is None or st["checked"] < self._cams[hurry]["checked"]:
                        hurry = cam
                if best is None or st["checked"] < self._cams[best]["checked"]:
                    best = cam
            for cam in list(self._cams):
                if cam not in status:
                    self._cams.pop(cam)
        return hurry or best, hurry is not None

    def _run(self):
        try:
            self._model = self._load_model()
            self._state = "running"
            log.info("fire: watching with %s", os.path.basename(MODEL))
        except Exception as e:                         # noqa: BLE001
            self._state, self._error = "error", str(e)[:200]
            log.exception("fire: model not loaded")
            return
        while True:
            try:
                rules = config.get("fire")
                if not rules.get("enabled", True):
                    self._state = "off"
                    time.sleep(3)
                    continue
                self._state = "running"
                now = time.time()
                cam, hurry = self._next(rules, now)
                if cam is None:
                    time.sleep(2)
                    continue
                stream = self._manager.get(cam)
                frame, ts, seq = stream.latest() if stream else (None, 0, None)
                with self._lock:
                    st = self._cams[cam]
                    same = seq is not None and seq == st["seq"]
                    st["seq"] = seq
                if frame is None or now - (ts or 0) > self.STALE or same:
                    with self._lock:
                        st["checked"] = now          # nothing new to look at: the next one
                    time.sleep(0.2)
                    continue
                self.check(cam, frame, ts, rules)
                ended = [s for s in list(self._cams.values())
                         if s["incident"] and now - s["incident"]["last"] > self.GAP]
                for s in ended:
                    self._end(s)
                rest = self._cost * (1.0 / self.SHARE - 1.0)
                time.sleep(max(self.MIN_GAP, min(rest, 1.0) if hurry else rest))
            except Exception:                          # noqa: BLE001
                log.exception("fire watch")
                time.sleep(5)

    # ---------- read-outs ----------

    def open_count(self):
        with self._lock:
            return sum(1 for s in self._cams.values() if s["incident"])

    def report(self, rules):
        now = time.time()
        off = set(rules.get("cameras_off") or [])
        with self._lock:
            cams = {}
            for cam, st in self._cams.items():
                inc = st["incident"]
                state = ("off" if cam in off else inc["kind"] if inc
                         else "checking" if st["hits"] else "clear" if st["checked"] else "waiting")
                cams[cam] = {"state": state, "checked_ago": round(now - st["checked"]) if st["checked"] else None,
                             "event": inc["event"] if inc else None,
                             "spots": len(self._spots.get(cam, []))}
            watched = [c for c in cams if c not in off]
            cycle = self._cost / self.SHARE * max(1, len(watched))
            if self._today != datetime.now().strftime("%Y-%m-%d"):
                self._today, self._stopped, self._checks = datetime.now().strftime("%Y-%m-%d"), {}, 0
            return {"state": self._state, "error": self._error, "cameras": cams,
                    "every": round(cycle), "checks_today": self._checks,
                    "stopped_today": dict(self._stopped),
                    "spots": sum(len(v) for v in self._spots.values())}

    def test(self, frame):
        """A picture somebody sent: what the model sees and what the checks
        say - the 3-checks and flicker tests need a camera, not a photo."""
        if self._model is None:
            return None
        boxes = self._predict(frame)
        keep, why = self.judge(None, frame, boxes, config.get("fire"))
        return ([{"kind": k, "box": b, "score": round(s, 2), "pass": True, "why": ""} for k, b, s in keep] +
                [{"kind": k, "box": b, "score": round(s, 2), "pass": False, "why": r} for k, b, s, r in why])


watch = FireWatch()
