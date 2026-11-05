"""Seat monitor (the "Monitoring agent" page): who sits at each seat, from
when to when, with a photo and a short clip of every arrival - so each
seat's history can be looked back at.

Cameras are switched on and off for it in the page's Settings tab. On a
camera that is on, the seats find themselves: a place where somebody stays
put for LEARN_AFTER seconds becomes a seat ("Seat 4", renamed or moved on
the page if wanted). Seats can be drawn by hand too: a box around the chair
and where a person sitting on it shows. Then, after every model check of a
camera that is on (no extra model time):

1. Each person goes to the seat that holds the most of them (or that they
   cover most of) - one person a seat.
2. Somebody at a seat for "sit_after" seconds is a sitting. From the first
   sighting that camera decodes every frame for a few seconds and a clip of
   a few seconds is written: the seat and around it, as H.264 MP4 that
   any browser plays. Somebody only walking past is dropped, clip and all.
3. The sitting is written down once it is sure: when it began, a photo of
   the person (and a close-up), the clip and the colour of their clothes.
   It ends when the seat has been seen empty for "gone_after" seconds, with
   a photo of the last look at them.

Nobody is recognised - there is no face recognition in this app. Who sat is
what the photos and the clip show; the colour of the clothes helps to tell
one sitting from the next. Clips stay on the office PC (in the photo folder,
cleared with the photos); the cloud view gets the photos only.
"""

import logging
import os
import threading
import time
from datetime import datetime, timedelta
from fractions import Fraction

import av
import cv2

import cloud
import config
import events
import photos
import search
from cameras import manager
from detection import Detector, pipeline

log = logging.getLogger(__name__)

KIND = "seat_sitting"
SECTION = "seats"               # config: {camera: [{"id", "name", "box": [x1, y1, x2, y2]}]}
MAX_SEATS = 40                  # on one camera

TEAL = (160, 140, 20)           # BGR
EMPTY, PENDING, SITTING = "empty", "pending", "sitting"


RULES = "seat_rules"            # config: the page's Settings tab
DEFAULT_RULES = {
    "cameras": None,            # the cameras watched; None: those with seats
    "sit_after": 20,            # at a seat this many seconds: sitting, not walking past
    "gone_after": 60,           # seen empty this long: gone (a missed look or two is not leaving)
    "auto_seats": True,         # a seat is added where somebody stays put, none drawn needed
    "clips": True,
    "clip_seconds": 10,
}
LIMITS = {"sit_after": (5, 600), "gone_after": (15, 1800), "clip_seconds": (4, 30)}


def seats_of(camera):
    return list(config.get(SECTION).get(camera) or [])


def rules():
    r = dict(DEFAULT_RULES, **config.get(RULES))
    if r["cameras"] is None:
        r["cameras"] = [cam for cam, items in config.get(SECTION).items() if items]
    return r


def clean_rules(body, cameras):
    """The Settings tab's answer, checked: known counting cameras only,
    numbers within their limits."""
    out = {}
    if isinstance(body.get("cameras"), list):
        out["cameras"] = [c for c in dict.fromkeys(str(c) for c in body["cameras"]) if c in cameras]
    for key, (lo, hi) in LIMITS.items():
        if key in body:
            try:
                out[key] = int(min(max(float(body[key]), lo), hi))
            except (TypeError, ValueError):
                pass
    for key in ("auto_seats", "clips"):
        if key in body:
            out[key] = bool(body[key])
    return out


def clean_seats(items):
    """What the page sent, made safe to keep: boxes inside the picture,
    a name each, ids that stay the same while a seat is kept."""
    out = []
    for s in items or []:
        if not isinstance(s, dict) or len(out) >= MAX_SEATS:
            continue
        try:
            x1, y1, x2, y2 = [min(max(float(v), 0.0), 1.0) for v in s.get("box")]
        except (TypeError, ValueError):
            continue
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x2 - x1 < 0.02 or y2 - y1 < 0.02:
            continue
        name = " ".join(str(s.get("name") or "").split())[:40] or "Seat {}".format(len(out) + 1)
        out.append({"id": str(s.get("id") or "")[:20], "name": name,
                    "box": [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]})
    # A kept seat keeps its id (its history goes by it); new ones get free ids.
    used = set()
    for s in out:
        if s["id"] in used:
            s["id"] = ""
        used.add(s["id"])
    for s in out:
        if not s["id"]:
            s["id"] = next("s{}".format(k) for k in range(1, 1000) if "s{}".format(k) not in used)
            used.add(s["id"])
    return out


def _px(box, w, h):
    return (box[0] * w, box[1] * h, box[2] * w, box[3] * h)


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _iou(a, b):
    inter = _inter(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _inter(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def assign(seats, people, w, h, share=0.4):
    """{seat id: person box}: each person to the seat holding the most of
    them, each seat to one person. A person counts at a seat when at least
    `share` of them is inside it, or they cover that much of it."""
    pairs = []
    for i, b in enumerate(people):
        for s in seats:
            r = _px(s["box"], w, h)
            inter = _inter(b, r)
            if not inter:
                continue
            k = max(inter / max(1.0, _area(b)), inter / max(1.0, _area(r)))
            if k >= share:
                pairs.append((k, i, s["id"]))
    out, used = {}, set()
    for k, i, sid in sorted(pairs, reverse=True):
        if sid not in out and i not in used:
            out[sid] = people[i]
            used.add(i)
    return out


def clothes(frame, box):
    """The colour of a person's clothes ("white", "blue"...) - only when it
    is clear, else "". From a ceiling camera a box takes in the desk, hair
    and a neighbour too, and a wrong colour would mislead more than none."""
    h, w = frame.shape[:2]
    nb = [box[0] / w, box[1] / h, box[2] / w, box[3] / h]
    if search.black_and_white(frame):
        return ""
    dark, shares = search.colour_shares(frame, nb, True)
    if dark is None:
        return ""
    best = max(shares, key=shares.get) if shares else ""
    if best and best != "grey" and shares[best] >= 0.45 and search.has_colour(frame, nb, best, True):
        return best
    if dark >= 0.7 and max((v for k, v in shares.items() if k in search.CHROMA), default=0) < 0.3:
        return "black"
    return ""


# ---------------------------------------------------------------- clips

class Clip:
    """One arrival filmed: the camera's newest frames, a few a second, cut
    to the seat and around it. Written to a temporary name and renamed when
    done, so a page never gets half a file."""

    def __init__(self, camera, seat, size, started, seconds):
        self.camera, self.seat_id = camera, seat["id"]
        w, h = size
        x1, y1, x2, y2 = _px(seat["box"], w, h)
        mx, my = (x2 - x1) * 0.6, (y2 - y1) * 0.6           # around the seat too
        x1, y1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
        x2, y2 = int(min(w, x2 + mx)), int(min(h, y2 + my))
        self.crop = (x1, y1, x2, y2)
        self.seat_px = [int(v) for v in _px(seat["box"], w, h)]
        scale = min(1.0, Recorder.WIDTH / max(1, x2 - x1))
        self.size = (max(2, int((x2 - x1) * scale) // 2 * 2), max(2, int((y2 - y1) * scale) // 2 * 2))
        ts = datetime.fromtimestamp(started)
        self.day = ts.strftime("%Y-%m-%d")
        self.name = "{}_seat-{}_{}_{:03d}.mp4".format(camera, self.seat_id, ts.strftime("%H%M%S"),
                                                     ts.microsecond // 1000)
        self.path = os.path.join(photos.PHOTO_DIR, self.day, self.name)
        self.started, self.until = started, started + seconds
        self.out = self.stream = None
        self.seq, self.frames, self.t0 = -1, 0, None
        self.done = False
        self.keep = None            # True: a sitting, False: walked past, None: not decided yet

    @property
    def rel(self):
        return self.day + "/" + self.name

    def add(self, frame, ts):
        x1, y1, x2, y2 = self.crop
        img = frame[y1:y2, x1:x2]
        if img.size == 0:
            return
        img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
        sx, sy = self.size[0] / max(1, x2 - x1), self.size[1] / max(1, y2 - y1)
        a, b, c, d = self.seat_px
        cv2.rectangle(img, (int((a - x1) * sx), int((b - y1) * sy)), (int((c - x1) * sx), int((d - y1) * sy)),
                      TEAL, 1, cv2.LINE_AA)
        cv2.putText(img, datetime.fromtimestamp(ts).strftime("%H:%M:%S"), (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        if self.out is None:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self.out = av.open(self.path + ".part", "w", format="mp4", options={"movflags": "+faststart"})
            self.stream = self.out.add_stream("libx264", rate=25)
            self.stream.width, self.stream.height = self.size
            self.stream.pix_fmt = "yuv420p"
            self.stream.codec_context.time_base = Fraction(1, 1000)
            self.stream.options = {"preset": "ultrafast", "crf": "28", "threads": "1"}
            self.t0 = ts
        vf = av.VideoFrame.from_ndarray(img, format="bgr24").reformat(format="yuv420p")
        vf.pts, vf.time_base = int((ts - self.t0) * 1000), Fraction(1, 1000)
        for p in self.stream.encode(vf):
            self.out.mux(p)
        self.frames += 1

    def finish(self):
        """Close the file; True when there is a clip to keep."""
        self.done = True
        if self.out is None:
            return False
        try:
            for p in self.stream.encode():
                self.out.mux(p)
            self.out.close()
            os.replace(self.path + ".part", self.path)
            return self.frames >= 2
        except Exception:                                 # noqa: BLE001
            log.exception("seat clip failed")
            self.drop()
            return False

    def discard(self):
        """Not a sitting after all (walked past): no clip, written or not."""
        self.keep = False
        if self.done:
            self.drop()

    def drop(self):
        for p in (self.path + ".part", self.path):
            try:
                os.remove(p)
            except OSError:
                pass


class Recorder:
    """Films the arrivals. While a clip of a camera is going on, the camera
    decodes every frame (as a restricted one does), without being checked
    any more often; a few frames a second go into the clip."""

    EVERY = 0.2                 # a frame this often (5 a second)
    WIDTH = 480                 # clips at most this wide

    def __init__(self):
        self._lock = threading.Lock()
        self._clips = []
        self._wake = threading.Event()
        threading.Thread(target=self._run, name="seat-clips", daemon=True).start()

    def start(self, camera, seat, size, ts, seconds):
        clip = Clip(camera, seat, size, ts, seconds)
        stream = manager.get(camera)
        if stream is None:
            return None
        with self._lock:
            self._clips.append(clip)
        stream.set_full_rate(True)
        self._wake.set()
        return clip

    def _restore(self, camera):
        with self._lock:
            still = any(c.camera == camera for c in self._clips)
        stream = manager.get(camera)
        if stream is not None and not still:
            stream.set_full_rate(config.camera_group(camera) in (config.RESTRICTED, config.WASHROOM))

    def _run(self):
        while True:
            with self._lock:
                clips = list(self._clips)
            if not clips:
                self._wake.wait(5.0)
                self._wake.clear()
                continue
            now = time.time()
            for clip in clips:
                stream = manager.get(clip.camera)
                if stream is not None and now < clip.until:
                    frame, ts, seq = stream.latest()
                    if frame is not None and seq != clip.seq and ts >= clip.started - 1:
                        clip.seq = seq
                        try:
                            clip.add(frame, ts)
                        except Exception:                 # noqa: BLE001
                            log.exception("seat clip failed")
                            clip.until = 0
                if stream is None or now >= clip.until or clip.keep is False:
                    with self._lock:
                        self._clips.remove(clip)
                    kept = clip.keep is not False and clip.finish()
                    if not kept or clip.keep is False:
                        clip.drop()
                    self._restore(clip.camera)
            time.sleep(self.EVERY)


# ---------------------------------------------------------------- the seats

class _Seat:
    __slots__ = ("state", "first", "seen_at", "checked_at", "box", "frame", "clip", "event", "shots",
                 "noted_at")

    def __init__(self):
        self.state = EMPTY
        self.first = self.seen_at = self.checked_at = self.noted_at = 0.0
        self.box = self.frame = self.clip = None
        self.event = None
        self.shots = []


class SeatMonitor(Detector):
    key = "seats"
    title = "Seat monitor"
    groups = (config.COUNT,)

    PENDING_DROP = 15.0         # not seen again this soon: only walked past
    NOTE_EVERY = 300.0          # a long sitting writes how long it has been, now and then
    LEARN_AFTER = 90.0          # somebody this long in one place that is no seat yet: a new seat
    LEARN_LOOKS = 3             # ...seen there in this many looks at least
    LEARN_MISSES = 2            # a place empty in this many looks in a row is forgotten

    def __init__(self):
        super().__init__()
        self._state = {}        # (camera, seat id) -> _Seat
        self._spots = {}        # camera -> places somebody stays that are no seat yet
        self.recorder = Recorder()
        threading.Thread(target=self._close_open, name="seat-close", daemon=True).start()

    def _st(self, camera, sid):
        st = self._state.get((camera, sid))
        if st is None:
            st = self._state[(camera, sid)] = _Seat()
        return st

    # ---------- after a model check (pipeline worker) ----------

    def process(self, camera, frame, ts, seen):
        r = rules()
        on = camera in r["cameras"]
        if not on and not any(k[0] == camera for k in self._state):
            with self._lock:
                self._spots.pop(camera, None)
            return None
        seats = seats_of(camera) if on else []       # turned off: every sitting there ends
        h, w = frame.shape[:2]
        taken = assign(seats, seen.people, w, h)
        born = {}
        if on and r["auto_seats"]:
            free = [b for b in seen.people if all(b is not t for t in taken.values())]
            new = self._learn(camera, free, ts, w, h, seats)
            if new:
                kept = clean_seats(seats + [s for s, _ in new])
                config.update(SECTION, {camera: kept})
                for s, (_, first) in zip(kept[len(seats):], new):
                    born[s["id"]] = first            # the sitting began when they came
                log.info("seat monitor: %d seat(s) found on %s", len(kept) - len(seats), camera)
                seats = kept
                taken = assign(seats, seen.people, w, h)
        begin, sat, ends, notes = [], [], [], []
        with self._lock:
            ids = {s["id"] for s in seats}
            for key in [k for k in self._state if k[0] == camera and k[1] not in ids]:
                st = self._state.pop(key)            # seat removed on the page, camera turned off
                if st.state == SITTING:
                    ends.append((st.event, st.first, st.seen_at, None, None, list(st.shots), None))
                elif st.clip is not None:
                    st.clip.discard()
            for seat in seats:
                st = self._st(camera, seat["id"])
                st.checked_at = ts
                box = taken.get(seat["id"])
                if box is not None:
                    st.seen_at, st.box, st.frame = ts, box, frame
                    if st.state == EMPTY:
                        st.state, st.first, st.event, st.shots = PENDING, born.get(seat["id"], ts), None, []
                        begin.append((seat, st))
                    elif st.state == PENDING and ts - st.first >= r["sit_after"]:
                        st.state, st.noted_at = SITTING, ts
                        sat.append((seat, st, box))
                    elif st.state == SITTING and st.event and ts - st.noted_at >= self.NOTE_EVERY:
                        st.noted_at = ts
                        notes.append((st.event, st.first, ts))
                elif st.state == PENDING and ts - st.seen_at >= self.PENDING_DROP:
                    if st.clip is not None:
                        st.clip.discard()
                    self._state[(camera, seat["id"])] = _Seat()
                elif st.state == SITTING and ts - st.seen_at >= r["gone_after"]:
                    ends.append((st.event, st.first, st.seen_at, st.frame, st.box, list(st.shots), seat))
                    self._state[(camera, seat["id"])] = _Seat()
        for seat, st in begin:                        # outside the lock: opens a file
            if not r["clips"]:
                continue
            clip = self.recorder.start(camera, seat, (w, h), ts, r["clip_seconds"])
            with self._lock:
                if st.state != EMPTY:
                    st.clip = clip
                elif clip is not None:
                    clip.discard()
        for seat, st, box in sat:
            self._begin(camera, seat, st, frame, box, ts)
        for eid, first, at in notes:
            events.update_meta(eid, {"last": _iso(at), "seconds": int(at - first)})
        for end in ends:
            self._end(camera, *end)
        return None

    def _learn(self, camera, people, ts, w, h, seats):
        """Places where somebody stays put that are no seat yet. Once
        somebody has been at one for LEARN_AFTER seconds it becomes a seat:
        [(seat, when they came)]. A drawn seat is never needed this way."""
        with self._lock:
            spots = self._spots.setdefault(camera, [])
            hit = []
            for b in people:
                best, k = None, 0.5
                for sp in spots:
                    j = _iou(sp["box"], b)
                    if j >= k and all(sp is not x for x in hit):
                        best, k = sp, j
                if best is None:
                    best = {"box": [float(v) for v in b[:4]], "first": ts, "looks": 0, "missed": 0}
                    spots.append(best)
                else:                                 # a little sway is the same place
                    best["box"] = [a * 0.7 + c * 0.3 for a, c in zip(best["box"], b[:4])]
                best["looks"] += 1
                best["missed"] = 0
                hit.append(best)
            for sp in spots:
                if all(sp is not x for x in hit):
                    sp["missed"] += 1
            spots[:] = [sp for sp in spots if sp["missed"] < self.LEARN_MISSES]
            ready = [sp for sp in spots if sp["looks"] >= self.LEARN_LOOKS and ts - sp["first"] >= self.LEARN_AFTER]
            for sp in ready:
                spots.remove(sp)
        new, have = [], [s["box"] for s in seats]
        numbers = [int(s["name"][5:]) for s in seats if s["name"][:5] == "Seat " and s["name"][5:].isdigit()]
        n = max(numbers + [len(seats)])
        for sp in ready:
            x1, y1, x2, y2 = sp["box"]
            mx, my = (x2 - x1) * 0.1, (y2 - y1) * 0.05
            box = [max(0.0, x1 - mx) / w, max(0.0, y1 - my) / h, min(w, x2 + mx) / w, min(h, y2 + my) / h]
            if any(_iou(box, o) > 0.3 for o in have):
                continue                              # mostly a seat already
            n += 1
            have.append(box)
            new.append(({"id": "", "name": "Seat {}".format(n), "box": box}, sp["first"]))
        return new

    def learning(self):
        """{camera: places being watched to become seats}, for the page."""
        with self._lock:
            return {cam: len(spots) for cam, spots in self._spots.items() if spots}

    def _begin(self, camera, seat, st, frame, box, ts):
        clip = st.clip
        if clip is not None:
            clip.keep = True
        shot = photos.save(camera, "seat", frame, [box], zone=_poly(seat["box"]), colour=TEAL,
                           ts=datetime.fromtimestamp(ts))
        meta = {"seat": seat["id"], "seat_name": seat["name"], "photos": [shot],
                "start": _iso(st.first), "last": _iso(ts), "seconds": int(ts - st.first),
                "clothes": clothes(frame, box)}
        if clip is not None:
            meta["clip"] = clip.rel
        eid = self.emit(camera, KIND, 0, seat["name"], meta)
        with self._lock:
            st.event, st.shots = eid, [shot]

    def _end(self, camera, eid, first, last, frame, box, shots, seat):
        if eid is None:
            return
        patch = {"end": _iso(last), "last": _iso(last), "seconds": int(last - first)}
        if frame is not None and box is not None:
            patch["photos"] = shots + [photos.save(camera, "seat", frame, [box], zone=_poly(seat["box"]),
                                                   colour=TEAL, ts=datetime.fromtimestamp(last))]
        events.update_meta(eid, patch)

    def _close_open(self):
        """Sittings left open by a restart of the app end at their last look."""
        if cloud.CLOUD:
            return                  # the office PC's rows: it closes them itself
        try:
            since = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
            for r in events.between([KIND], since, "9999"):
                m = r["meta"]
                if not m.get("end"):
                    events.update_meta(r["id"], {"end": m.get("last") or r["ts"]})
        except Exception:                                 # noqa: BLE001
            log.exception("closing open sittings failed")

    def busy(self, camera):
        """Somebody just came to a seat: look again soon, moving or not -
        sitting down or walking past is only told by a second look."""
        with self._lock:
            return any(k[0] == camera and st.state == PENDING for k, st in self._state.items())

    def waiting(self, camera):
        """Somebody at a seat was not seen in the last look: the routine
        second look (gone, or only missed) comes before the quiet cameras."""
        with self._lock:
            return any(k[0] == camera and st.state == SITTING and st.seen_at < st.checked_at
                       for k, st in self._state.items())

    # ---------- read-outs ----------

    def now(self):
        """{camera: {seat id: {"state", "since", "seen_ago", "event"}}} for the page."""
        t = time.time()
        out = {}
        with self._lock:
            for (cam, sid), st in self._state.items():
                if st.state == EMPTY:
                    continue
                out.setdefault(cam, {})[sid] = {"state": st.state, "since": _iso(st.first),
                                                "seen_ago": round(t - st.seen_at), "event": st.event}
        return out

    def forget(self, camera):
        with self._lock:
            self._spots.pop(camera, None)
            for key in [k for k in self._state if k[0] == camera]:
                st = self._state.pop(key)
                if st.clip is not None and st.state != SITTING:
                    st.clip.discard()


def _iso(ts):
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def _poly(box):
    x1, y1, x2, y2 = box
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def suggest(camera):
    """Seat boxes from what the last check of a camera saw: each chair, and
    each person not on one of them - a start that is then fixed by hand."""
    latest, _ = pipeline.latest()
    info = latest.get(camera)
    if not info:
        return []
    w, h = info["size"]
    seen = info["seen"]
    chairs = [b for b in seen.objects if b[5] in ("chair", "couch", "bench")]
    boxes = [list(b[:4]) for b in chairs]
    for p in seen.people:
        if not any(_inter(p, c) > 0.3 * _area(p) for c in boxes):
            boxes.append(list(p[:4]))
    out = []
    for b in sorted(boxes, key=lambda b: (b[1] + b[3], b[0])):
        bw, bh = b[2] - b[0], b[3] - b[1]           # a little room: a person shows above the chair
        x1, y1 = max(0, b[0] - bw * 0.1), max(0, b[1] - bh * 0.35)
        x2, y2 = min(w, b[2] + bw * 0.1), min(h, b[3] + bh * 0.05)
        if any(_inter((x1, y1, x2, y2), _px(o["box"], w, h)) > 0.5 * _area((x1, y1, x2, y2)) for o in out):
            continue
        out.append({"id": "", "name": "", "box": [x1 / w, y1 / h, x2 / w, y2 / h]})
    return clean_seats(out)


detector = SeatMonitor()
pipeline.add(detector)
