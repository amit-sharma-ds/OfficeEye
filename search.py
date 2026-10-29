"""Search the cameras: "kitne log", "phone", "car", "bottle", "kitab"...

Answers come from each camera's latest model check. The check that counts
people also finds every one of the model's 80 objects (detection.Vision),
so a search costs no model time. Its answer is only as fresh as those
checks - on a busy PC a quiet camera is looked at every few minutes - so
every camera in the answer says when it was looked at, and "Check now"
(Pipeline.look_all) has every camera looked at again within a minute or so.

What can be asked:
- people ("log", "aadmi", "staff")
- people on a phone ("phone", "mobile") - as the Phone use page counts
  them, so only in its watch hours
- vehicles ("gaadi"), cars, bikes, cycles
- any of the model's objects by its English name or a common Hindi /
  Hinglish word ("botal", "kitab", "kursi", "chhata"...)
- any of these with a colour: "laal t-shirt" (a person's chest), "neela
  bag", "white car". Read off the small picture kept from the last check
  (Track.jpg); a night (infrared) picture has no colour.
- anything else ("helmet", "printer") goes to the AI search (AISearch):
  YOLOE, a model that takes the typed words as what to find. It runs only
  when asked, one camera at a time on the detection worker - never before
  a restricted-area check - so about a minute for 30 cameras on this PC.
  Only the office PC runs it (the cloud copy asks the office), so it is
  not in requirements.txt. Its text model needs, installed by hand:
      py -3.12 -m pip install --user --no-deps git+https://github.com/ultralytics/CLIP.git
      py -3.12 -m pip install --user ftfy regex tqdm
  Not Ultralytics' own AutoUpdate: that tries to replace torch and numpy.
  The model files (yoloe-11s-seg.pt, mobileclip_blt.ts, 600 MB) download
  themselves on the first AI search.
"""

import base64
import logging
import os
import re
import threading
import time

import cv2
import numpy as np

import config

log = logging.getLogger(__name__)

# Words people type -> what to look for. A target is ("people",), ("phone",),
# ("vehicle", {categories}) or ("object", {model class names}).
CAR, BIKE, CYCLE = {"car"}, {"bike"}, {"cycle"}
VEHICLE_KIND = {"car": "car", "truck": "car", "bus": "car", "motorcycle": "bike", "bicycle": "cycle"}

WORDS = {
    ("person", "persons", "people", "log", "logo", "aadmi", "admi", "insaan", "insan", "banda",
     "bande", "staff", "employee", "employees", "worker", "workers", "human", "humans"): ("people",),
    ("phone", "phones", "mobile", "mobiles", "cell phone", "smartphone", "phone use",
     "phone chala", "mobile chala"): ("phone",),
    ("vehicle", "vehicles", "gaadi", "gadi", "gaadiyan", "gadiyan", "gaadiya", "gadiya",
     "vahan", "parking"): ("vehicle", CAR | BIKE | CYCLE),
    ("car", "cars"): ("vehicle", CAR),
    ("truck", "trucks", "tempo", "bus", "van"): ("vehicle", CAR),
    ("bike", "bikes", "motorcycle", "motorcycles", "motorbike", "scooter", "scooty",
     "activa", "motor cycle"): ("vehicle", BIKE),
    ("cycle", "cycles", "bicycle", "bicycles", "saikil", "cycle wala"): ("vehicle", CYCLE),
    ("bottle", "bottles", "bottel", "botal", "botel", "botle"): ("object", {"bottle"}),
    ("book", "books", "kitab", "kitaab", "kitabe", "kitaabein"): ("object", {"book"}),
    ("cup", "cups", "mug", "kap"): ("object", {"cup"}),
    ("glass", "wine glass"): ("object", {"wine glass", "cup"}),
    ("chair", "chairs", "kursi", "kursiyan"): ("object", {"chair"}),
    ("sofa", "couch"): ("object", {"couch"}),
    ("table", "tables", "mez", "dining table"): ("object", {"dining table"}),
    ("laptop", "laptops"): ("object", {"laptop"}),
    ("tv", "television", "monitor", "screen"): ("object", {"tv"}),
    ("keyboard", "keyboards"): ("object", {"keyboard"}),
    ("mouse",): ("object", {"mouse"}),
    ("remote",): ("object", {"remote"}),
    ("bag", "bags", "thaila", "jhola"): ("object", {"backpack", "handbag", "suitcase"}),
    ("backpack", "bagpack", "school bag", "college bag"): ("object", {"backpack"}),
    ("handbag", "purse", "ladies bag"): ("object", {"handbag"}),
    ("suitcase", "attachi", "trolley"): ("object", {"suitcase"}),
    ("umbrella", "chhata", "chata", "chhatri", "chatri"): ("object", {"umbrella"}),
    ("clock", "ghadi", "ghari", "wall clock"): ("object", {"clock"}),
    ("scissors", "kainchi", "kenchi"): ("object", {"scissors"}),
    ("knife", "chaku", "chakoo"): ("object", {"knife"}),
    ("spoon", "chammach", "chamach"): ("object", {"spoon"}),
    ("fork",): ("object", {"fork"}),
    ("bowl", "katori"): ("object", {"bowl"}),
    ("plant", "plants", "gamla", "paudha", "pot plant", "potted plant"): ("object", {"potted plant"}),
    ("vase", "flower pot", "guldasta"): ("object", {"vase"}),
    ("fridge", "refrigerator"): ("object", {"refrigerator"}),
    ("microwave",): ("object", {"microwave"}),
    ("oven",): ("object", {"oven"}),
    ("toaster",): ("object", {"toaster"}),
    ("sink", "washbasin", "wash basin"): ("object", {"sink"}),
    ("bed", "palang"): ("object", {"bed"}),
    ("bench", "benches"): ("object", {"bench"}),
    ("tie",): ("object", {"tie"}),
    ("teddy", "teddy bear"): ("object", {"teddy bear"}),
    ("toothbrush",): ("object", {"toothbrush"}),
    ("hair dryer", "hair drier", "dryer"): ("object", {"hair drier"}),
    ("dog", "dogs", "kutta", "kutte"): ("object", {"dog"}),
    ("cat", "cats", "billi"): ("object", {"cat"}),
    ("bird", "birds", "chidiya", "kabutar"): ("object", {"bird"}),
    ("cow", "gaay"): ("object", {"cow"}),
    ("ball", "gend", "sports ball", "football"): ("object", {"sports ball"}),
    ("kite", "patang"): ("object", {"kite"}),
    ("banana", "kela"): ("object", {"banana"}),
    ("apple", "seb"): ("object", {"apple"}),
    ("orange", "santra"): ("object", {"orange"}),
    ("carrot", "gajar"): ("object", {"carrot"}),
    ("pizza",): ("object", {"pizza"}),
    ("cake",): ("object", {"cake"}),
    ("donut", "doughnut"): ("object", {"donut"}),
    ("sandwich",): ("object", {"sandwich"}),
    ("traffic light",): ("object", {"traffic light"}),
}
# Words a question is made of, left out before looking for what is asked.
FILLER = re.compile(r"\b(kitne|kitni|kitna|kahan|kaha|kidhar|kon|kaun|kis|kya|hai|hain|he|h|me|mein|"
                    r"ka|ki|ke|ko|se|par|pe|abhi|kar|rahe|raha|rahi|use|how|many|much|where|is|are|"
                    r"the|a|an|in|on|of|any|show|find|search|dikhao|batao|camera|cameras|live|"
                    r"with|wearing|pehna|pehne|pehni|pahna|pahne|pahni|hua|hue|huyi|rang|color|colour)\b")
LABELS = {"people": "people", "phone": "people on a phone"}

COLOURS = {
    ("red", "laal", "lal"): "red",
    ("blue", "neela", "nila", "neeli", "nili", "neele", "nile", "navy"): "blue",
    ("green", "hara", "hari", "hare"): "green",
    ("yellow", "peela", "pila", "peeli", "pili", "peele", "pile"): "yellow",
    ("black", "kala", "kaala", "kaali", "kali", "kale", "kaale"): "black",
    ("white", "safed", "safaid", "sufaid", "safet"): "white",
    ("grey", "gray", "sleti"): "grey",
    ("orange", "narangi", "kesari", "kesariya"): "orange",
    ("pink", "gulabi"): "pink",
    ("purple", "baingani", "jamuni", "violet"): "purple",
    ("brown", "bhura", "bhoora", "bhure"): "brown",
}
COLOUR_WORDS = {w: c for words, c in COLOURS.items() for w in words}
# Said about a person: "laal t-shirt", "kaali jacket wala".
CLOTHES = re.compile(r"\b(shirt|shirts|tshirt|t shirt|kurta|kurti|top|jacket|hoodie|sweater|kapde|kapda|"
                     r"dress|suit|saree|wala|wali|wale|ladka|ladki|aurat|uniform)\b")
COLOUR_SHARE = 0.3          # this much of a person's chest (of a thing's middle) in the colour


_PHRASES = {}


def _phrases(names):
    """Every word or phrase with its target, longest first: "cell phone"
    before "phone", "hot dog" before "dog". The model's own class names
    are in it too, so any of its 80 objects can be typed."""
    key = tuple(names)
    if key not in _PHRASES:
        out = [(w, t) for words, t in WORDS.items() for w in words]
        have = {w for w, _ in out}
        for name in names:
            for w in (name, name + "s"):
                if w not in have:
                    out.append((w, ("object", {name})))
        _PHRASES[key] = sorted(out, key=lambda p: -len(p[0]))
    return _PHRASES[key]


def _norm(text):
    q = " " + re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()) + " "
    return re.sub(r"\s+", " ", q)


def parse(text, names=()):
    """What a search asks for: (target, label, colour). target None: not one
    of the things the model always looks for - the AI search takes it."""
    q = _norm(text)
    colour = None
    for word, c in COLOUR_WORDS.items():
        if " {} ".format(word) in q:
            colour, q = c, q.replace(" {} ".format(word), " ")
            break
    target, label = None, None
    phrases = _phrases(names)
    for phrase, t in phrases:
        if " {} ".format(phrase) in q:
            target, label = t, _label(phrase, t)
            break
    rest = FILLER.sub(" ", CLOTHES.sub(" ", q)).split()
    if target is None and len(rest) == 1 and len(rest[0]) >= 4:      # "bottl" - near enough?
        for phrase, t in phrases:
            if phrase.startswith(rest[0]) or rest[0].startswith(phrase):
                target, label = t, _label(phrase, t)
                break
    if target is None and colour and (CLOTHES.search(q) or not rest):  # "laal t-shirt (wala)"
        target, label = ("people",), "people"
    if target is not None and colour:
        label = "people in {}".format(colour) if target[0] in ("people", "phone") else \
            "{} {}".format(colour, label)
    return target, label, colour


def ai_words(text):
    """What the AI search is asked to find: the typed words without the
    question around them, a colour put in English ("laal helmet" -> "red helmet")."""
    q = _norm(text)
    for word, c in COLOUR_WORDS.items():
        q = q.replace(" {} ".format(word), " {} ".format(c))
    return " ".join(FILLER.sub(" ", q).split())[:60]


def _label(phrase, target):
    if target[0] in LABELS:
        return LABELS[target[0]]
    if target[0] == "vehicle":
        cats = target[1]
        return "vehicles" if len(cats) > 1 else {"car": "cars", "bike": "bikes", "cycle": "cycles"}[next(iter(cats))]
    return phrase if len(target[1]) > 1 else next(iter(target[1]))


def _found(target, camera, seen, phone_use):
    """The boxes of what is searched for in one camera's last check."""
    kind = target[0]
    if kind == "people":
        return list(seen.people)
    if kind == "phone":
        return phone_use.using(camera) if phone_use else []
    if kind == "vehicle":
        return [b for b in seen.vehicles if VEHICLE_KIND.get(b[5]) in target[1]]
    return [b for b in seen.objects if b[5] in target[1]]


# ---------------------------------------------------------------- colours

def _decode(jpg):
    return cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR) if jpg else None


def black_and_white(img):
    """An infrared (night) picture: there is no colour in it at all."""
    hsv = cv2.cvtColor(cv2.resize(img, (64, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean()) < 12


def colour_shares(img, box, person):
    """{colour: share} of a person's chest, or of the middle of a thing.
    box: 0-1 fractions of the picture."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box[:4]
    bw, bh = x2 - x1, y2 - y1
    if person:      # below the head, above the legs
        x1, x2, y1, y2 = x1 + bw * .25, x2 - bw * .25, y1 + bh * .25, y1 + bh * .6
    else:
        x1, x2, y1, y2 = x1 + bw * .2, x2 - bw * .2, y1 + bh * .2, y2 - bh * .2
    top, left = int(y1 * h), int(x1 * w)
    crop = img[top:max(top + 1, int(y2 * h)), left:max(left + 1, int(x2 * w))]
    if crop.shape[0] * crop.shape[1] < 20:
        return {}
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int32)
    H, S, V = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    lab = np.full(len(H), "grey", dtype="<U6")
    plain = S < 45
    lab[plain & (V < 60)] = "black"
    lab[plain & (V >= 150)] = "white"       # office light makes white look grey
    lab[~plain & (V < 50)] = "black"
    c = ~plain & (V >= 50)
    for m, name in (((H < 8) | (H >= 170), "red"), ((H >= 8) & (H < 20), "orange"),
                    ((H >= 20) & (H < 34), "yellow"), ((H >= 34) & (H < 85), "green"),
                    ((H >= 85) & (H < 130), "blue"), ((H >= 130) & (H < 150), "purple"),
                    ((H >= 150) & (H < 170), "pink")):
        lab[c & m] = name
    lab[c & ((H < 20) | (H >= 170)) & (V < 110)] = "brown"                 # dark red / orange
    lab[c & ((H < 8) | (H >= 160)) & (S < 110) & (V >= 170)] = "pink"      # light red
    names, counts = np.unique(lab, return_counts=True)
    return {str(n): float(k) / len(lab) for n, k in zip(names, counts)}


def has_colour(img, box, colour, person):
    shares = colour_shares(img, box, person)
    top = sorted(shares, key=shares.get, reverse=True)[:2]
    return shares.get(colour, 0.0) >= COLOUR_SHARE and colour in top


def _picture(jpg):
    """The picture of a check, for the page to draw the boxes on."""
    return "data:image/jpeg;base64," + base64.b64encode(jpg).decode("ascii") if jpg else None


# ---------------------------------------------------------------- the answer

def _checking(latest, look_all_at, now):
    live = [v for v in latest.values() if v.get("live", True)]       # an offline camera is not waited for
    looked = sum(1 for v in live if v["at"] >= look_all_at)
    if look_all_at and now - look_all_at < 300 and looked < len(live):
        return {"done": looked, "total": len(live), "since": round(now - look_all_at)}
    return None


def answer(text, latest, look_all_at, cameras, names, phone_use):
    """The answer to one search. latest: Pipeline.latest(); cameras: the
    camera list from the settings; names: the model's class names."""
    target, label, colour = parse(text, names)
    now = time.time()
    order = {c["id"]: i for i, c in enumerate(cameras)}
    base = {"query": text, "label": label, "colour": colour,
            "checking": _checking(latest, look_all_at, now)}
    if target is None:
        words = ai_words(text)
        return dict(base, ok=False, ai=bool(words), words=words,
                    message="Not one of the things always looked for - the AI search can look for it")
    out, total, grey = [], 0, []
    person = target[0] in ("people", "phone")
    for cam in cameras:
        info = latest.get(cam["id"])
        if not info:
            continue
        w, h = info["size"]
        boxes = [[round(b[0] / w, 4), round(b[1] / h, 4), round(b[2] / w, 4), round(b[3] / h, 4),
                  round(float(b[4]), 2)] for b in _found(target, cam["id"], info["seen"], phone_use)]
        if colour and boxes:
            img = _decode(info.get("jpg"))
            if img is None or black_and_white(img):
                grey.append(cam.get("name", cam["id"]))
                boxes = []
            else:
                boxes = [b for b in boxes if has_colour(img, b, colour, person)]
        total += len(boxes)
        entry = {"id": cam["id"], "name": cam.get("name", cam["id"]), "group": cam.get("group", config.COUNT),
                 "count": len(boxes), "checked_ago": round(now - info["at"]),
                 "offline": not info.get("live", True), "boxes": boxes}
        if boxes:
            entry["image"] = _picture(info.get("jpg"))
        out.append(entry)
    out.sort(key=lambda c: (-c["count"], order.get(c["id"], 0)))
    hits = [c for c in out if c["count"]]
    return dict(base, ok=True, total=total, cameras=out, found_in=len(hits), looked_at=len(out),
                no_colour=grey,
                oldest=max((c["checked_ago"] for c in out if not c["offline"]), default=None))


# ---------------------------------------------------------------- AI search

class AISearch:
    """Anything typed, found by YOLOE - an open-vocabulary model: the words
    become what it looks for. Only when asked, one camera's last-check
    picture at a time on the detection worker (Pipeline.run_soon), so a
    restricted-area check never waits behind it."""

    MODEL = "yoloe-11s-seg.pt"
    CONF = 0.2
    IMGSZ = 640

    def __init__(self):
        self._lock = threading.Lock()           # the search going on
        self._model_lock = threading.Lock()     # the model: its words, a picture looked at
        self._model, self._words, self._device = None, None, "cpu"
        self._run = None

    def start(self, text, words, latest, cameras, pipeline):
        frames = [(c, latest[c["id"]]) for c in cameras
                  if c["id"] in latest and latest[c["id"]].get("jpg")]
        run = {"query": text, "words": words, "started": time.time(), "total": len(frames),
               "done": 0, "found": {}, "error": "", "ready": False}
        pipeline.drop_jobs()                    # a search still going is dropped
        with self._lock:
            self._run = run
        threading.Thread(target=self._prepare, args=(run, frames, pipeline),
                         name="ai-search", daemon=True).start()
        return self.public()

    def _prepare(self, run, frames, pipeline):
        """Load the model (once) and give it the words - off the detection
        worker, which would otherwise stand still meanwhile."""
        try:
            with self._model_lock:
                if self._model is None:
                    import torch
                    from ultralytics import YOLOE
                    path = os.path.join(config.BASE_DIR, self.MODEL)
                    self._model = YOLOE(path if os.path.exists(path) else self.MODEL)
                    self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
                    log.info("AI search model ready: %s", self.MODEL)
                if self._words != run["words"]:
                    self._model.set_classes([run["words"]], self._model.get_text_pe([run["words"]]))
                    self._words = run["words"]
        except Exception:                                 # noqa: BLE001
            log.exception("AI search model failed")
            with self._lock:
                run["error"] = "The AI search model could not be loaded"
            return
        with self._lock:
            if self._run is not run:
                return
            run["ready"] = True
        for cam, info in frames:
            pipeline.run_soon(lambda cam=cam, info=info: self._one(run, cam, info))

    def _one(self, run, cam, info):
        with self._lock:
            if self._run is not run:
                return                                    # a newer search took over
        img = _decode(info.get("jpg"))
        boxes = []
        if img is not None:
            with self._model_lock:
                if self._words != run["words"]:
                    return
                r = self._model.predict(img, imgsz=self.IMGSZ, conf=self.CONF, verbose=False,
                                        device=self._device)[0]
            h, w = img.shape[:2]
            if r.boxes is not None:
                for (x1, y1, x2, y2), s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    boxes.append([round(float(x1) / w, 4), round(float(y1) / h, 4),
                                  round(float(x2) / w, 4), round(float(y2) / h, 4), round(float(s), 2)])
        with self._lock:
            run["done"] += 1
            if boxes:
                run["found"][cam["id"]] = {"id": cam["id"], "name": cam.get("name", cam["id"]),
                                           "group": cam.get("group", config.COUNT), "count": len(boxes),
                                           "at": info["at"], "offline": not info.get("live", True),
                                           "boxes": boxes, "image": _picture(info.get("jpg"))}

    def public(self, have=()):
        """The search going on, or the last one. have: cameras the page shows
        already - their pictures are not sent again."""
        now = time.time()
        with self._lock:
            run = self._run
            if run is None:
                return {"ok": True, "running": False, "query": None, "cameras": []}
            out = []
            for c in sorted(run["found"].values(), key=lambda c: (-c["count"], c["name"])):
                item = {k: v for k, v in c.items() if k not in ("at", "image")}
                item["checked_ago"] = round(now - c["at"])
                if c["id"] not in have:
                    item["image"] = c["image"]
                out.append(item)
            return {"ok": not run["error"], "query": run["query"], "words": run["words"],
                    "message": run["error"], "ready": run["ready"],
                    "running": not run["error"] and run["done"] < run["total"],
                    "done": run["done"], "total": run["total"], "seconds": round(now - run["started"]),
                    "cameras": out, "found_in": len(out), "total_found": sum(c["count"] for c in out)}


ai = AISearch()
