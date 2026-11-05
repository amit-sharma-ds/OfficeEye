"""Search the cameras: "kitne log", "phone", "car", "bottle", "kitab"...

Answers come from each camera's latest model check. The check that counts
people also finds every one of the model's 80 objects (detection.Vision),
so a search costs no model time. Its answer is only as fresh as those
checks - on a busy PC a quiet camera is looked at every few minutes - so
every camera in the answer says when it was looked at, and "Check now"
(Pipeline.look_all) has every camera looked at again within a minute or so.

What can be asked:
- people ("log", "aadmi", "staff", "ladki")
- phones ("phone", "mobile"), in a hand or on a desk
- people on a phone ("phone use", "phone pe") - as the Phone use page
  counts them, so only in its watch hours
- vehicles ("gaadi"), cars, bikes, cycles
- any of the model's objects by its English name or a common Hindi /
  Hinglish word ("botal", "kitab", "kursi", "chhata"...)
- any of these with a colour: "laal t-shirt" (a person's chest), "neela
  bag", "white car". Read off the small picture kept from the last check
  (Track.jpg); a night (infrared) picture has no colour.
- anything else ("helmet", "blue jeans", "man with a cap") goes to the AI
  search (AISearch): YOLOE, a model that takes the typed words as what to
  find. It runs only when asked, one camera at a time on the detection
  worker - never before a restricted-area check - so a minute or two for
  30 cameras on this PC.

A search on pictures older than a minute has those cameras looked at again
first (Pipeline.look_all), and the answer follows as they come in.
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

PEOPLE = ("person", "persons", "people", "log", "logo", "aadmi", "admi", "insaan", "insan", "banda",
          "bande", "staff", "employee", "employees", "worker", "workers", "human", "humans", "man", "men",
          "woman", "women", "boy", "boys", "girl", "girls", "guy", "guys", "lady", "ladies", "gents",
          "ladka", "ladke", "ladki", "ladkiyan", "aurat", "auraten", "visitor", "visitors")
WORDS = {
    PEOPLE: ("people",),
    ("phone", "phones", "mobile", "mobiles", "cell phone", "cellphone", "smartphone", "smartphones",
     "fone"): ("phones",),
    ("phone use", "on phone", "on the phone", "using phone", "using mobile", "phone chala",
     "mobile chala", "phone dekh", "mobile dekh", "phone pe", "mobile pe", "phone par", "mobile par",
     "talking on phone", "phone chalate", "mobile chalate"): ("phone",),
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
# Words that do not change what is looked for: "kitne log hai office me",
# "bottle on the desk". Any other word the model has no name for ("man
# with helmet", "blue jeans") sends the search to the AI.
PLACE = {"office", "room", "floor", "area", "andar", "bahar", "inside", "outside", "here", "there",
         "yahan", "yaha", "wahan", "waha", "today", "aaj", "now", "total", "number", "count", "sab",
         "sabhi", "saare", "sare", "all", "every", "please", "plz", "pls", "mujhe", "hame", "hamein",
         "dekho", "dekhna", "dikha", "dikhe", "dikhai", "dikhaye", "dikhaiye", "do", "de", "koi", "kuch",
         "some", "photo", "picture", "pic", "image", "look", "wala", "wali", "wale"}
IGNORE = PLACE | {"sitting", "standing", "walking", "baitha", "baithe", "baithi", "khada", "khade",
                  "khadi", "chalta", "chalte", "working", "kaam", "near", "paas", "side", "left",
                  "right", "big", "small", "bada", "badi", "bade", "chhota", "chhoti", "chhote",
                  "new", "naya", "nayi", "old", "purana", "purani", "water", "desk"}
# The AI search's text model knows English: common Hindi words put in it.
ENGLISH = {"topi": "cap", "chashma": "glasses", "chasma": "glasses", "joota": "shoes", "joote": "shoes",
           "jute": "shoes", "chappal": "slippers", "darwaza": "door", "darwaja": "door", "khidki": "window",
           "pankha": "fan", "deewar": "wall", "diwar": "wall", "kachra": "garbage", "kooda": "garbage",
           "kuda": "garbage", "dabba": "box", "dibba": "box", "paani": "water", "pani": "water",
           "kagaz": "paper", "kagaj": "paper", "almari": "cupboard", "almirah": "cupboard",
           "takiya": "pillow", "kambal": "blanket", "seedhi": "stairs", "sidhi": "stairs", "taala": "lock",
           "tala": "lock", "chabi": "key", "chaabi": "key", "batua": "wallet", "dupatta": "scarf",
           "aag": "fire", "dhuan": "smoke", "dhuaan": "smoke", "dastane": "gloves", "tiffin": "lunch box",
           "chai": "tea cup", "khana": "food", "jhanda": "flag", "gamchha": "towel", "tauliya": "towel",
           "balti": "bucket", "jhadu": "broom", "pochha": "mop", "sigret": "cigarette", "cigrette": "cigarette"}
LABELS = {"people": "people", "phone": "people on a phone", "phones": "phones"}

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
COLOUR_SHARE = 0.35         # this much of a person's clothes (of a thing's middle) in the colour


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
            if len(phrase) >= 4 and abs(len(phrase) - len(rest[0])) <= 2 and \
                    (phrase.startswith(rest[0]) or rest[0].startswith(phrase)):
                target, label = t, _label(phrase, t)
                rest = []
                break
    known = _known_words(phrases)
    if any(w not in known and w not in IGNORE and not w.isdigit() for w in rest):
        return None, None, colour           # more than the model has a name for - the AI search
    if target is None and colour and (CLOTHES.search(q) or not rest):  # "laal t-shirt (wala)"
        target, label = ("people",), "people"
    if target is not None and colour:
        label = "people in {}".format(colour) if target[0] in ("people", "phone") else \
            "{} {}".format(colour, label)
    return target, label, colour


def _known_words(phrases):
    return {w for p, _ in phrases for w in p.split()}


def ai_words(text):
    """What the AI search is asked to find: the typed words without the
    question around them, in English ("laal topi wala" -> "red cap"), and
    without the person when there is more ("man with helmet" -> "helmet")."""
    q = _norm(text)
    for word, c in COLOUR_WORDS.items():
        q = q.replace(" {} ".format(word), " {} ".format(c))
    words = [ENGLISH.get(w, w) for w in FILLER.sub(" ", q).split()
             if w not in PLACE and not w.isdigit()]
    things = [w for w in words if w not in PEOPLE]
    if any(w not in COLOUR_WORDS for w in things):
        words = things
    return " ".join(words)[:60]


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
    if kind == "phones":
        return list(seen.phones)
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


CHROMA = ("red", "orange", "yellow", "green", "blue", "purple", "pink", "brown")
BLACK_SHARE = 0.5           # this much of a person's upper body dark: black clothes
GREY_SHARE = 0.7            # "grey" only when nothing else: shade turns everything greyish


def colour_shares(img, box, person):
    """(dark, {colour: share}) of a person's upper body, or of the middle of a
    thing. dark: the share of it too dark to have a colour - black cloth, but
    also hair and shadow, which a camera on the ceiling sees a lot of. The
    colours are shares of the rest, skin left out (it reads as orange or red).
    Tuned on 86 office people sorted by hand (2026-11-04). box: 0-1 fractions."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box[:4]
    bw, bh = x2 - x1, y2 - y1
    if person:      # the upper body - from above, the head is in it too
        x1, x2, y1, y2 = x1 + bw * .2, x2 - bw * .2, y1 + bh * .2, y1 + bh * .65
    else:
        x1, x2, y1, y2 = x1 + bw * .2, x2 - bw * .2, y1 + bh * .2, y2 - bh * .2
    top, left = int(y1 * h), int(x1 * w)
    crop = img[top:max(top + 1, int(y2 * h)), left:max(left + 1, int(x2 * w))]
    if crop.shape[0] * crop.shape[1] < 20:
        return None, {}
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int32)
    H, S, V = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    dark = float(np.count_nonzero((V < 75) & (S < 100))) / len(H)
    lab = np.full(len(H), "grey", dtype="<U6")
    plain = S < 45
    lab[plain & (V >= 145)] = "white"       # office light makes white look grey
    c = ~plain
    for m, name in (((H < 8) | (H >= 170), "red"), ((H >= 8) & (H < 20), "orange"),
                    ((H >= 20) & (H < 34), "yellow"), ((H >= 34) & (H < 85), "green"),
                    ((H >= 85) & (H < 128), "blue"), ((H >= 128) & (H < 160), "purple"),
                    ((H >= 160) & (H < 170), "pink")):
        lab[c & m] = name
    lab[c & ((H < 20) | (H >= 170)) & (V < 110)] = "brown"                 # dark red / orange
    lab[c & ((H < 8) | (H >= 160)) & (S < 110) & (V >= 170)] = "pink"      # light red
    if person:
        lab[(H < 20) & (S >= 30) & (S < 150) & (V >= 60) & (lab != "brown")] = "skin"
    lab[(V < 50) | (plain & (V < 60))] = "black"
    rest = lab[(lab != "black") & (lab != "skin")]
    if len(rest) < 0.1 * len(lab):
        return dark, {}
    names, counts = np.unique(rest, return_counts=True)
    return dark, {str(n): float(k) / len(rest) for n, k in zip(names, counts)}


def has_colour(img, box, colour, person):
    dark, shares = colour_shares(img, box, person)
    if dark is None:
        return False
    if colour == "black":
        return dark >= BLACK_SHARE and max((v for k, v in shares.items() if k in CHROMA), default=0) < 0.4
    top = sorted(shares, key=shares.get, reverse=True)[:2]
    if colour == "grey":
        return shares.get("grey", 0.0) >= GREY_SHARE and top[0] == "grey"
    share = shares.get(colour, 0.0)
    if colour == "red" and share >= 0.1 and "brown" in top:        # maroon is "laal" too
        share += shares["brown"]
        top.append("red")
    return share >= COLOUR_SHARE and colour in top


def _picture(jpg):
    """The picture of a check, for the page to draw the boxes on."""
    return "data:image/jpeg;base64," + base64.b64encode(jpg).decode("ascii") if jpg else None


def _small_jpg(frame, width=640):
    h, w = frame.shape[:2]
    if w > width:
        frame = cv2.resize(frame, (width, max(1, int(h * width / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buf.tobytes() if ok else None


# ---------------------------------------------------------------- the answer

FRESH = 60          # a search on pictures older than this has them looked at again
LOOK_AGAIN = 120    # ...at most this often


def _checking(latest, look, now):
    """The fresh looks still going: look is {camera: asked at}."""
    asked = {cam: t for cam, t in look.items()
             if now - t < 300 and latest.get(cam, {}).get("live")}     # an offline camera is not waited for
    looked = sum(1 for cam, t in asked.items() if latest[cam]["at"] >= t)
    if looked < len(asked):
        return {"done": looked, "total": len(asked), "since": round(now - max(asked.values()))}
    return None


def stale(latest, look):
    """True when a search would answer from old pictures that nobody asked
    to be looked at again lately: the app then looks at them first."""
    now = time.time()
    return any(v.get("live") and now - v["at"] > FRESH and now - look.get(cam, 0) > LOOK_AGAIN
               for cam, v in latest.items())


def answer(text, latest, look, cameras, names, phone_use):
    """The answer to one search. latest, look: Pipeline.latest(); cameras:
    the camera list from the settings; names: the model's class names."""
    target, label, colour = parse(text, names)
    now = time.time()
    order = {c["id"]: i for i, c in enumerate(cameras)}
    base = {"query": text, "label": label, "colour": colour,
            "checking": _checking(latest, look, now)}
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
    become what it looks for. Only when asked, one camera at a time on the
    detection worker (Pipeline.run_soon), so a restricted-area check never
    waits behind it: 2-3 s a camera on this PC. Each camera's newest frame
    is looked at, not its last check's picture, which can be minutes old."""

    MODEL = "yoloe-11s-seg.pt"
    # On 14 office cameras (2026-11-04) every bottle it found scored 0.16 or
    # more, its wrong finds ("helmet", "umbrella") 0.13 or less. The frames
    # are 640 wide: a smaller size loses the small things (bottle 0.2 -> 0.11).
    CONF = 0.15
    IMGSZ = 640
    # Cameras print the time along the top and their name along the bottom;
    # the AI took the "12" of a time for a helmet. Nothing wholly in these
    # strips counts.
    TEXT_BAND = 0.07
    KEEP_WORDS = 50     # searches whose words stay turned into the model's terms

    def __init__(self):
        self._lock = threading.Lock()           # the search going on
        self._model_lock = threading.Lock()     # the model: its words, a picture looked at
        self._model, self._words, self._device = None, None, "cpu"
        self._pe = {}                           # words -> the text model's output for them
        self._run = None

    def warm(self):
        """Load the model before the search is typed (the search box got
        focus): its text model alone takes a minute to load on this PC."""
        if self._model is None:
            threading.Thread(target=self._warm, name="ai-warm", daemon=True).start()

    def _warm(self):
        try:
            with self._model_lock:
                self._load()
                self._text_pe("person")
        except Exception:                                 # noqa: BLE001
            log.exception("AI search model failed")

    def _load(self):
        if self._model is None:
            import torch
            from ultralytics import YOLOE
            path = os.path.join(config.BASE_DIR, self.MODEL)
            self._model = YOLOE(path if os.path.exists(path) else self.MODEL)
            self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
            log.info("AI search model ready: %s", self.MODEL)

    def _text_pe(self, words):
        if words not in self._pe:
            if len(self._pe) >= self.KEEP_WORDS:
                self._pe.pop(next(iter(self._pe)))
            self._pe[words] = self._model.get_text_pe([words])
        return self._pe[words]

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
                self._load()
                if self._words != run["words"]:
                    self._model.set_classes([run["words"]], self._text_pe(run["words"]))
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
            pipeline.run_soon(lambda cam=cam, info=info: self._one(run, cam, info, pipeline))

    def _one(self, run, cam, info, pipeline):
        with self._lock:
            if self._run is not run:
                return                                    # a newer search took over
        img, jpg, at = pipeline.frame_now(cam["id"]), None, time.time()
        if img is None:                                   # not coming now: the last check's picture
            jpg, at = info.get("jpg"), info["at"]
            img = _decode(jpg)
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
                    if y2 <= h * self.TEXT_BAND or y1 >= h * (1 - self.TEXT_BAND):
                        continue                          # the camera's own time / name text
                    boxes.append([round(float(x1) / w, 4), round(float(y1) / h, 4),
                                  round(float(x2) / w, 4), round(float(y2) / h, 4), round(float(s), 2)])
        with self._lock:
            run["done"] += 1
            if boxes:
                run["found"][cam["id"]] = {"id": cam["id"], "name": cam.get("name", cam["id"]),
                                           "group": cam.get("group", config.COUNT), "count": len(boxes),
                                           "at": at, "offline": jpg is not None,
                                           "boxes": boxes, "image": _picture(jpg or _small_jpg(img))}

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
