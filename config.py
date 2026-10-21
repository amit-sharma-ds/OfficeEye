"""Settings store. Everything lives in data/config.json so the app can be
reconfigured from the browser without touching the code or restarting."""

import json
import os
import threading
from copy import deepcopy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CONFIG_PATH = os.path.join(DATA_DIR, "config.json")

# What a camera is there for. Counting cameras feed the Person count page;
# restricted cameras watch an area nobody should be in and raise alerts.
COUNT = "count"
RESTRICTED = "restricted"
GROUPS = (COUNT, RESTRICTED)

DEFAULTS = {
    # As many cameras as the office needs. Each one is
    # {"id": "cam1", "name": "Camera 1", "group": "count",
    #  "ip": "...", "url": "rtsp://...",
    #  "zone": [[x, y], ...],      restricted only: 0-1 fractions of the picture
    #  "ignore": [[[x, y], ...]]}  areas never counted: posters, TVs, mannequins
    "cameras": [],
    "office": {
        "open": "09:30",
        "close": "18:30",
        "late_after": "09:45",
        "timezone": "Asia/Kolkata",
    },
    "detection": {
        "enabled": True,
        "confidence": 0.3,        # low on purpose: ceiling cameras see people
        "model": "yolo26s.pt",    # at odd angles; the count smoother and ignore
                                  # areas keep false boxes out
        "device": "auto",         # auto = the NVIDIA GPU when there is one
        "imgsz": 480,             # model input: 320 fast, 480 balanced, 640 sharpest
        "phone": True,            # look for phones as well as people
        "count_refresh": 30,      # re-check a quiet counting camera at least this often (s)
        "restricted_fps": 4,      # most checks per second for one restricted camera
    },
    "photos": {
        "keep_days": 30,
    },
}

_lock = threading.RLock()
_cache = None


def _merge(base, extra):
    out = deepcopy(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load(force=False):
    global _cache
    with _lock:
        if _cache is not None and not force:
            return deepcopy(_cache)
        raw = {}
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
                    raw = json.load(fh)
            except (OSError, ValueError):
                raw = {}
        _cache = _merge(DEFAULTS, _migrate(raw))
        return deepcopy(_cache)


def _migrate(raw):
    """Older versions stored a fixed camera_in / camera_out pair."""
    if not raw.get("cameras") and ("camera_in" in raw or "camera_out" in raw):
        cams = []
        for key in ("camera_in", "camera_out"):
            old = raw.pop(key, None) or {}
            if old.get("url"):
                n = len(cams) + 1
                cams.append({"id": "cam{}".format(n), "name": "Camera {}".format(n),
                             "ip": old.get("ip", ""), "url": old["url"]})
        raw["cameras"] = cams
    raw.pop("camera_in", None)
    raw.pop("camera_out", None)
    for cam in raw.get("cameras") or []:
        cam.setdefault("group", COUNT)
    # "fps" was a per-camera detection rate; the scheduler decides that now.
    (raw.get("detection") or {}).pop("fps", None)
    return raw


def save(cfg):
    global _cache
    with _lock:
        merged = _merge(DEFAULTS, cfg)
        os.makedirs(DATA_DIR, exist_ok=True)
        tmp = CONFIG_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(merged, fh, indent=2)
        os.replace(tmp, CONFIG_PATH)
        _cache = merged
        return deepcopy(merged)


def update(section, values):
    """Patch one section (e.g. 'office') and write the file."""
    with _lock:
        cfg = load()
        cfg[section] = _merge(cfg.get(section, {}), values)
        return save(cfg)


def get(section):
    return load().get(section, {})


# ---------------------------------------------------------------- cameras

def cameras(group=None):
    items = load().get("cameras", [])
    if group is None:
        return items
    return [c for c in items if c.get("group", COUNT) == group]


def camera_group(cam_id):
    cam = camera(cam_id)
    return (cam or {}).get("group", COUNT)


def set_cameras(items):
    with _lock:
        cfg = load()
        cfg["cameras"] = list(items)
        return save(cfg)


def camera(cam_id):
    for cam in cameras():
        if cam.get("id") == cam_id:
            return cam
    return None


def next_camera_id():
    used = {c.get("id") for c in cameras()}
    n = 1
    while f"cam{n}" in used:
        n += 1
    return f"cam{n}"
