"""Evidence photos.

When somebody walks into a restricted area, or is caught using a phone, the
frame is saved twice: the whole picture with the boxes, the zone and a time
stamp burned in, and a close-up of each person. They live under
data/photos/<day>/ and are cleared after the number of days set in Settings.
"""

import logging
import os
import shutil
from datetime import datetime, timedelta

import cv2

import config

log = logging.getLogger(__name__)

PHOTO_DIR = os.path.join(config.DATA_DIR, "photos")
MAX_CROPS = 3
CROP_PAD = 0.18           # room around a person in the close-up
CROP_MIN_H = 320          # small people are scaled up so a face is visible

RED = (40, 40, 220)       # BGR
AMBER = (0, 170, 255)
GREY = (170, 170, 170)


def save(camera, kind, frame, targets, others=(), zone=None, colour=RED, ts=None):
    """Write the photos for one event and return their paths, relative to
    PHOTO_DIR: {"full": ..., "crops": [...], "ts": ...}. `targets` are the
    people the event is about, `others` everybody else in view."""
    ts = ts or datetime.now()
    day = ts.strftime("%Y-%m-%d")
    stem = "{}_{}_{}_{:03d}".format(camera, kind, ts.strftime("%H%M%S"),
                                    ts.microsecond // 1000)
    folder = os.path.join(PHOTO_DIR, day)
    os.makedirs(folder, exist_ok=True)

    name = (config.camera(camera) or {}).get("name", camera)
    full = annotate(frame, targets, others, zone, colour,
                    "{}  {}".format(name, ts.strftime("%Y-%m-%d %H:%M:%S")))
    out = {"full": _write(folder, day, stem + ".jpg", full), "crops": [],
           "ts": ts.isoformat(timespec="seconds")}

    biggest = sorted(targets, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
                     reverse=True)[:MAX_CROPS]
    for i, box in enumerate(biggest, 1):
        crop = _crop(frame, box)
        if crop is not None:
            out["crops"].append(_write(folder, day,
                                       "{}_p{}.jpg".format(stem, i), crop))
    out["crops"] = [c for c in out["crops"] if c]
    return out


def annotate(frame, targets, others=(), zone=None, colour=RED, stamp=""):
    img = frame.copy()
    if zone:
        draw_zone(img, zone, colour)
    for x1, y1, x2, y2, *_ in others:
        cv2.rectangle(img, (x1, y1), (x2, y2), GREY, 1)
    for x1, y1, x2, y2, *_ in targets:
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
    if stamp:
        cv2.rectangle(img, (0, img.shape[0] - 26), (img.shape[1], img.shape[0]),
                      (20, 20, 20), -1)
        cv2.putText(img, stamp, (8, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def draw_zone(img, zone, colour=RED):
    """Shade a normalised polygon [[x, y], ...] onto the image."""
    import numpy as np
    h, w = img.shape[:2]
    pts = np.array([[int(x * w), int(y * h)] for x, y in zone], dtype=np.int32)
    if len(pts) < 3:
        return img
    shade = img.copy()
    cv2.fillPoly(shade, [pts], colour)
    cv2.addWeighted(shade, 0.18, img, 0.82, 0, dst=img)
    cv2.polylines(img, [pts], True, colour, 2, cv2.LINE_AA)
    return img


def _crop(frame, box):
    x1, y1, x2, y2 = box[:4]
    h, w = frame.shape[:2]
    px, py = int((x2 - x1) * CROP_PAD), int((y2 - y1) * CROP_PAD)
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w, x2 + px), min(h, y2 + py)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.shape[0] < CROP_MIN_H:
        scale = CROP_MIN_H / float(crop.shape[0])
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
    return crop


def _write(folder, day, filename, img):
    path = os.path.join(folder, filename)
    if not cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 88]):
        log.warning("could not write %s", path)
        return ""
    return day + "/" + filename


def purge(keep_days):
    """Delete whole days older than keep_days."""
    if not os.path.isdir(PHOTO_DIR):
        return 0
    cutoff = (datetime.now() - timedelta(days=int(keep_days))).strftime("%Y-%m-%d")
    gone = 0
    for day in os.listdir(PHOTO_DIR):
        if len(day) == 10 and day < cutoff:
            shutil.rmtree(os.path.join(PHOTO_DIR, day), ignore_errors=True)
            gone += 1
    return gone


def usage():
    """(files, megabytes) on disk."""
    files = size = 0
    for root, _, names in os.walk(PHOTO_DIR):
        for n in names:
            files += 1
            try:
                size += os.path.getsize(os.path.join(root, n))
            except OSError:
                pass
    return files, round(size / 1e6, 1)
