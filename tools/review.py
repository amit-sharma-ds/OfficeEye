"""Label review page: check and fix the pre-labels before any training.

    py -3.12 tools/review.py            then open http://localhost:8001

Shows every collected frame with its boxes as a contact sheet; click one to
fix it (draw, move, resize, delete, person/phone). "Correct" saves the label
file and marks the frame checked in data/train/review.json. Only checked
frames go into training (tools/make_dataset.py).

A separate little server on its own port - the camera app is not touched.
"""

import argparse
import glob
import json
import os
import threading
from datetime import datetime

from flask import Flask, abort, jsonify, request, send_from_directory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(DATA, "train")
IMAGES = os.path.join(OUT, "images")
LABELS = os.path.join(OUT, "labels")
AUTO = os.path.join(OUT, "auto")
REVIEW = os.path.join(OUT, "review.json")
HERE = os.path.dirname(os.path.abspath(__file__))

CLASSES = (0, 67)                     # COCO person, cell phone

app = Flask(__name__)
_lock = threading.Lock()


def _read_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _review():
    return _read_json(REVIEW, {})


def _write_review(rev):
    tmp = REVIEW + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rev, f, indent=0, sort_keys=True)
    os.replace(tmp, REVIEW)


def _frame_info():
    rows = {}
    path = os.path.join(OUT, "frames.jsonl")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows[r["file"][:-4]] = r
    return rows


def _labels(stem):
    """Boxes of a frame, with the pre-label score where the box is untouched."""
    auto = {}
    for b in _read_json(os.path.join(AUTO, stem + ".json"), {}).get("boxes", []):
        auto[(b["cls"], f"{b['cx']:.5f}", f"{b['cy']:.5f}", f"{b['w']:.5f}", f"{b['h']:.5f}")] = b["score"]
    boxes = []
    try:
        with open(os.path.join(LABELS, stem + ".txt"), encoding="utf-8") as f:
            for line in f:
                p = line.split()
                if len(p) != 5:
                    continue
                k = int(p[0])
                cx, cy, w, h = (float(v) for v in p[1:])
                box = {"cls": k, "cx": cx, "cy": cy, "w": w, "h": h}
                key = (k, f"{cx:.5f}", f"{cy:.5f}", f"{w:.5f}", f"{h:.5f}")
                if key in auto:
                    box["score"] = auto[key]
                boxes.append(box)
    except OSError:
        return None
    return boxes


@app.get("/")
def index():
    return send_from_directory(HERE, "review.html", max_age=0)


@app.get("/img/<name>")
def img(name):
    return send_from_directory(IMAGES, name, max_age=3600)


@app.get("/api/frames")
def frames():
    cfg = _read_json(os.path.join(DATA, "config.json"), {})
    cams = {c["id"]: {"name": c.get("name", c["id"]), "ignore": c.get("ignore", [])}
            for c in cfg.get("cameras", [])}
    info = _frame_info()
    rev = _review()
    out = []
    for path in sorted(glob.glob(os.path.join(IMAGES, "*.jpg"))):
        stem = os.path.basename(path)[:-4]
        boxes = _labels(stem)
        if boxes is None:
            continue                               # not pre-labelled yet
        r = info.get(stem, {})
        cam = r.get("cam") or stem.split("_")[0]
        out.append({
            "name": stem, "cam": cam, "time": r.get("time", ""),
            "status": rev.get(stem, {}).get("status", ""),
            "boxes": boxes,
        })
    out.sort(key=lambda f: (int("0" + "".join(ch for ch in f["cam"] if ch.isdigit())), f["time"]))
    return jsonify({"frames": out, "cameras": cams})


@app.post("/api/label/<stem>")
def save(stem):
    if not os.path.exists(os.path.join(IMAGES, stem + ".jpg")):
        abort(404)
    body = request.get_json(force=True)
    status = body.get("status", "ok")
    if status not in ("ok", "excluded", ""):
        abort(400)
    with _lock:
        if "boxes" in body:
            lines = []
            for b in body["boxes"]:
                k = int(b["cls"])
                if k not in CLASSES:
                    abort(400)
                cx, cy, w, h = (min(max(float(b[v]), 0.0), 1.0) for v in ("cx", "cy", "w", "h"))
                if w < 0.003 or h < 0.003:
                    continue
                lines.append(f"{k} {cx:.5f} {cy:.5f} {w:.5f} {h:.5f}\n")
            tmp = os.path.join(LABELS, stem + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                f.writelines(lines)
            os.replace(tmp, os.path.join(LABELS, stem + ".txt"))
        rev = _review()
        if status:
            rev[stem] = {"status": status, "time": datetime.now().isoformat(timespec="seconds")}
        else:
            rev.pop(stem, None)
        _write_review(rev)
    return jsonify({"ok": True, "boxes": _labels(stem), "status": status})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 to review from another PC on the LAN")
    args = ap.parse_args()
    print(f"review page: http://localhost:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
