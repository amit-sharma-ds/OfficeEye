"""Person cut-outs for the phone-use model.

Finds every person in the collected frames (data/train/images) with the
app's own detector and cuts them out the way the app will at run time
(15% margin). Labels - "phone" (holding it to read or type, or at the ear)
or "no_phone" - live in data/phone/labels.json, one per cut-out; "skip"
keeps an unclear one out of training.

    py -3.12 tools/phone_data.py            cut out new frames only
    py -3.12 tools/phone_data.py --sheets   also write labelled contact sheets
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FRAMES = os.path.join(DATA, "train", "images")
OUT = os.path.join(DATA, "phone")
CROPS = os.path.join(OUT, "crops")
INDEX = os.path.join(OUT, "crops.json")
PAD = 0.15                 # the same margin the app uses at run time
MIN_H = 60                 # smaller people are too far away to judge


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def cut(frame, box, pad=PAD):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box[:4]
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    return frame[int(max(0, y1 - py)):int(min(h, y2 + py)), int(max(0, x1 - px)):int(min(w, x2 + px))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.join(ROOT, "yolo26s.pt"))
    ap.add_argument("--sheets", action="store_true")
    ap.add_argument("--fill", action="store_true",
                    help="look at every frame again and add only people not cut out yet")
    args = ap.parse_args()
    os.makedirs(CROPS, exist_ok=True)
    index = json.load(open(INDEX)) if os.path.exists(INDEX) else {}
    done = set() if args.fill else {v["frame"] for v in index.values()}
    have = {}
    for v in index.values():
        have.setdefault(v["frame"], []).append(v["box"])
    cfg = json.load(open(os.path.join(DATA, "config.json"), encoding="utf-8"))
    ignore = {c["id"]: c.get("ignore") or [] for c in cfg["cameras"]}
    todo = [f for f in sorted(glob.glob(os.path.join(FRAMES, "*.jpg")))
            if os.path.basename(f) not in done]
    # Frames where the app saw people first: empty stairs teach nothing.
    seen = {}
    try:
        for line in open(os.path.join(DATA, "train", "frames.jsonl"), encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                seen[r["file"]] = r.get("app_count") or 0
    except OSError:
        pass
    todo.sort(key=lambda f: -seen.get(os.path.basename(f), 0))
    print(len(todo), "new frames", flush=True)
    if todo:
        from ultralytics import YOLO
        model = YOLO(args.model)
    for n, path in enumerate(todo, 1):
        name = os.path.basename(path)
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        cam = name.split("_")[0]
        r = model.predict(img, imgsz=480, conf=0.3, classes=[0], verbose=False)[0]
        for pi, (b, s) in enumerate(zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())):
            cx, cy = (b[0] + b[2]) / 2 / w, (b[1] + b[3]) / 2 / h
            if any(cv2.pointPolygonTest(np.array(p, np.float32), (cx, cy), False) >= 0
                   for p in ignore.get(cam, [])):
                continue
            if b[3] - b[1] < MIN_H:
                continue
            if any(_iou(b, o) > 0.4 for o in have.get(name, [])):
                continue                            # this person is cut out already
            crop = cut(img, b)
            key = "{}_{}{}".format(name[:-4], "s" if args.fill else "", pi)
            cv2.imwrite(os.path.join(CROPS, key + ".jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            index[key] = {"frame": name, "cam": cam, "box": [round(v) for v in b], "score": round(s, 2)}
        if n % 25 == 0:
            json.dump(index, open(INDEX, "w"), indent=0)
            print(n, "frames,", len(index), "people", flush=True)
        done.add(name)
    json.dump(index, open(INDEX, "w"), indent=0)
    print("people:", len(index), flush=True)
    if args.sheets:
        sheets(index)
    return 0


def sheets(index, per=40):
    """Contact sheets of all cut-outs, numbered, for labelling by eye."""
    keys = sorted(index)
    numbers = {k: i for i, k in enumerate(keys)}
    json.dump(numbers, open(os.path.join(OUT, "numbers.json"), "w"), indent=0)
    for s in range(0, len(keys), per):
        tiles = []
        for k in keys[s:s + per]:
            img = cv2.imread(os.path.join(CROPS, k + ".jpg"))
            img = cv2.resize(img, (max(1, int(img.shape[1] * 220 / img.shape[0])), 220))
            img = img[:, :180] if img.shape[1] > 180 else cv2.copyMakeBorder(
                img, 0, 0, 0, 180 - img.shape[1], cv2.BORDER_CONSTANT)
            cv2.rectangle(img, (0, 0), (46, 18), (0, 0, 0), -1)
            cv2.putText(img, "{:04d}".format(numbers[k]), (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            tiles.append(img)
        while len(tiles) % 8:
            tiles.append(np.zeros((220, 180, 3), np.uint8))
        grid = np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)])
        cv2.imwrite(os.path.join(OUT, "sheet_{:04d}.jpg".format(s)), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("sheets:", (len(keys) + per - 1) // per)


if __name__ == "__main__":
    sys.exit(main())
