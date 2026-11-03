"""Train the app's detector (yolo26s) on the office's own camera pictures,
overnight, and check whether it is better - it is never put in use from here.

The labels come from a bigger model (yolo26l, a "teacher", far too slow for
the app: ~10 s a picture on this PC) run on every collected frame
(data/train/images). People it finds are kept only if the pose model sees a
body in them - the same check the app makes - so empty chairs are not taught
as people. All 80 kinds of object are labelled, so cars and bikes (Parking)
and objects (Search) are learnt along with people, not forgotten.

    py -3.12 tools/train_people.py                    label, then train at --train-at
    py -3.12 tools/train_people.py --label-only
    py -3.12 tools/train_people.py --wait-collect "2026-11-02 19:35" --hours 7
    py -3.12 tools/train_people.py --student yolo26s-office.pt --hours 6

Things the teacher cannot see either (a truck from straight above at night,
behind a lit railing) are boxed by hand in data/train/manual.json:

    {"cam24_truck_": {"boxes": [[7, cx, cy, w, h]], "copies": 4}}

Every picture whose name starts with the key gets those boxes (class, then
centre and size, 0-1) on top of the teacher's, and is put into training
`copies` more times, so a handful of pictures weigh enough. The report then
says how many of the hand-boxed things the old and the new model find.

Everything lands in data/train/people/: images + labels (the dataset), runs/
(training), report.json (old model against new, on pictures kept out of
training). Runs at idle priority, so the camera app always goes first.
"""

import argparse
import ctypes
import glob
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES = os.path.join(ROOT, "data", "train", "images")
OUT = os.path.join(ROOT, "data", "train", "people")
TEACHER = os.path.join(ROOT, "yolo26l.pt")
STUDENT = os.path.join(ROOT, "yolo26s.pt")
POSE = os.path.join(ROOT, "yolo26n-pose.pt")
MANUAL = os.path.join(ROOT, "data", "train", "manual.json")
PERSON_SURE = 0.6          # a teacher person this sure needs no pose check
PERSON_MIN = 0.25          # ... down to this with a body the pose model sees
OTHER_MIN = 0.35           # every other kind of object


def idle():
    if os.name == "nt":                                   # IDLE_PRIORITY_CLASS
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)


def say(*a):
    print(datetime.now().strftime("%H:%M:%S"), *a, flush=True)


def is_val(name):
    """About one picture in seven is kept out of training, the same ones every run."""
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % 7 == 0


def body(pose, img, box):
    x1, y1, x2, y2 = box
    h, w = img.shape[:2]
    px, py = (x2 - x1) * 0.15, (y2 - y1) * 0.15
    crop = img[max(0, int(y1 - py)):min(h, int(y2 + py)), max(0, int(x1 - px)):min(w, int(x2 + px))]
    if crop.size == 0:
        return False
    r = pose.predict(crop, imgsz=256, conf=0.2, verbose=False)[0]
    if r.keypoints is None or r.keypoints.conf is None:
        return False
    return any(sum(c >= 0.3 for c in pts) >= 2 for pts in r.keypoints.conf.tolist())


def manual():
    """{name prefix: {"boxes": [[class, cx, cy, w, h]], "copies": n}} - see the top."""
    if not os.path.exists(MANUAL):
        return {}
    with open(MANUAL, encoding="utf-8") as f:
        return json.load(f)


def by_hand(name, marks):
    return next((m for p, m in marks.items() if name.startswith(p)), None)


def label(teacher, pose):
    import cv2
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    marks = manual()
    todo = [f for f in sorted(glob.glob(os.path.join(FRAMES, "*.jpg")))
            if not os.path.exists(os.path.join(OUT, "labels", "val" if is_val(os.path.basename(f)) else "train",
                                               os.path.basename(f)[:-4] + ".txt"))]
    say(len(todo), "pictures to label")
    t0 = time.time()
    for i, f in enumerate(todo, 1):
        name = os.path.basename(f)
        part = "val" if is_val(name) else "train"
        img = cv2.imread(f)
        if img is None:
            continue
        h, w = img.shape[:2]
        r = teacher.predict(img, imgsz=960, conf=PERSON_MIN, verbose=False)[0]
        rows = []
        for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            c = int(c)
            if c == 0 and s < PERSON_SURE and not body(pose, img, (x1, y1, x2, y2)):
                continue
            if c != 0 and s < OTHER_MIN:
                continue
            rows.append("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(
                c, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h))
        hand = by_hand(name, marks)
        if hand:
            rows += ["{} {:.6f} {:.6f} {:.6f} {:.6f}".format(int(c), *b) for c, *b in hand["boxes"]]
        copies = [""] + (["_c{}".format(k) for k in range(1, int(hand.get("copies", 0)) + 1)]
                         if hand and part == "train" else [])
        for tail in copies:
            shutil.copy2(f, os.path.join(OUT, "images", part, name[:-4] + tail + ".jpg"))
            with open(os.path.join(OUT, "labels", part, name[:-4] + tail + ".txt"), "w") as fh:
                fh.write("\n".join(rows))
        if i % 20 == 0:
            each = (time.time() - t0) / i
            say("{}/{}  {:.1f} s a picture, {:.0f} min left".format(i, len(todo), each, each * (len(todo) - i) / 60))


def dataset(names):
    path = os.path.join(OUT, "office.yaml")
    with open(path, "w", encoding="utf-8") as f:
        f.write("path: {}\ntrain: images/train\nval: images/val\nnames:\n".format(OUT.replace("\\", "/")))
        for k in sorted(names):
            f.write("  {}: {}\n".format(k, json.dumps(names[k])))
    return path


def counts(model, files, conf=0.3):
    out = {}
    for f in files:
        r = model.predict(f, imgsz=480, conf=conf, classes=[0], verbose=False)[0]
        out[os.path.basename(f)] = r.boxes.xyxy.tolist()
    return out


def compare(old, new):
    """Old against new on the pictures kept out of training, people only:
    found / missed / extra against the teacher's labels, at the app's cutoff."""
    import cv2
    files = sorted(glob.glob(os.path.join(OUT, "images", "val", "*.jpg")))
    truth = {}
    for f in files:
        img = cv2.imread(f)
        h, w = img.shape[:2]
        rows = [l.split() for l in open(os.path.join(OUT, "labels", "val", os.path.basename(f)[:-4] + ".txt"))
                if l.strip()]
        truth[os.path.basename(f)] = [((float(cx) - float(bw) / 2) * w, (float(cy) - float(bh) / 2) * h,
                                       (float(cx) + float(bw) / 2) * w, (float(cy) + float(bh) / 2) * h)
                                      for c, cx, cy, bw, bh in rows if c == "0"]

    def iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
        return ix * iy / u if u else 0

    report = {"pictures": len(files), "people": sum(len(v) for v in truth.values())}
    for key, model in (("old", old), ("new", new)):
        got = counts(model, files)
        found = extra = err = 0
        for name, boxes in got.items():
            want, used = truth[name], set()
            for b in boxes:
                j = max(((iou(b, t), k) for k, t in enumerate(want) if k not in used), default=(0, None))
                if j[0] >= 0.4:
                    used.add(j[1])
                    found += 1
                else:
                    extra += 1
            err += abs(len(boxes) - len(want))
        report[key] = {"found": found, "missed": report["people"] - found, "extra": extra,
                       "count_off_per_picture": round(err / max(1, len(files)), 3)}
    # The hand-boxed things, at the app's vehicle cutoff (detection.Vision.VEHICLE_CONF).
    marks = manual()
    hand = [(f, by_hand(os.path.basename(f), marks)) for f in files]
    hand = [(f, m) for f, m in hand if m]
    if hand:
        report["by_hand"] = {"pictures": len(hand), "boxes": sum(len(m["boxes"]) for _, m in hand)}
        for key, model in (("old", old), ("new", new)):
            found = 0
            for f, m in hand:
                img = cv2.imread(f)
                h, w = img.shape[:2]
                r = model.predict(img, imgsz=480, conf=0.25, verbose=False)[0]
                got = [(b, int(c)) for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist())]
                for c, cx, cy, bw, bh in m["boxes"]:
                    want = ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
                    found += any(k == int(c) and iou(b, want) >= 0.4 for b, k in got)
            report["by_hand"][key + "_found"] = found
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-collect", default="", help='label the frames collected until then too, "YYYY-mm-dd HH:MM"')
    ap.add_argument("--hours", type=float, default=7.0, help="training time")
    ap.add_argument("--label-only", action="store_true")
    ap.add_argument("--student", default=STUDENT, help="the model trained further (and compared against)")
    a = ap.parse_args()
    idle()
    import torch
    torch.set_num_threads(2)            # two of the four left to the camera app
    from ultralytics import YOLO
    teacher, pose = YOLO(TEACHER), YOLO(POSE)
    label(teacher, pose)
    if a.wait_collect:
        until = datetime.strptime(a.wait_collect, "%Y-%m-%d %H:%M")
        while datetime.now() < until:
            time.sleep(60)
        label(teacher, pose)
    if a.label_only:
        return
    del teacher
    student = a.student if os.path.isabs(a.student) else os.path.join(ROOT, a.student)
    say("training {} for {} h".format(os.path.basename(student), a.hours))
    model = YOLO(student)
    run = "office-" + datetime.now().strftime("%Y%m%d")       # last night's run is kept
    model.train(data=dataset(model.names), imgsz=480, time=a.hours, epochs=100, batch=8, workers=0,
                device="cpu", freeze=10, lr0=0.002, lrf=0.2, warmup_epochs=1, patience=50,
                project=os.path.join(OUT, "runs"), name=run, exist_ok=True, plots=False, amp=False,
                verbose=False)
    best = os.path.join(OUT, "runs", run, "weights", "best.pt")
    say("checking old against new")
    report = dict(compare(YOLO(student), YOLO(best)), best=best, at=datetime.now().isoformat(timespec="seconds"))
    with open(os.path.join(OUT, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    say(json.dumps(report))


if __name__ == "__main__":
    sys.exit(main())
