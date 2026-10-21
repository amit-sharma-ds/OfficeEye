"""Pre-label collected frames with the best models we have, for a person to
correct.

People and phones come from yolo26s (twice the people yolo11n finds on these
ceiling cameras), run with test-time augmentation at a low cutoff. Phones are
what every stock model misses, so a second model (yolo26n trained on
Objects365) adds its phone guesses too; the reviewer deletes the wrong ones -
deleting is quicker than drawing. Output, one per frame:

    data/train/labels/<frame>.txt      class cx cy w h   (0-1 of the picture)
    data/train/auto/<frame>.json       the same boxes with their scores

Class ids stay COCO's - 0 person, 67 cell phone - so a model trained on them
is a drop-in for the app (which looks the ids up by name anyway).

Only frames with no label file yet are done, so it can be run again as the
collector adds frames. Never delete label files to redo them: a frame marked
checked in review.json would then count as checked with labels nobody saw.

    py -3.12 tools/prelabel.py
"""

import argparse
import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "train")
IMAGES = os.path.join(OUT, "images")
LABELS = os.path.join(OUT, "labels")
AUTO = os.path.join(OUT, "auto")

PERSON, PHONE = 0, 67              # COCO ids written to the label files


def _ids(model):
    names = {str(v).lower(): int(k) for k, v in model.names.items()}
    return names.get("person"), names.get("cell phone")


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _boxes(model, path, imgsz, conf, want, tta):
    """[(coco class, score, x1, y1, x2, y2 as 0-1)] for the classes in want
    ({model class id: coco class})."""
    r = model.predict(path, imgsz=imgsz, conf=conf, classes=list(want), augment=tta,
                      verbose=False)[0]
    return [(want[int(k)], float(s), *b) for b, s, k in
            zip(r.boxes.xyxyn.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]


def _merge(boxes, iou):
    """Highest score first; drop a box that overlaps a kept one of its class."""
    kept = []
    for b in sorted(boxes, key=lambda b: -b[1]):
        if all(k[0] != b[0] or _iou(k[2:], b[2:]) < iou for k in kept):
            kept.append(b)
    return kept


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolo26s.pt", help="people and phones")
    ap.add_argument("--phone-model", default="yolo26n-objv1-150.pt",
                    help="extra phone guesses ('' = none)")
    ap.add_argument("--conf", type=float, default=0.15, help="keep people from this score up")
    ap.add_argument("--phone-conf", type=float, default=0.08, help="keep phones from this score up")
    ap.add_argument("--iou", type=float, default=0.5, help="merge boxes overlapping more than this")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--no-tta", action="store_true", help="skip test-time augmentation (faster)")
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0 = default)")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO
    if args.threads:
        torch.set_num_threads(args.threads)

    os.makedirs(LABELS, exist_ok=True)
    os.makedirs(AUTO, exist_ok=True)
    todo = [p for p in sorted(glob.glob(os.path.join(IMAGES, "*.jpg")))
            if not os.path.exists(os.path.join(LABELS, os.path.basename(p)[:-4] + ".txt"))]
    print(f"{len(todo)} frames to pre-label", flush=True)
    if not todo:
        return 0
    path = lambda m: m if os.path.isabs(m) else os.path.join(ROOT, m)
    main_model = YOLO(path(args.model))
    p_id, ph_id = _ids(main_model)
    phone_model = YOLO(path(args.phone_model)) if args.phone_model else None
    extra_ph = _ids(phone_model)[1] if phone_model else None
    used = [os.path.basename(args.model)] + ([os.path.basename(args.phone_model)] if phone_model else [])

    t0 = time.time()
    for i, img in enumerate(todo, 1):
        stem = os.path.basename(img)[:-4]
        found = [b for b in _boxes(main_model, img, args.imgsz, min(args.conf, args.phone_conf),
                                   {p_id: PERSON, ph_id: PHONE}, not args.no_tta)
                 if b[1] >= (args.conf if b[0] == PERSON else args.phone_conf)]
        if phone_model is not None and extra_ph is not None:
            found += _boxes(phone_model, img, args.imgsz, args.phone_conf, {extra_ph: PHONE}, False)
        boxes = []
        for k, s, x1, y1, x2, y2 in _merge(found, args.iou):
            boxes.append({"cls": k, "cx": round((x1 + x2) / 2, 5), "cy": round((y1 + y2) / 2, 5),
                          "w": round(x2 - x1, 5), "h": round(y2 - y1, 5), "score": round(s, 3)})
        with open(os.path.join(AUTO, stem + ".json"), "w", encoding="utf-8") as f:
            json.dump({"models": used, "boxes": boxes}, f)
        with open(os.path.join(LABELS, stem + ".txt"), "w", encoding="utf-8") as f:
            for b in boxes:
                f.write(f"{b['cls']} {b['cx']:.5f} {b['cy']:.5f} {b['w']:.5f} {b['h']:.5f}\n")
        if i % 20 == 0 or i == len(todo):
            el = time.time() - t0
            print(f"{i}/{len(todo)}  {el / i:.1f} s a frame, "
                  f"{(len(todo) - i) * el / i / 60:.0f} min left", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
