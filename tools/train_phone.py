"""Train the phone-use model on the office's own people, check it, and put
it in use only if it is good enough.

The stock models cannot see phones from these ceiling cameras (they found
1-2 of 12 clear uses), so this learns from labelled cut-outs of the office's
own people (tools/phone_data.py, data/phone/labels.json): is this person
using a phone - holding it to read or type, or at the ear - or not.

1. Split by time: the latest quarter of each kind is kept back to test on,
   so the test is on moments the model never saw.
2. Fine-tune yolo26n-cls (ImageNet-trained, small: ~20 ms a person here).
3. Test it; pick the cutoff where most of its alarms are right.
4. Only if at least MIN_PRECISION of its alarms are right and it finds at
   least MIN_RECALL of the real uses, write phone_cls.pt and
   data/phone/model.json - the app picks it up by itself within a minute.
   Otherwise the report says why, and the app keeps working as before.

    py -3.12 tools/train_phone.py [--start "2026-10-21 20:15"] [--epochs 40]
"""

import argparse
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "phone")
CROPS = os.path.join(OUT, "crops")
SET = os.path.join(OUT, "dataset")
MIN_PRECISION = 0.75
MIN_RECALL = 0.5


def log(*a):
    line = "{}  {}".format(datetime.now().strftime("%H:%M:%S"), " ".join(str(x) for x in a))
    print(line, flush=True)
    with open(os.path.join(OUT, "train.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def frame_time(key):
    # cam10_20261021-124712_2 -> 20261021-124712
    return key.split("_")[1]


def build(labels, val_share):
    if os.path.isdir(SET):
        shutil.rmtree(SET)
    split = {"train": {"phone": [], "no_phone": []}, "val": {"phone": [], "no_phone": []}}
    for cls in ("phone", "no_phone"):
        keys = sorted((k for k, v in labels.items() if v == cls), key=frame_time)
        cut = int(len(keys) * (1 - val_share))
        split["train"][cls], split["val"][cls] = keys[:cut], keys[cut:]
    # Few phones: show each one to the model a few times (with its own
    # random changes each time), so it is not drowned by the "no phone" ones.
    rep = max(1, round(len(split["train"]["no_phone"]) / max(1, 2 * len(split["train"]["phone"]))))
    for part, classes in split.items():
        for cls, keys in classes.items():
            d = os.path.join(SET, part, cls)
            os.makedirs(d, exist_ok=True)
            for k in keys:
                for r in range(rep if (part == "train" and cls == "phone") else 1):
                    shutil.copy(os.path.join(CROPS, k + ".jpg"), os.path.join(d, "{}_{}.jpg".format(k, r)))
    return split, rep


def evaluate(model, split):
    """Phone probability for every kept-back person."""
    out = []
    for cls in ("phone", "no_phone"):
        for k in split["val"][cls]:
            r = model.predict(os.path.join(CROPS, k + ".jpg"), imgsz=224, verbose=False)[0]
            names = {v: i for i, v in r.names.items()}
            out.append((float(r.probs.data[names["phone"]]), cls == "phone", k))
    return out


def pick(scored):
    """The cutoff with the best balance, among those where most alarms are right."""
    best = None
    for cut in [x / 100 for x in range(30, 96, 5)]:
        tp = sum(1 for p, y, _ in scored if p >= cut and y)
        fp = sum(1 for p, y, _ in scored if p >= cut and not y)
        pos = sum(1 for _, y, _ in scored if y)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / pos if pos else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        row = {"cutoff": cut, "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
               "right_alarms": tp, "false_alarms": fp, "real_uses": pos}
        if prec >= MIN_PRECISION and (best is None or f1 > best["f1"]):
            best = row
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="", help='wait until "YYYY-mm-dd HH:MM"')
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--val", type=float, default=0.25)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.start:
        start = datetime.strptime(args.start, "%Y-%m-%d %H:%M")
        log("waiting until", start)
        while datetime.now() < start:
            time.sleep(30)

    labels = json.load(open(os.path.join(OUT, "labels.json")))
    labels = {k: v for k, v in labels.items() if v in ("phone", "no_phone")
              and os.path.exists(os.path.join(CROPS, k + ".jpg"))}
    split, rep = build(labels, args.val)
    log("people: train {} phone (x{}) / {} no phone; test {} phone / {} no phone".format(
        len(split["train"]["phone"]), rep, len(split["train"]["no_phone"]),
        len(split["val"]["phone"]), len(split["val"]["no_phone"])))

    random.seed(0)
    from ultralytics import YOLO
    base = os.path.join(ROOT, "yolo26n-cls.pt")
    model = YOLO(base if os.path.exists(base) else "yolo26n-cls.pt")
    t0 = time.time()
    model.train(data=SET, epochs=args.epochs, imgsz=224, batch=16, workers=0, device="cpu",
                patience=12, project=OUT, name="run", exist_ok=True, plots=False, verbose=False,
                amp=False, seed=0, fliplr=0.5, erasing=0.2, hsv_v=0.3)
    log("trained in {:.0f} min".format((time.time() - t0) / 60))

    best_pt = os.path.join(OUT, "run", "weights", "best.pt")
    trained = YOLO(best_pt)
    scored = evaluate(trained, split)
    choice = pick(scored)
    report = {"trained_at": datetime.now().isoformat(timespec="seconds"),
              "train_phone": len(split["train"]["phone"]), "train_no_phone": len(split["train"]["no_phone"]),
              "test_phone": len(split["val"]["phone"]), "test_no_phone": len(split["val"]["no_phone"]),
              "choice": choice,
              "missed": [k for p, y, k in scored if y and (not choice or p < choice["cutoff"])],
              "false": [k for p, y, k in scored if not y and choice and p >= choice["cutoff"]]}
    good = bool(choice and choice["recall"] >= MIN_RECALL and choice["real_uses"] >= 5)
    report["in_use"] = good
    json.dump(report, open(os.path.join(OUT, "report.json"), "w"), indent=1)
    if good:
        shutil.copy(best_pt, os.path.join(ROOT, "phone_cls.pt"))
        json.dump({"model": "phone_cls.pt", "cutoff": choice["cutoff"], "precision": choice["precision"],
                   "recall": choice["recall"], "trained_at": report["trained_at"],
                   "people": len(labels)}, open(os.path.join(OUT, "model.json"), "w"), indent=1)
        log("IN USE: cutoff {cutoff}, {right_alarms} of {real_uses} uses found, "
            "{false_alarms} false alarms (precision {precision}, recall {recall})".format(**choice))
    else:
        log("NOT put in use - not good enough yet:", json.dumps(choice))
    return 0


if __name__ == "__main__":
    sys.exit(main())
