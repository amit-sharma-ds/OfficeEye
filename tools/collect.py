"""Collect training frames from the running app.

Polls /frame/<cam>.jpg?overlay=0&w=1280 for every counting camera and keeps a
frame when the picture moved on from the last one kept for that camera, so
the set has people walking, sitting down, half hidden - not 50 copies of the
same desk. A camera that stays still is still sampled now and then (seated
people, light changes). Quotas spread the frames over cameras and hours.

    py -3.12 tools/collect.py                       # until --until, then stops
    py -3.12 tools/collect.py --until "2026-10-20 12:30" --per-cam 30
    py -3.12 tools/collect.py --office-only --until "2026-10-20 19:30"

Frames where a small model sees a possible phone are kept as extras (up to
--phone-extra per camera), because phones are what the stock models miss.

Frames land in data/train/images/<cam>_<YYYYmmdd-HHMMSS>.jpg (the JPEG the app
served, not re-encoded) with one line each in data/train/frames.jsonl.
Outside the app's code path: it only reads the HTTP API.
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(DATA, "train")
IMAGES = os.path.join(OUT, "images")
INDEX = os.path.join(OUT, "frames.jsonl")


def get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def small(img):
    g = cv2.cvtColor(cv2.resize(img, (160, 120), interpolation=cv2.INTER_AREA),
                     cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(g, (5, 5), 0)


def changed(a, b):
    """Share of the picture that differs: 0.01 = 1% of the pixels moved."""
    return float((cv2.absdiff(a, b) > 22).mean())


def office_open(now, office):
    hm = now.strftime("%H:%M")
    return office.get("open", "09:00") <= hm < office.get("close", "19:00")


def load_index():
    rows = []
    if os.path.exists(INDEX):
        with open(INDEX, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--app", default="http://localhost:8000")
    ap.add_argument("--start", default="", help='wait until this time, "YYYY-mm-dd HH:MM"')
    ap.add_argument("--until", default="", help='stop time, "YYYY-mm-dd HH:MM" (default: never)')
    ap.add_argument("--per-cam", type=int, default=30, help="frames per camera in all")
    ap.add_argument("--boost", default="cam4,cam12",
                    help="cameras that get 1.5x the frames and hourly quota (the evaluation cameras)")
    ap.add_argument("--per-hour", type=int, default=3, help="frames per camera per hour, office hours")
    ap.add_argument("--office-only", action="store_true", help="keep nothing outside office hours")
    ap.add_argument("--night-every", type=float, default=3.0,
                    help="outside office hours: one frame per camera every N hours")
    ap.add_argument("--min-gap", type=float, default=300, help="seconds between two frames of a camera")
    ap.add_argument("--quiet-every", type=float, default=1800,
                    help="keep a frame of a still camera after this many seconds")
    ap.add_argument("--move", type=float, default=0.01,
                    help="share of pixels that must differ from the last kept frame")
    ap.add_argument("--poll", type=float, default=8, help="seconds between rounds")
    ap.add_argument("--phone-model", default="yolo26n-objv1-150.pt",
                    help="small model that looks for phones, to keep extra phone frames ('' = off)")
    ap.add_argument("--phone-conf", type=float, default=0.1, help="phone score that counts as a maybe")
    ap.add_argument("--phone-extra", type=int, default=10, help="extra phone frames per camera")
    ap.add_argument("--phone-every", type=float, default=90, help="seconds between phone checks of a camera")
    ap.add_argument("--phone-gap", type=float, default=120, help="seconds between two phone frames of a camera")
    args = ap.parse_args()

    until = datetime.strptime(args.until, "%Y-%m-%d %H:%M") if args.until else None
    if args.start:
        start = datetime.strptime(args.start, "%Y-%m-%d %H:%M")
        print(f"waiting until {start}", flush=True)
        while datetime.now() < start:
            time.sleep(30)
    os.makedirs(IMAGES, exist_ok=True)
    with open(os.path.join(DATA, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    office = cfg.get("office", {})
    cams = [c["id"] for c in cfg["cameras"] if c.get("group") == "count" and c.get("url")]
    boost = set(filter(None, args.boost.split(",")))
    budget = {c: int(args.per_cam * (1.5 if c in boost else 1)) for c in cams}

    phone_model, phone_cls = None, None
    if args.phone_model:
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(1)                    # leave the cores to the app
        path = args.phone_model if os.path.isabs(args.phone_model) else os.path.join(ROOT, args.phone_model)
        phone_model = YOLO(path)
        names = {str(v).lower(): int(k) for k, v in phone_model.names.items()}
        phone_cls = names.get("cell phone", 67)

    # Pick up where an earlier run stopped.
    rows = load_index()
    kept = {c: [r for r in rows if r["cam"] == c and r.get("reason") != "phone"] for c in cams}
    extra = {c: sum(1 for r in rows if r["cam"] == c and r.get("reason") == "phone") for c in cams}
    last_small, last_t = {}, {}
    for r in rows:
        if r["cam"] in cams:
            last_t[r["cam"]] = r["ts"]
            img = cv2.imread(os.path.join(IMAGES, r["file"]))
            if img is not None:
                last_small[r["cam"]] = small(img)
    prev_small, phone_checked = {}, {}
    print(f"{len(cams)} counting cameras, {len(rows)} frames already kept; "
          f"budget {sum(budget.values())} + up to {args.phone_extra} phone frames each", flush=True)

    while True:
        now = datetime.now()
        if until and now >= until:
            break
        if all(len(kept[c]) >= budget[c] and extra[c] >= args.phone_extra for c in cams):
            print("every camera has its frames", flush=True)
            break
        is_open = office_open(now, office)
        if args.office_only and not is_open:
            time.sleep(60)
            continue
        try:
            status = json.loads(get(f"{args.app}/api/status"))["cameras"]
        except Exception as e:                     # app restarting
            print("status:", e, flush=True)
            time.sleep(30)
            continue
        hour_ago = time.time() - 3600
        for c in cams:
            st = status.get(c) or {}
            if st.get("state") != "connected" or not st.get("fresh"):
                continue
            t = time.time()
            since_last = t - last_t.get(c, 0)
            if is_open:
                per_hour = args.per_hour * (1.5 if c in boost else 1)
                normal_ok = (len(kept[c]) < budget[c] and since_last >= args.min_gap and
                             sum(1 for r in kept[c] if r["ts"] > hour_ago) < per_hour)
            else:
                normal_ok = len(kept[c]) < budget[c] and since_last >= args.night_every * 3600
            phone_ok = (phone_model is not None and is_open and (st.get("count") or 0) > 0 and
                        extra[c] < args.phone_extra and since_last >= args.phone_gap and
                        t - phone_checked.get(c, 0) >= args.phone_every)
            if not (normal_ok or phone_ok):
                continue
            try:
                jpg = get(f"{args.app}/frame/{c}.jpg?overlay=0&w=1280")
            except Exception as e:
                print(c, e, flush=True)
                continue
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            sw, sh = st.get("width") or 0, st.get("height") or 0
            want_w = min(sw, 1280)                  # the app scales big streams down
            if img is None or not sw or img.shape[1] != want_w or \
                    abs(img.shape[0] - sh * want_w / sw) > 2:
                continue                            # "connecting..." placeholder
            s = small(img)
            since_kept = changed(s, last_small[c]) if c in last_small else 1.0
            moving = changed(s, prev_small[c]) if c in prev_small else 0.0
            prev_small[c] = s
            phone_score = 0.0
            if phone_ok:
                phone_checked[c] = t
                r = phone_model.predict(img, imgsz=640, conf=args.phone_conf,
                                        classes=[phone_cls], verbose=False)[0]
                phone_score = max(r.boxes.conf.tolist(), default=0.0)
            if phone_ok and phone_score >= args.phone_conf:
                reason = "phone"
            elif normal_ok and (not is_open or since_kept >= args.move or since_last >= args.quiet_every):
                reason = "night" if not is_open else "move" if since_kept >= args.move else "quiet"
            else:
                continue
            name = f"{c}_{now.strftime('%Y%m%d-%H%M%S')}.jpg"
            with open(os.path.join(IMAGES, name), "wb") as f:
                f.write(jpg)
            row = {"file": name, "cam": c, "ts": t, "time": now.isoformat(timespec="seconds"),
                   "reason": reason, "change": round(since_kept, 4), "moving": round(moving, 4),
                   "phone_score": round(phone_score, 3), "app_count": st.get("count"),
                   "office_open": is_open}
            with open(INDEX, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            if reason == "phone":
                extra[c] += 1
            else:
                kept[c].append(row)
            last_small[c], last_t[c] = s, t
            print(f"{now:%H:%M:%S} {c:6s} {reason:5s} #{len(kept[c])}+{extra[c]} change {since_kept:.3f} "
                  f"phone {phone_score:.2f} app count {st.get('count')}", flush=True)
        time.sleep(args.poll)

    total = sum(len(v) for v in kept.values()) + sum(extra.values())
    print(f"done: {total} frames in {IMAGES}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
