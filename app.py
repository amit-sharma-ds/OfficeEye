"""Web app: pages, the JSON API they talk to, and the video endpoints."""

import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timedelta

import cv2
import numpy as np
from flask import (Flask, Response, abort, jsonify, redirect, render_template,
                   request, send_from_directory)

import cloud
import config
import events
import mongo
import photos
import scan
import washroom                     # noqa: F401 - adds its detection to the pipeline
import sleeping                     # head down / sleeping, added the same way
from cameras import manager
from detection import phone_watch_state, pipeline

if cloud.CLOUD:
    # The cloud copy has no cameras of its own: the pages read what the
    # office PC sent (see cloud.py).
    manager, pipeline = cloud.ManagerView(), cloud.PipelineView()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("werkzeug").setLevel(logging.WARNING)   # one line per image is noise
log = logging.getLogger("app")

PORT = int(os.environ.get("PORT", 8000))      # Render and the like set PORT

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
cloud.install(app)


# ---------------------------------------------------------------- pages

def _page(template, page, **extra):
    return render_template(template, page=page, **extra)


@app.get("/")
def dashboard():
    return _page("dashboard.html", "dashboard", cameras=manager.listing())


@app.get("/live")
def live_page():
    return _page("live.html", "live", cameras=manager.listing())


@app.get("/running-time")
def running_time_page():
    return _page("running_time.html", "running")


@app.get("/uptime")
def uptime_page():
    return redirect("/running-time")


@app.get("/restricted")
def restricted_page():
    return _page("restricted.html", "restricted",
                 cameras=manager.listing(config.RESTRICTED))


@app.get("/washroom")
def washroom_page():
    return _page("washroom.html", "washroom", cameras=manager.listing(config.WASHROOM))


@app.get("/phone")
def phone_page():
    return _page("phone.html", "phone", cameras=manager.listing(config.COUNT))


@app.get("/sleeping")
def sleeping_page():
    return _page("sleeping.html", "sleeping", cameras=manager.listing(config.COUNT))


@app.get("/cameras")
def cameras_page():
    return _page("cameras.html", "cameras", cameras=manager.listing())


# Detection models offered on Settings, best first. Measured 2026-10-19 on 34
# frames from the office cameras (18 people): yolo26s found 13, yolo26n-objv1
# 9, yolo11n 7. Other .pt files dropped in the folder (a trained model) are
# listed after these.
MODELS = [
    ("yolo26s.pt", "YOLO26s - most accurate, finds about twice the people (recommended)"),
    ("yolo26n-objv1-150.pt", "YOLO26n Objects365 - fast, a little better than YOLO11n"),
    ("yolo26n.pt", "YOLO26n - fastest"),
    ("yolo11n.pt", "YOLO11n - the old default"),
    ("yolo11s.pt", "YOLO11s"),
]


def _models():
    """The known models (one not on disk yet downloads itself when chosen),
    then any other .pt in the folder."""
    base = os.path.dirname(os.path.abspath(__file__))
    extra = sorted(f for f in os.listdir(base)
                   if f.endswith(".pt") and f not in dict(MODELS))
    return MODELS + [(f, f) for f in extra]


@app.get("/settings")
def settings_page():
    return _page("settings.html", "settings", models=_models(),
                 office=cloud.section("office"),
                 detection=cloud.section("detection"),
                 keep_days=cloud.section("photos").get("keep_days", 30))


# ---------------------------------------------------------------- api

@app.get("/api/status")
def api_status():
    if cloud.CLOUD:
        return jsonify(cloud.store.status_payload())
    return jsonify(_status_payload())


def _status_payload():
    cams = manager.status()
    live = pipeline.live()
    peaks = events.peak("person_count")
    total = on_phone = 0
    reporting = 0
    alerting = []
    for cam_id, st in cams.items():
        st.update(live.get(cam_id, {}))
        st["peak"] = peaks.get(cam_id)
        if st.get("count") is not None:
            total += st["count"]
            reporting += 1
        on_phone += st.get("on_phone") or 0
        if (st.get("alert") or {}).get("active"):
            alerting.append(cam_id)
    return {
        "cameras": cams,
        "detection": pipeline.status(),
        "total": total,
        "reporting": reporting,
        "on_phone_total": on_phone,
        "peak_today": max(peaks.values()) if peaks else None,
        "alerting": alerting,
        "latest_alert": events.latest_id("restricted_alert"),
        "alerts_today": events.count_today("restricted_alert"),
        "phone_today": events.count_today("phone_use", with_photos=True),
        "server_time": datetime.now().isoformat(timespec="seconds"),
    }


def _day_arg():
    day = (request.args.get("day") or "").strip()
    try:
        return datetime.strptime(day, "%Y-%m-%d").strftime("%Y-%m-%d") if day \
            else datetime.now().strftime("%Y-%m-%d")
    except ValueError:
        abort(400)


@app.get("/api/alerts")
def api_alerts():
    limit = min(int(request.args.get("limit", 50)), 500)
    return jsonify({"alerts": events.recent(limit, "restricted_alert",
                                            day=_day_arg(),
                                            camera=request.args.get("camera"))})


@app.get("/api/alerts/<int:event_id>")
def api_alert(event_id):
    ev = events.get(event_id)
    if not ev:
        abort(404)
    return jsonify(ev)


@app.post("/api/alerts/<int:event_id>/delete")
def api_alert_delete(event_id):
    """Delete one restricted-area alert and its photos (the cloud view and
    the MongoDB copy follow within a minute or two)."""
    meta = events.delete(event_id, "restricted_alert")
    if meta is None:
        return jsonify({"ok": False, "message": "That alert is already gone"})
    photos.remove(meta.get("photos"))
    return jsonify({"ok": True, "message": "Alert deleted"})


@app.get("/api/people/day")
def api_people_day():
    """People in view hour by hour, for the dashboard chart."""
    out = events.people_by_hour(_day_arg())
    office = cloud.section("office")
    out["open"], out["close"] = office.get("open"), office.get("close")
    return jsonify(out)


def _uptime_report():
    day = _day_arg()
    days = min(max(int(request.args.get("days", 1)), 1), 90)
    data = events.uptime(day, days)
    start = datetime.strptime(data["first"], "%Y-%m-%d")
    end = min(datetime.strptime(day, "%Y-%m-%d") + timedelta(days=1), datetime.now())
    cams = []
    for cam_id, st in manager.status().items():
        u = data["cameras"].get(cam_id, {})
        cams.append({"id": cam_id, "name": st["name"], "group": st["group"],
                     "ip": st["ip"], "state": st.get("state"),
                     "online": round(u.get("online", 0)),
                     "offline": round(u.get("offline", 0)),
                     "drops": u.get("drops", 0),
                     "last_online": u.get("last_online"),
                     "hours": u.get("hours", {})})
    return {"first": data["first"], "last": day, "days": days,
            "period": max(0, round((end - start).total_seconds())), "cameras": cams}


@app.get("/api/running-time")
def api_running_time():
    """How many hours each camera was connected and how many it was off."""
    return jsonify(_uptime_report())


@app.get("/api/running-time.csv")
def api_running_time_csv():
    r = _uptime_report()
    hours = lambda s: f"{s / 3600:.2f}"
    lines = ["camera,name,purpose,ip,online_hours,offline_hours,app_not_running_hours,"
             "running_percent,drops,last_online,from,to"]
    for c in r["cameras"]:
        seen = c["online"] + c["offline"]
        cells = [c["id"], c["name"], c["group"], c["ip"], hours(c["online"]),
                 hours(c["offline"]), hours(max(0, r["period"] - seen)),
                 f"{100 * c['online'] / seen:.1f}" if seen else "",
                 str(c["drops"]), c["last_online"] or "", r["first"], r["last"]]
        lines.append(",".join('"' + str(v).replace('"', '""') + '"' for v in cells))
    name = f"camera-running-time-{r['first']}" + ("" if r["days"] == 1 else f"-to-{r['last']}")
    return Response("\r\n".join(lines) + "\r\n", mimetype="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{name}.csv"'})


def _daily_rows():
    """Saved daily records (per camera per day) for the last `days` days,
    plus today so far, newest first. ?camera= keeps one camera."""
    days = min(max(int(request.args.get("days", 30)), 1), 3660)
    camera = request.args.get("camera") or None
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    first = (now - timedelta(days=days - 1)).strftime("%Y-%m-%d")
    names = {k: v["name"] for k, v in manager.status().items()}
    live = events.uptime(today)["cameras"]
    since_midnight = (now - now.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
    rows = []
    for cam_id, u in live.items():
        if camera and cam_id != camera:
            continue
        rows.append({"day": today, "today": True, "camera": cam_id,
                     "name": names.get(cam_id, cam_id),
                     "running": round(u["online"]), "off": round(u["offline"]),
                     "not_watched": max(0, round(since_midnight - u["online"] - u["offline"])),
                     "drops": u["drops"], "last_online": u["last_online"]})
    rows.sort(key=lambda r: r["camera"])
    rows += events.camera_days(first, today, camera)
    return first, today, camera, rows


@app.get("/api/running-time/days")
def api_running_days():
    """The day-by-day record: per camera, or every camera added up."""
    first, last, camera, rows = _daily_rows()
    if camera:
        return jsonify({"first": first, "last": last, "camera": camera, "rows": rows})
    days = {}
    for r in rows:
        d = days.setdefault(r["day"], {"day": r["day"], "today": r.get("today", False),
                                       "cameras": 0, "running": 0, "off": 0, "drops": 0,
                                       "worst": None})
        d["cameras"] += 1
        d["running"] += r["running"]
        d["off"] += r["off"]
        d["drops"] += r["drops"]
        if r["off"] >= 60 and (d["worst"] is None or r["off"] > d["worst"]["off"]):
            d["worst"] = {"name": r["name"], "off": r["off"]}
    return jsonify({"first": first, "last": last, "camera": None,
                    "rows": sorted(days.values(), key=lambda d: d["day"], reverse=True)})


@app.get("/api/running-time/days.csv")
def api_running_days_csv():
    first, last, camera, rows = _daily_rows()
    hours = lambda s: f"{s / 3600:.2f}"
    lines = ["day,camera,name,running_hours,off_hours,app_not_running_hours,"
             "running_percent,drops,last_online,complete_day"]
    for r in rows:
        seen = r["running"] + r["off"]
        cells = [r["day"], r["camera"], r["name"], hours(r["running"]), hours(r["off"]),
                 hours(r["not_watched"]), f"{100 * r['running'] / seen:.1f}" if seen else "",
                 r["drops"], r["last_online"] or "", "no - today so far" if r.get("today") else "yes"]
        lines.append(",".join('"' + str(v).replace('"', '""') + '"' for v in cells))
    name = f"camera-daily-running-time-{first}-to-{last}" + (f"-{camera}" if camera else "")
    return Response("\r\n".join(lines) + "\r\n", mimetype="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{name}.csv"'})


_HM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


def _phone_model():
    """The office-trained phone model: in use, or what its last test said."""
    folder = os.path.join(config.DATA_DIR, "phone")
    out = {}
    for name in ("model", "report"):
        try:
            with open(os.path.join(folder, name + ".json"), encoding="utf-8") as f:
                out[name] = json.load(f)
        except (OSError, ValueError):
            pass
    return out


def _phone_rules_out():
    rules, office = cloud.section("phone"), cloud.section("office")
    return {"rules": rules, "office": {"open": office.get("open"), "close": office.get("close")},
            "now": phone_watch_state(rules, office, datetime.now()), "model": _phone_model()}


@app.get("/api/phone/rules")
def api_phone_rules():
    return jsonify(_phone_rules_out())


@app.post("/api/phone/rules")
def api_phone_rules_save():
    body = request.get_json(silent=True) or {}
    patch = {}
    try:
        if "min_seconds" in body:
            patch["min_seconds"] = min(max(int(body["min_seconds"]), 0), 600)
        if "photo_every" in body:
            patch["photo_every"] = min(max(float(body["photo_every"]), 0), 120)
    except (TypeError, ValueError):
        return jsonify({"ok": False, "message": "Seconds and minutes must be numbers"})
    for key in ("watch_from", "watch_to"):
        if key in body:
            v = str(body[key] or "").strip()
            if v and not _HM.match(v):
                return jsonify({"ok": False, "message": "Times look like 09:30"})
            patch[key] = v
    if "breaks" in body:
        breaks = []
        for b in body["breaks"] or []:
            if not (isinstance(b, (list, tuple)) and len(b) == 2 and all(_HM.match(str(x)) for x in b)):
                return jsonify({"ok": False, "message": "Each break needs a start and an end, like 13:00 - 14:00"})
            breaks.append([str(b[0]), str(b[1])])
        patch["breaks"] = breaks
    if body.get("sensitivity") in ("low", "medium", "high"):
        patch["sensitivity"] = body["sensitivity"]
    config.update("phone", patch)
    pipeline.reload()
    return jsonify({"ok": True, "message": "Phone rules saved", **_phone_rules_out()})


@app.get("/api/mongo")
def api_mongo():
    """How far the MongoDB copy of the records has got."""
    return jsonify(mongo.sync.info())


@app.get("/api/phone/uses")
def api_phone_uses():
    """Every phone use of a day: from when to when, with a photo when one
    was taken. (Older records only have photos.)"""
    rows = events.recent(1000, "phone_use", day=_day_arg(),
                         camera=request.args.get("camera") or None)
    uses = [r for r in rows if r["meta"].get("started") or r["meta"].get("photos")]
    return jsonify({"uses": uses, "seconds": sum(r["meta"].get("seconds") or 0 for r in uses)})


@app.get("/api/phone")
def api_phone():
    limit = min(int(request.args.get("limit", 60)), 500)
    return jsonify({"events": events.recent(limit, "phone_use", day=_day_arg(),
                                            camera=request.args.get("camera"),
                                            with_photos=True)})


def _sleep_rules_out():
    rules, office = cloud.section("sleeping"), cloud.section("office")
    return {"rules": rules, "office": {"open": office.get("open"), "close": office.get("close")},
            "now": sleeping.watch_state(rules, office, datetime.now())}


@app.get("/api/sleeping/rules")
def api_sleeping_rules():
    return jsonify(_sleep_rules_out())


@app.post("/api/sleeping/rules")
def api_sleeping_rules_save():
    body = request.get_json(silent=True) or {}
    patch = {}
    try:
        if "min_minutes" in body:
            patch["min_minutes"] = min(max(float(body["min_minutes"]), 1), 120)
    except (TypeError, ValueError):
        return jsonify({"ok": False, "message": "Minutes must be a number"})
    for key in ("watch_from", "watch_to"):
        if key in body:
            v = str(body[key] or "").strip()
            if v and not _HM.match(v):
                return jsonify({"ok": False, "message": "Times look like 09:30"})
            patch[key] = v
    if "breaks" in body:
        breaks = []
        for b in body["breaks"] or []:
            if not (isinstance(b, (list, tuple)) and len(b) == 2 and all(_HM.match(str(x)) for x in b)):
                return jsonify({"ok": False, "message": "Each break needs a start and an end, like 13:00 - 14:00"})
            breaks.append([str(b[0]), str(b[1])])
        patch["breaks"] = breaks
    config.update("sleeping", patch)
    return jsonify({"ok": True, "message": "Rules saved", **_sleep_rules_out()})


@app.get("/api/sleeping/uses")
def api_sleeping_uses():
    """Every head-down spell of a day: from when to when, with a photo."""
    uses = events.recent(1000, "head_down", day=_day_arg(),
                         camera=request.args.get("camera") or None)
    return jsonify({"uses": uses, "seconds": sum(r["meta"].get("seconds") or 0 for r in uses)})


WASH_KINDS = ["washroom_visit", "washroom_in", "washroom_out", "washroom_door", "washroom_check"]
WASH_USES = ["washroom_visit", "washroom_in", "washroom_out", "washroom_door"]
PROBLEMS = ("dirty", "damaged")


@app.get("/api/washroom")
def api_washroom():
    """One day at the washroom doors: visits, door-only uses and checks."""
    day = _day_arg()
    end = (datetime.strptime(day, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    return jsonify({"day": day, "events": events.between(
        WASH_KINDS, day, end, camera=request.args.get("camera") or None)})


@app.get("/api/washroom/since")
def api_washroom_since():
    """Who went in between the last 'checked clean' and a problem report (or
    now): the people to ask when the washroom is found dirty or damaged."""
    camera = request.args.get("camera") or None
    now = datetime.now().isoformat(timespec="seconds")
    report = request.args.get("at") or None
    if report is not None:
        try:
            report = datetime.fromisoformat(report).isoformat(timespec="seconds")
        except ValueError:
            abort(400)
    else:
        latest = events.between(["washroom_check"], "2000", "9999", camera, PROBLEMS, limit=1)
        clean_after = events.between(["washroom_check"], latest[0]["ts"] if latest else "9999",
                                     "9999", camera, ("clean",), limit=1) if latest else []
        # The newest problem report, unless it was cleaned since.
        report = latest[0]["ts"] if latest and not clean_after else now
    clean = events.between(["washroom_check"], "2000", report, camera, ("clean",), limit=1)
    start = clean[0]["ts"] if clean else \
        (datetime.fromisoformat(report) - timedelta(days=1)).isoformat(timespec="seconds")
    visits = events.between(WASH_USES, start, report + "~", camera, limit=300)
    return jsonify({"from": start, "clean": clean[0] if clean else None, "to": report,
                    "is_report": report != now, "visits": visits})


@app.post("/api/washroom/check")
def api_washroom_check():
    body = request.get_json(silent=True) or {}
    status = body.get("status")
    cam = config.camera(body.get("camera") or "")
    if status not in ("clean",) + PROBLEMS or not cam:
        return jsonify({"ok": False, "message": "Pick a washroom camera and what was found"})
    eid = events.log("washroom_check", camera=cam["id"], label=status, value=0,
                     meta={"note": str(body.get("note") or "")[:300]})
    return jsonify({"ok": True, "id": eid, "message": {
        "clean": "Marked clean", "dirty": "Marked dirty", "damaged": "Marked damaged"}[status]})


@app.post("/api/cameras/add")
def api_camera_add():
    body = request.get_json(silent=True) or {}
    return jsonify({"ok": True, "camera": manager.add(body.get("group",
                                                              config.COUNT))})


@app.post("/api/cameras/<cam_id>/save")
def api_camera_save(cam_id):
    body = request.get_json(silent=True) or {}
    ok, message, values = manager.save(cam_id, body.get("ip"), body.get("url"),
                                       body.get("name"), body.get("group"))
    return jsonify({"ok": ok, "message": message, "values": values,
                    "status": manager.status().get(cam_id, {})})


@app.post("/api/cameras/<cam_id>/zone")
def api_camera_zone(cam_id):
    if config.camera(cam_id) is None:
        abort(404)
    body = request.get_json(silent=True) or {}
    zone = manager.set_zone(cam_id, body.get("zone"))
    return jsonify({"ok": True, "zone": zone,
                    "message": "Zone saved" if zone else
                               "No zone - the whole picture is watched"})


@app.post("/api/cameras/<cam_id>/ignore")
def api_camera_ignore(cam_id):
    if config.camera(cam_id) is None:
        abort(404)
    body = request.get_json(silent=True) or {}
    shapes = manager.set_ignore(cam_id, body.get("ignore"))
    return jsonify({"ok": True, "ignore": shapes,
                    "message": "{} ignore area{} saved".format(
                        len(shapes), "" if len(shapes) == 1 else "s")})


@app.post("/api/cameras/<cam_id>/reset")
def api_camera_reset(cam_id):
    values = manager.reset(cam_id)
    return jsonify({"ok": True, "message": "Cleared", "values": values,
                    "status": manager.status().get(cam_id, {})})


@app.post("/api/cameras/<cam_id>/remove")
def api_camera_remove(cam_id):
    manager.remove(cam_id)
    return jsonify({"ok": True, "message": "Camera removed"})


@app.post("/api/scan")
def api_scan_start():
    started = scan.scanner.start()
    return jsonify({
        "ok": started,
        "message": "Scanning the network..." if started
                   else "A scan is already running",
        "scan": scan.scanner.public(),
    })


@app.get("/api/scan")
def api_scan_status():
    return jsonify(scan.scanner.public())


@app.post("/api/scan/add")
def api_scan_add():
    body = request.get_json(silent=True) or {}
    ok, message, values = manager.add_scanned(body.get("ip"),
                                              body.get("group", config.COUNT))
    return jsonify({"ok": ok, "message": message, "values": values})


@app.post("/api/scan/add-all")
def api_scan_add_all():
    body = request.get_json(silent=True) or {}
    added = manager.add_all_scanned(body.get("group", config.COUNT))
    return jsonify({"ok": True, "added": added,
                    "message": "Added {} camera{}".format(
                        len(added), "" if len(added) == 1 else "s")})


@app.post("/api/office")
def api_office():
    body = request.get_json(silent=True) or {}
    keep = {k: body[k] for k in ("open", "close", "late_after", "timezone")
            if k in body}
    cfg = config.update("office", keep)
    return jsonify({"ok": True, "message": "Office timing saved",
                    "office": cfg["office"]})


@app.post("/api/detection")
def api_detection():
    body = request.get_json(silent=True) or {}
    patch = {}
    for key in ("enabled", "phone"):
        if key in body:
            patch[key] = bool(body[key])
    limits = {"confidence": (0.1, 0.95), "count_refresh": (5, 600),
              "restricted_fps": (0.5, 15)}
    for key, (lo, hi) in limits.items():
        if key in body:
            try:
                patch[key] = min(max(float(body[key]), lo), hi)
            except (TypeError, ValueError):
                pass
    if body.get("imgsz") in (320, 416, 480, 640, "320", "416", "480", "640"):
        patch["imgsz"] = int(body["imgsz"])
    if body.get("model") in dict(_models()):
        patch["model"] = body["model"]
    config.update("detection", patch)
    pipeline.reload()
    return jsonify({"ok": True, "message": "Detection settings saved",
                    "detection": pipeline.status()})


@app.post("/api/photos")
def api_photos():
    body = request.get_json(silent=True) or {}
    try:
        days = min(max(int(body.get("keep_days", 30)), 1), 3650)
    except (TypeError, ValueError):
        days = 30
    config.update("photos", {"keep_days": days})
    gone = photos.purge(days)
    return jsonify({"ok": True, "keep_days": days,
                    "message": "Saved - photos kept for {} days".format(days)
                               + (" ({} old day(s) cleared)".format(gone) if gone else "")})


@app.get("/api/photos/usage")
def api_photos_usage():
    files, mb = photos.usage()
    return jsonify({"files": files, "mb": mb})


@app.get("/api/events")
def api_events():
    limit = min(int(request.args.get("limit", 50)), 500)
    return jsonify({"events": events.recent(limit, request.args.get("kind"))})


# ---------------------------------------------------------------- images

_jpeg_cache = {}            # (camera, width, overlay) -> (key, bytes)
_jpeg_lock = threading.Lock()


def _render(cam_id, width, overlay=True, quality=75, caption=True):
    """Newest frame of a camera as JPEG, with the boxes drawn. Encoded once
    per new frame or new result, however many people are looking."""
    stream = manager.get(cam_id)
    if stream is None:
        return None, None
    frame, _, seq = stream.latest()
    if frame is None:
        return None, None
    key = (seq, pipeline.version(cam_id) if overlay else 0)
    ck = (cam_id, width, overlay, caption)
    with _jpeg_lock:
        hit = _jpeg_cache.get(ck)
        if hit and hit[0] == key:
            return hit[1], key
    img = frame.copy()
    if overlay:
        img = pipeline.overlay(cam_id, img, caption)
    if width and img.shape[1] > width:
        scale = width / float(img.shape[1])
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None, None
    jpg = buf.tobytes()
    with _jpeg_lock:
        _jpeg_cache[ck] = (key, jpg)
    return jpg, key


def _placeholder(text="no signal"):
    img = np.full((360, 480, 3), 30, dtype="uint8")
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(img, text, ((480 - tw) // 2, 190), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (130, 130, 130), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes() if ok else b""


_BLANK = {}


def _blank(text):
    if text not in _BLANK:
        _BLANK[text] = _placeholder(text)
    return _BLANK[text]


@app.get("/frame/<cam_id>.jpg")
def frame_jpg(cam_id):
    """The camera wall polls this - one small request per tile, not a video
    stream per tile (a browser only opens 6 connections to one server)."""
    if cloud.CLOUD:
        return cloud.frame_response(cam_id, _blank)
    width = min(int(request.args.get("w", 480)), 1280)
    overlay = request.args.get("overlay", "1") != "0"
    caption = request.args.get("caption", "1") != "0"
    jpg, _ = _render(cam_id, width, overlay, quality=70, caption=caption)
    if jpg is None:
        st = manager.get(cam_id)
        text = "no IP set" if st is None or not st.url else "connecting..."
        return Response(_blank(text), mimetype="image/jpeg",
                        headers={"Cache-Control": "no-store"})
    return Response(jpg, mimetype="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.get("/snapshot/<cam_id>.jpg")
def snapshot(cam_id):
    return frame_jpg(cam_id)


@app.get("/live/<cam_id>")
def live(cam_id):
    """Full-rate video of one camera. While it is open the camera decodes
    every frame and is checked more often."""
    if cloud.CLOUD:
        return cloud.live_response(cam_id, _blank)
    stream = manager.get(cam_id)
    if stream is None:
        abort(404)
    width = min(int(request.args.get("w", 960)), 1920)

    def gen():
        stream.watch()
        try:
            last, last_sent = None, 0.0
            while manager.get(cam_id) is stream:
                jpg, key = _render(cam_id, width, True, quality=78)
                now = time.time()
                if jpg is None:
                    jpg, key = _blank("connecting..."), "blank"
                # Re-send now and then even when nothing changed: it is the
                # only way to notice the viewer has gone.
                if key != last or now - last_sent > 2.0:
                    last, last_sent = key, now
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                           b"Content-Length: " + str(len(jpg)).encode()
                           + b"\r\n\r\n" + jpg + b"\r\n")
                time.sleep(1 / 20.0)
        finally:
            stream.unwatch()

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-store"})


@app.get("/photos/<path:name>")
def photo(name):
    return send_from_directory(photos.PHOTO_DIR, name, max_age=86400)


# ---------------------------------------------------------------- main

def _housekeeping():
    """Once an hour: drop old events and old photos."""
    while True:
        try:
            events.purge_older_than(90)
            photos.purge(config.get("photos").get("keep_days", 30))
        except Exception:                                   # noqa: BLE001
            log.exception("housekeeping failed")
        time.sleep(3600)


def _uptime_watch(every=10, flush=60):
    """How long each camera is connected: its state is looked at every few
    seconds and added up per hour, written once a minute. Cameras with no IP
    yet are left out; time the app itself is not running counts as neither."""
    was_up, pending, closed_for = {}, {}, None
    last = last_flush = time.time()
    while True:
        time.sleep(every)
        now = time.time()
        span = min(now - last, 3 * every)        # PC asleep: don't count the gap
        last = now
        stamp = datetime.now()
        try:
            status = manager.status()
            for cam_id, st in status.items():
                if st.get("state") == "off":
                    was_up.pop(cam_id, None)
                    continue
                up = st.get("state") == "connected"
                key = (stamp.strftime("%Y-%m-%d"), stamp.hour, cam_id)
                row = pending.setdefault(key, [0.0, 0.0, 0, None])
                row[0 if up else 1] += span
                if was_up.get(cam_id) and not up:
                    row[2] += 1
                if up:
                    row[3] = stamp.isoformat(timespec="seconds")
                was_up[cam_id] = up
            if now - last_flush >= flush:
                events.add_uptime([k + tuple(v) for k, v in pending.items()])
                pending, last_flush = {}, now
                day = stamp.strftime("%Y-%m-%d")
                if closed_for != day:             # first flush of a new day
                    saved = events.close_days(day, {k: v["name"] for k, v in status.items()})
                    if saved:
                        log.info("daily camera running time saved for %s", ", ".join(saved))
                    closed_for = day
        except Exception:                                   # noqa: BLE001
            log.exception("uptime watch failed")


def _already_running(port):
    """Windows lets a second process bind the same port, and then half the
    requests go to the wrong app. Refuse to start instead."""
    import socket
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def main():
    if _already_running(PORT):
        log.error("port %s is already in use - the app is probably open in "
                  "another window. Close it first.", PORT)
        return
    os.makedirs(photos.PHOTO_DIR, exist_ok=True)
    threading.Thread(target=_housekeeping, name="housekeeping", daemon=True).start()
    if cloud.CLOUD:
        log.info("cloud view: showing what the office PC sends, on port %s", PORT)
        app.run(host="0.0.0.0", port=PORT, threaded=True, debug=False, use_reloader=False)
        return
    manager.start()
    pipeline.start()
    threading.Thread(target=_uptime_watch, name="uptime", daemon=True).start()
    cloud.uplink.start(_status_payload, _render, manager.listing)
    mongo.sync.start(manager.listing)
    log.info("open  http://localhost:%s", PORT)
    try:
        app.run(host="0.0.0.0", port=PORT, threaded=True,
                debug=False, use_reloader=False)
    finally:
        pipeline.stop()
        manager.stop()


if __name__ == "__main__":
    main()
