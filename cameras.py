"""Owns one stream per camera and the add / save / remove logic behind the
Cameras page. As many cameras as the office needs."""

import logging
import threading

import camera
import config
import rtsp
import scan
from detection import pipeline

log = logging.getLogger(__name__)


class Manager:
    def __init__(self):
        self._lock = threading.Lock()
        self.streams = {}          # camera id -> CameraStream
        pipeline.attach(watched=self.watched)
        self.sync()

    # ---------- keeping streams in step with the saved list ----------

    def sync(self):
        """Start, retarget or stop streams so they match what is saved.
        Restricted cameras decode every frame; counting cameras only
        keyframes, unless somebody is watching them live."""
        saved = {c["id"]: c for c in config.cameras()}
        gone = []
        with self._lock:
            for cam_id in list(self.streams):
                if cam_id not in saved:
                    gone.append(self.streams.pop(cam_id))
            for cam_id, cam in saved.items():
                full = cam.get("group", config.COUNT) == config.RESTRICTED
                stream = self.streams.get(cam_id)
                if stream is None:
                    stream = camera.CameraStream(cam_id, cam.get("url", ""),
                                                 on_frame=pipeline.on_frame,
                                                 full_rate=full)
                    stream.start()
                    self.streams[cam_id] = stream
                else:
                    stream.set_full_rate(full)
                    stream.set_url(cam.get("url", ""))
        for stream in gone:              # stopping can wait on the camera
            threading.Thread(target=stream.stop, daemon=True).start()

    def start(self):
        with self._lock:
            for stream in self.streams.values():
                stream.start()

    def stop(self):
        with self._lock:
            streams = list(self.streams.values())
        for stream in streams:
            stream.stop()

    def get(self, cam_id):
        return self.streams.get(cam_id)

    def watched(self, cam_id):
        stream = self.streams.get(cam_id)
        return bool(stream and stream.watched)

    # ---------- the Cameras page ----------

    def _template_url(self, skip_id=None):
        """An already working camera to copy the login and stream path from,
        so a second camera only needs its ip typed in."""
        for cam in config.cameras():
            if cam.get("id") != skip_id and cam.get("url"):
                return cam["url"]
        return ""

    def form_values(self, cam):
        """What one camera card shows. A camera with nothing saved yet is
        pre-filled from the first working camera."""
        url, ip = cam.get("url", ""), cam.get("ip", "")
        prefilled = False
        if not url:
            url = self._template_url(skip_id=cam.get("id"))
            ip = ""
            prefilled = bool(url)
        return {
            "id": cam.get("id"),
            "name": cam.get("name", ""),
            "group": cam.get("group", config.COUNT),
            "ip": ip,
            "url": rtsp.mask(url),
            "prefilled": prefilled,
            "zone": cam.get("zone") or [],
            "ignore": cam.get("ignore") or [],
        }

    def listing(self, group=None):
        return [self.form_values(cam) for cam in config.cameras(group)]

    def add(self, group=config.COUNT):
        if group not in config.GROUPS:
            group = config.COUNT
        cams = config.cameras()
        same = [c for c in cams if c.get("group", config.COUNT) == group]
        label = "Area" if group == config.RESTRICTED else "Camera"
        cam = {
            "id": config.next_camera_id(),
            "name": "{} {}".format(label, len(same) + 1),
            "group": group,
            "ip": "",
            "url": "",
        }
        cams.append(cam)
        config.set_cameras(cams)
        self.sync()
        return self.form_values(cam)

    def add_all_scanned(self, group=config.COUNT):
        """Add every camera the scan found that is not added yet, in one go.
        Does not wait for each one to connect - with 30 cameras that would
        take minutes; the page shows them coming online instead."""
        if group not in config.GROUPS:
            group = config.COUNT
        cams = config.cameras()
        taken = {rtsp.host_of(c.get("url", "")) for c in cams}
        label = "Area" if group == config.RESTRICTED else "Camera"
        added = []
        for found in scan.scanner.public()["found"]:
            ip = found["ip"]
            url = scan.scanner.found_url(ip)
            if not url or ip in taken:
                continue
            same = [c for c in cams if c.get("group", config.COUNT) == group]
            used = {c.get("id") for c in cams}
            n = 1
            while "cam{}".format(n) in used:
                n += 1
            cam = {"id": "cam{}".format(n),
                   "name": "{} {}".format(label, len(same) + 1),
                   "group": group, "ip": ip, "url": url}
            cams.append(cam)
            taken.add(ip)
            added.append(cam["id"])
        if added:
            config.set_cameras(cams)
            self.sync()
        return added

    def add_scanned(self, ip, group=config.COUNT):
        """Add a camera the network scan found. The scan already worked out a
        login and stream path that answered, so this just creates the camera
        pointing straight at it and waits to see a frame. Returns
        (ok, message, values)."""
        ip = (ip or "").strip()
        url = scan.scanner.found_url(ip)
        if not url:
            return False, "That camera is no longer in the scan results", {}
        if group not in config.GROUPS:
            group = config.COUNT
        for other in config.cameras():
            if rtsp.host_of(other.get("url", "")) == ip:
                return (False,
                        "Already added as {}".format(other.get("name", "another camera")),
                        {})
        cams = config.cameras()
        same = [c for c in cams if c.get("group", config.COUNT) == group]
        label = "Area" if group == config.RESTRICTED else "Camera"
        cam = {
            "id": config.next_camera_id(),
            "name": "{} {}".format(label, len(same) + 1),
            "group": group,
            "ip": ip,
            "url": url,
        }
        cams.append(cam)
        config.set_cameras(cams)
        self.sync()

        stream = self.streams.get(cam["id"])
        ok = stream.wait_for_frame(timeout=12) if stream else False
        status = stream.status() if stream else {}
        if ok:
            msg = "Connected - {}x{}".format(status["width"], status["height"])
        else:
            msg = status.get("error") or "Added, but no frame yet - check the camera"
        return ok, msg, self.form_values(cam)

    def remove(self, cam_id):
        cams = [c for c in config.cameras() if c.get("id") != cam_id]
        config.set_cameras(cams)
        self.sync()
        pipeline.forget(cam_id)
        return True

    def save(self, cam_id, ip_in, url_in, name=None, group=None):
        """Work out the url the user meant, store it and switch the stream
        over straight away. Returns (ok, message, values)."""
        cam = config.camera(cam_id)
        if cam is None:
            return False, "That camera was removed", {}

        saved_url = cam.get("url", "")
        ip_in = (ip_in or "").strip()
        submitted = (url_in or "").strip()

        # Nothing saved yet? Unmask against the camera the card was
        # pre-filled from, so its password carries over.
        secret_source = saved_url or self._template_url(skip_id=cam_id)

        if submitted.isdigit():
            url = submitted             # "0" = a USB webcam, handy for testing
        else:
            untouched = submitted == rtsp.mask(secret_source)
            url = rtsp.unmask(submitted, secret_source)

            if not url:
                if not ip_in:
                    return False, "Enter the camera IP first", self.form_values(cam)
                base = secret_source or rtsp.build(ip_in)
                url = rtsp.with_host(base, ip_in)
            elif untouched and ip_in and rtsp.host_of(url) != ip_in:
                # The usual case: only the IP box was changed.
                url = rtsp.with_host(url, ip_in)

            if not rtsp.parse(url):
                return (False, "That does not look like an RTSP link",
                        self.form_values(cam))

        for other in config.cameras():
            if other.get("id") != cam_id and other.get("url") == url:
                return (False,
                        "Already added as {}".format(other.get("name", "another camera")),
                        self.form_values(cam))

        cam["url"] = url
        cam["ip"] = rtsp.host_of(url) or ("USB webcam" if url.isdigit() else ip_in)
        if name is not None and name.strip():
            cam["name"] = name.strip()
        if group in config.GROUPS and group != cam.get("group", config.COUNT):
            cam["group"] = group
            pipeline.forget(cam_id)      # a count is not an alert - start over
        config.set_cameras([cam if c.get("id") == cam_id else c
                            for c in config.cameras()])
        self.sync()

        stream = self.streams.get(cam_id)
        ok = stream.wait_for_frame(timeout=12) if stream else False
        status = stream.status() if stream else {}
        if ok:
            msg = "Connected - {}x{}".format(status["width"], status["height"])
        else:
            msg = status.get("error") or "Could not connect - check the IP, password or cable"
        return ok, msg, self.form_values(cam)

    @staticmethod
    def _polygon(points):
        """[x, y] points clamped to 0-1, or [] if fewer than three."""
        clean = []
        for pt in points or []:
            try:
                x, y = float(pt[0]), float(pt[1])
            except (TypeError, ValueError, IndexError):
                continue
            clean.append([round(min(max(x, 0.0), 1.0), 4),
                          round(min(max(y, 0.0), 1.0), 4)])
        return clean if len(clean) >= 3 else []

    def _set(self, cam_id, key, value):
        cams = config.cameras()
        for cam in cams:
            if cam.get("id") == cam_id:
                cam[key] = value
        config.set_cameras(cams)

    def set_zone(self, cam_id, zone):
        """Store a restricted zone: [x, y] points as fractions of the picture
        (0-1). Fewer than three points means the whole picture."""
        clean = self._polygon(zone)
        self._set(cam_id, "zone", clean)
        return clean

    def set_ignore(self, cam_id, shapes):
        """Store a camera's ignore areas: a list of polygons, each like a
        zone. Anything whose middle falls inside one is never counted."""
        clean = [p for p in (self._polygon(s) for s in (shapes or [])[:20]) if p]
        self._set(cam_id, "ignore", clean)
        return clean

    def reset(self, cam_id):
        cams = config.cameras()
        for cam in cams:
            if cam.get("id") == cam_id:
                cam["ip"] = cam["url"] = ""
        config.set_cameras(cams)
        self.sync()
        return self.form_values(config.camera(cam_id) or {})

    # ---------- dashboard ----------

    def status(self):
        out = {}
        for cam in config.cameras():
            cam_id = cam["id"]
            stream = self.streams.get(cam_id)
            st = stream.status() if stream else {"state": "off", "configured": False}
            st["name"] = cam.get("name", cam_id)
            st["id"] = cam_id
            st["ip"] = cam.get("ip", "")
            st["group"] = cam.get("group", config.COUNT)
            st["zone"] = bool(cam.get("zone"))
            st["ignore"] = cam.get("ignore") or []
            out[cam_id] = st
        return out


manager = Manager()
