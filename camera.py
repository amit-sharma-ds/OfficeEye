"""RTSP camera reader.

One CameraStream per camera. It owns a background thread that keeps the
connection alive, always holds the newest frame, and reconnects on its own
when the camera drops or its ip changes. Nothing else in the app ever opens
a camera.

Decoding is what costs CPU, not reading. With 30+ cameras one machine cannot
decode every frame of every camera, so each stream works in one of two modes:

- keyframes only: every packet is read (cheap), but only the camera's
  keyframes - roughly one every 1-2 seconds - are decoded. Plenty for
  counting people, and about a third of the CPU of full decoding.
- every frame: full video. Used for restricted-area cameras (an intruder must
  not slip between two keyframes) and for any camera somebody is watching
  live in the browser.

A stream switches between the two on its own; going up waits for the next
keyframe so the picture is never smeared.
"""

import logging
import threading
import time

import av
import cv2

log = logging.getLogger(__name__)

OPEN_TIMEOUT = 8.0          # seconds
READ_TIMEOUT = 8.0
BACKOFF_START = 2.0
BACKOFF_MAX = 15.0
WATCH_GRACE = 4.0           # keep full video this long after the last viewer leaves

OFF = "off"
CONNECTING = "connecting"
CONNECTED = "connected"
ERROR = "error"


class CameraStream:
    def __init__(self, name, url="", on_frame=None, full_rate=False):
        self.name = name
        self._url = (url or "").strip()
        self._on_frame = on_frame        # called with (name, frame, ts, seq)
        self._full_rate = bool(full_rate)

        self._lock = threading.Lock()
        self._frame = None
        self._frame_ts = 0.0
        self._seq = 0                    # bumps on every new frame

        self._state = OFF if not self._url else CONNECTING
        self._error = ""
        self._size = (0, 0)
        self._fps = 0.0                  # what the camera sends
        self._decoded_fps = 0.0          # what we actually decode
        self._full = False               # decoding every frame right now?
        self._connected_since = 0.0

        self._viewers = 0
        self._last_view = 0.0

        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread = None

    # ---------- lifecycle ----------

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="cam-" + self.name,
                                        daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=READ_TIMEOUT + 2)
        self._thread = None

    def set_url(self, url):
        """Point the camera somewhere else. Takes effect within a second,
        no restart of the app needed."""
        url = (url or "").strip()
        with self._lock:
            if url == self._url:
                return False
            self._url = url
            self._frame = None
            self._frame_ts = 0.0
            self._size = (0, 0)
            self._state = OFF if not url else CONNECTING
            self._error = ""
            self._fps = self._decoded_fps = 0.0
            self._connected_since = 0.0
        self._wake.set()              # drop the current connection now
        self.start()
        return True

    def set_full_rate(self, full_rate):
        """Restricted cameras decode every frame, counting cameras only
        keyframes. Changes when a camera is moved between the two."""
        self._full_rate = bool(full_rate)

    # ---------- live viewers ----------

    def watch(self):
        """Somebody opened this camera's live video: decode every frame."""
        with self._lock:
            self._viewers += 1
            self._last_view = time.time()

    def unwatch(self):
        with self._lock:
            self._viewers = max(0, self._viewers - 1)
            self._last_view = time.time()

    @property
    def watched(self):
        with self._lock:
            return self._viewers > 0 or time.time() - self._last_view < WATCH_GRACE

    def _wants_full(self):
        return self._full_rate or self.watched

    # ---------- what the rest of the app reads ----------

    @property
    def url(self):
        with self._lock:
            return self._url

    @property
    def is_connected(self):
        with self._lock:
            return self._state == CONNECTED

    def status(self):
        with self._lock:
            age = time.time() - self._frame_ts if self._frame_ts else None
            return {
                "name": self.name,
                "state": self._state,
                "error": self._error,
                "width": self._size[0],
                "height": self._size[1],
                "fps": round(self._fps, 1),
                "decoded_fps": round(self._decoded_fps, 1),
                "mode": "full" if self._full else "key",
                "watched": self._viewers > 0,
                "frame_age": round(age, 1) if age is not None else None,
                "seq": self._seq,
                "up_for": int(time.time() - self._connected_since) if self._connected_since else 0,
                "configured": bool(self._url),
            }

    def latest(self):
        """(frame, ts, seq) of the newest frame - the frame itself is shared,
        so copy it before drawing on it."""
        with self._lock:
            return self._frame, self._frame_ts, self._seq

    def read(self):
        """Newest frame, or None. Returns a copy - callers may draw on it."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def wait_for_frame(self, timeout=8.0):
        """Block until the first frame arrives. Used right after Save & connect
        so the user gets a real answer instead of a spinner. A camera in
        keyframe mode can take up to one keyframe interval."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._frame is not None:
                    return True
                if self._state in (ERROR, OFF):
                    return False
            time.sleep(0.2)
        return False

    # ---------- the worker ----------

    def _set_state(self, state, error=""):
        with self._lock:
            self._state = state
            self._error = error
            if state != CONNECTED:
                self._connected_since = 0.0
                self._fps = self._decoded_fps = 0.0
                self._full = False

    def _run(self):
        backoff = BACKOFF_START
        while not self._stop.is_set():
            url = self.url
            if not url:
                self._set_state(OFF)
                self._wake.wait(1.0)
                self._wake.clear()
                continue

            self._set_state(CONNECTING)
            self._wake.clear()
            try:
                if url.isdigit():                # "0" = local webcam, handy to test
                    self._pump_webcam(int(url), url)
                else:
                    self._pump_rtsp(url)
                backoff = BACKOFF_START
            except Exception as exc:                      # noqa: BLE001
                msg = _friendly(exc)
                log.warning("[%s] %s", self.name, msg)
                self._set_state(ERROR, msg)

            if self._stop.is_set():
                break
            if self.url != url:          # url changed - reconnect at once
                continue
            self._wake.wait(backoff)
            self._wake.clear()
            backoff = min(backoff * 2, BACKOFF_MAX)

    def _alive(self, url):
        return not self._stop.is_set() and self.url == url and not self._wake.is_set()

    def _publish(self, url, frame, now):
        """Make a decoded frame the newest one. Returns False if the camera
        was pointed elsewhere meanwhile - publishing then would show the old
        camera as if it were the new one."""
        with self._lock:
            if self._url != url or self._stop.is_set():
                return False
            self._frame = frame
            self._frame_ts = now
            self._seq += 1
            seq = self._seq
            self._size = (frame.shape[1], frame.shape[0])
            if self._state != CONNECTED:
                self._state = CONNECTED
                self._error = ""
                self._connected_since = now
        if self._on_frame:
            try:
                self._on_frame(self.name, frame, now, seq)
            except Exception:                              # noqa: BLE001
                log.exception("[%s] frame handler failed", self.name)
        return True

    def _pump_rtsp(self, url):
        container = av.open(url, options={"rtsp_transport": "tcp",
                                          "fflags": "nobuffer",
                                          "flags": "low_delay"},
                            timeout=(OPEN_TIMEOUT, READ_TIMEOUT))
        try:
            vs = container.streams.video[0]
            # One decoder thread per camera: with dozens of cameras, many
            # single-threaded decoders beat many multi-threaded ones.
            vs.codec_context.thread_count = 1
            full = False
            packets = decoded = 0
            tick = time.time()

            for packet in container.demux(vs):
                if not self._alive(url):
                    return
                if packet.size == 0:
                    continue
                packets += 1
                key = packet.is_keyframe

                want = self._wants_full()
                if want and not full and key:
                    full = True              # step up only on a keyframe
                elif not want and full:
                    full = False
                if not full and not key:
                    continue                 # read, but do not decode

                for frame in vs.codec_context.decode(packet):
                    img = frame.to_ndarray(format="bgr24")
                    decoded += 1
                    if not self._publish(url, img, time.time()):
                        return

                now = time.time()
                if now - tick >= 2.0:
                    with self._lock:
                        self._fps = packets / (now - tick)
                        self._decoded_fps = decoded / (now - tick)
                        self._full = full
                    packets = decoded = 0
                    tick = now
            raise RuntimeError("stream ended")
        finally:
            container.close()

    def _pump_webcam(self, index, url):
        cap = cv2.VideoCapture(index)
        try:
            if not cap.isOpened():
                raise RuntimeError("could not open the USB webcam")
            fails, ticks, tick = 0, 0, time.time()
            while self._alive(url):
                ok, img = cap.read()
                if not ok or img is None:
                    fails += 1
                    if fails >= 30:
                        raise RuntimeError("webcam stopped sending frames")
                    time.sleep(0.05)
                    continue
                fails = 0
                ticks += 1
                if not self._publish(url, img, time.time()):
                    return
                now = time.time()
                if now - tick >= 2.0:
                    with self._lock:
                        self._fps = self._decoded_fps = ticks / (now - tick)
                        self._full = True
                    ticks, tick = 0, now
        finally:
            cap.release()


def _friendly(exc):
    """FFmpeg errors are cryptic - say what to check instead."""
    text = str(exc)
    low = text.lower()
    if "401" in low or "unauthorized" in low:
        return "wrong user name or password"
    if "404" in low or "not found" in low:
        return "camera refused the stream path (or is busy) - retrying"
    if "timed out" in low or "timeout" in low or "immediate exit" in low:
        return "no answer from the camera - check the IP and cable"
    if "refused" in low:
        return "camera refused the connection - is RTSP enabled?"
    if "stream ended" in low:
        return "stream stopped sending video"
    return text[:160]
