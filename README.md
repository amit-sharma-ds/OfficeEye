# OfficeEye

Office camera app for 30+ IP cameras:

- **Counting cameras** - how many people each camera sees right now, and who
  is **using a phone** (with a photo).
- **Restricted cameras** - anybody who steps into a forbidden zone raises an
  **alert straight away**, with a beep on every open page and **photos of the
  person**.

It counts **people in view** - not a doorway in/out counter.

## Run it

Double-click `run.bat`, or:

```bash
py -3.12 app.py
```

Then open **http://localhost:8000** - from another PC in the office use this
machine's IP instead of `localhost`. The page is up at once; the detection
model takes 15-20 seconds to load in the background.

Only one copy can run at a time; a second one refuses to start rather than
fighting over the port. First time on a new PC: `py -3.12 -m pip install -r requirements.txt`.
The detection models are not in the repository; the chosen one (YOLO26s by
default) downloads itself on first start.

## Deploying

OfficeEye reads the cameras over the office network, so the real work runs
on a machine **inside that network** - the office PC. A cloud host cannot
reach cameras at 192.168.x.x addresses.

**Cloud view (open it from anywhere).** The same code runs on Render as a
read-only copy behind a password, and the office PC *sends* it what it sees
(`cloud.py`): the live status every few seconds, the alert, phone and count
records with their photos, the running-time records, and camera pictures -
only of cameras somebody is looking at there. Camera links and passwords
never leave the office.

1. Render: **New + -> Blueprint** -> pick this repository (`render.yaml`).
   A Render service runs as the cloud view by itself.
2. Office PC: **Settings -> Cloud view** - put in the Render link, press
   **Save**, **Copy** the push token.
3. Render -> the service -> **Environment**: add `OFFICEEYE_PUSH_TOKEN` (the
   token) and `OFFICEEYE_PASSWORD` (a password you choose for logging in).
4. Office PC: set **Send to the cloud** to *On*.

The bar at the top of the cloud view says when the office PC last sent
data; if it stops (PC off, no internet) the records stay but nothing is
live. Render's free plan wipes its disk on a restart - the office PC then
sends the last three days again by itself. While somebody watches, it uses
about 0.5-3 MB a minute of the office upload; with nobody watching, almost
nothing.

## Setting up 30 cameras

1. **Cameras** page -> **Scan network**. Every camera on the LAN is found
   with a login and stream path that works.
2. Pick *Add as: Counting* and press **Add all new**. All of them are added
   in one go and come online over the next few seconds.
3. For the restricted cameras, change **Purpose** to *Restricted* in the
   list and press **Save** (or scan again with *Add as: Restricted*).
4. **Restricted area** page -> **Edit zone** on each camera. Click around the
   floor area nobody may enter. A person counts as inside when their **feet**
   are inside the shape; with no shape the whole picture is watched.

A camera that needs a password or another stream path: press **...** on its
row and paste the full RTSP link. From then on the box shows the password as
`***` - leave the `***` alone and change only the IP when the camera moves.
Put `0` in the link box to use a USB webcam for testing.

## The pages

**Dashboard** - records only, no camera pictures: people in view now and
today's peak, phones, restricted alerts (red while someone is inside), cameras
online. Below: each restricted camera's status, the latest alerts and phone
photos, people hour by hour, and people by camera.

**Live cameras** - the camera wall: one tile per camera with its count, phone
use and an alert outline. Filter by *Counting / Restricted / People in view /
On phone / Offline*, or type a name. **Click a tile to watch it live.**

**Camera running time** - how many hours each camera ran and how many it was
off, hour by hour, checked every 10 seconds. Every finished day is saved as a
daily record per camera and kept for good (the hourly detail 90 days); both
download as CSV. Time the app itself was closed shows as *app not running*.

**Restricted area** - the restricted cameras, their zones, and every alert
by day. Each alert keeps up to 4 snapshots while the person is there - the
whole picture with the zone, boxes and time stamp, plus a close-up of every
person. Click an alert to flick through them.

- The tile turns red, the sidebar badge shows on **every** page, the browser
  tab title changes to `(n) ALERT`, and a pop-up with the person's photo
  appears bottom right.
- **Alert sound** (bottom of the sidebar) turns the beep on or off for that
  browser. Browsers only play sound after the page was clicked once.
- Somebody coming back within 15 seconds continues the same alert; it ends
  once the zone has looked empty for 5 seconds.

**Washroom** - who went into the washroom and when, from a camera in the
corridor outside the door (it must never see inside). Set that camera's
Purpose to *Washroom door*, then draw the **door zone** here: the door and the
floor right in front of it. A cheap check on every frame notices the door
moving; the model then looks twice a second. Somebody walking up just before
the door moved *went in*; the door opening with nobody outside means somebody
*came out* - paired with the in, which gives the time inside. Each keeps a
photo of the person at the door. Mark **Checked - clean** / **Found dirty** /
**Found damaged** and the page lists who used it in between.

**Phone use** - who is on a phone right now and a photo of every time
somebody was caught, by day and camera. A phone only counts when it is
raised (held up to read or to the ear) and seen twice in a row; at most one
photo per camera per minute.

**Cameras** - every camera in one list: name, IP, purpose, live status.

**Ignore areas** - a poster, a TV or a mannequin that keeps being counted as
a person: open the camera live, press **Ignore areas** and drag a box over it
(also under **...** on the Cameras page). Nothing whose middle is inside a
box is counted. Grey "ignored" boxes show on the live view.

**Steady counts** - from a ceiling, somebody sitting still is found in one
check and missed in the next. Each person is remembered across checks: a
missed person stays counted until missed twice and for 4 seconds, and a weak
first sighting only counts once it is seen again. The confidence cutoff is
0.3 (Settings) - low on purpose, because these two keep the false boxes out.

**Settings** - live engine load, detection settings, how long photos are
kept, office timing.

## How it keeps up with 30 cameras

Decoding video costs more CPU than reading it, and one PC cannot decode every
frame of 30 cameras. So:

| | Video decoded | Checked by the model |
| --- | --- | --- |
| Restricted camera | every frame | as soon as anything moves, up to 4x a second; nonstop while an alert is open |
| Camera open in live view | every frame | every 0.5 s while it changes |
| Counting camera | keyframes only (~1 every 2 s) | when the picture changed, else every 30 s |

One model is shared by all cameras and checks them in that order of
priority, a few frames per batch. A picture that has not changed is not
checked again, so an empty corridor costs nothing. **Settings -> How hard the
engine is working** shows the load and how quickly each kind of camera is
seen.

### What this PC can do

Measured on this machine (Intel i3-3240, 2 cores, no GPU):

- one 640x480 camera: 0.044 CPU cores in keyframe mode, 0.145 decoding every
  frame;
- the model: YOLO26s ~600 ms per picture at size 480 (about 1.6 a second);
  YOLO26n ~320 ms, the old YOLO11n ~420 ms. YOLO26s is the default because
  from the ceiling it finds about twice the people (13 of 18 on office
  frames, against 7 for YOLO11n). Pick the model under **Settings**.

That is enough for restricted alerts within about a second and counts that
refresh every few seconds - but when many of the 30 cameras are busy at once
the counting cameras wait their turn (up to ~10 s). For every camera in real
time at once, run it on a PC with an **NVIDIA GPU** (an RTX 3060 is plenty):
install the CUDA build of PyTorch and the app uses the GPU by itself with
bigger batches. A newer Intel/AMD CPU (with AVX2) is also 3-4x faster than
this one. Keep the app's PC free of other heavy programs.

## Adding another detection

Write a class in `detection.py` and that is all:

```python
@register
class HelmetDetector(Detector):
    key = "helmet"
    title = "Helmet check"
    groups = (config.COUNT,)        # which cameras it runs on

    def process(self, camera, frame, ts, seen):
        # seen.people / seen.phones: [(x1, y1, x2, y2, score), ...]
        return [{"kind": "no_helmet", "label": "gate", "meta": {"conf": 0.82}}]

    def overlay(self, camera, frame):
        return frame            # draw here, it shows on the live view
```

Anything `process` returns is written to the event log; anything `overlay`
draws appears on the live video.

## Files

| File | What it does |
| --- | --- |
| `app.py` | pages, JSON API, camera-wall images, live video, photos |
| `camera.py` | one RTSP stream: connect, reconnect, keyframe / full decoding |
| `cameras.py` | the camera list: save / add / scan-add-all / zones / remove |
| `detection.py` | the shared model, the scheduler, people / phone / restricted detections |
| `photos.py` | evidence photos and their clean-up |
| `rtsp.py` | reading, building and masking RTSP links |
| `scan.py` | finding cameras on the network |
| `events.py` | SQLite event log |
| `config.py` | settings, stored in `data/config.json` |
| `yolo26s.pt` | the detection model, downloaded on first start (choose another under Settings) |
| `yolo26n.pt`, `yolo26n-objv1-150.pt`, `yolo11n.pt`, `yolo11s.pt` | faster / older models to fall back to, if downloaded |
| `tools/` | training helpers: collect frames, pre-label, review labels |
| `washroom.py` | washroom visits: the door watch and who went in / came out |
| `cloud.py` | the cloud view: login and read-only copy on Render, and the office uplink |
| `Dockerfile`, `render.yaml` | container and Render deploy (see Deploying) |
| `data/` | config, `events.db` and `photos/` - back this up, do not commit it |

## Notes

- Streams are pulled over RTSP/TCP with PyAV (FFmpeg).
- A camera that drops is retried on its own with a growing delay, so a reboot
  or a flaky cable recovers without anyone touching the app.
- Photos are kept 30 days by default (Settings), events 90 days.
- Camera passwords sit in `data/config.json` in plain text, the way the
  camera itself needs them. Keep that folder off any shared drive.
- `app.py` runs Flask's development server - fine for a few people on the
  office LAN. If it ever needs to serve more, put waitress in front of it.
