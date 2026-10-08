# Phase 1 prototype

A throwaway Python script that checks whether head pose alone can tell your
monitors apart from where your camera sits. It is not part of the shipped tool.

Frames stay in memory and are never saved. Recordings in `data/` contain only
pose numbers: yaw, pitch, roll and face brightness.

## Setup

```sh
cd prototype
mkdir -p models
curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
sha256sum models/face_landmarker.task
# 64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff
```

`uv run` creates the virtual environment on first use.

## Commands

```sh
uv run proto.py live                # live yaw and pitch, Ctrl+C to stop
uv run proto.py record              # guided recording, then analysis
uv run proto.py analyze data/session-*.json
uv run proto.py classify data/session-*.json   # live monitor prediction
```

`record` reads your monitors from Hyprland and walks through each one three
times, alternating the order. For each step you get a desktop notification and
3 seconds to turn your head, then 2 seconds of recording. Look at the center of
the named monitor the way you normally would.

Options: `--rounds`, `--seconds`, `--lead`, `--camera-monitor` (defaults to the
rightmost monitor), and `--device` for a camera other than `/dev/video0`.

## Reading the analysis

- **face%** is how often a face was found. Low numbers at large yaw mean the
  model is losing you.
- **yaw sd / pit sd** is how much the pose jitters while you hold still.
- **per-round** centroids show whether you land in the same place each time you
  come back to a monitor. That matters more than jitter.
- **ratio** compares the distance between two monitors to their combined
  spread. Above 3 is comfortable, 1.5 to 3 should work with smoothing, and
  below 1.5 means those two monitors will be confused.
- **accuracy** trains a nearest-centroid classifier on one round and tests it on
  the others.

## Known issue

MediaPipe 1.1.0 imports `sounddevice` when it builds a vision task. On a
PipeWire system that goes through ALSA's JACK plugin and the process gets
killed. `proto.py` blocks that import before loading MediaPipe.
