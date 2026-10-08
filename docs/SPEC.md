# lookfocus design brief

The original brief lookfocus was built from, kept as the record of scope.
Decisions and measurements made while building it are in `NOTES.md`. Two
things changed along the way and are noted inline: the typing lock was
dropped, and adaptive centroids were added.

## What it does

A background daemon watches the user through a webcam, estimates head pose,
works out which monitor the user is facing, and after a short dwell switches
Hyprland focus and the cursor to that monitor. It replaces dragging the mouse
across screens.

## Setups to support

Nothing may be hardcoded to one setup. The development setup was three
monitors in a row with the webcam on the rightmost one, but the camera can be
on any monitor, and layouts can be two monitors, a row of three, a vertical
stack, or a grid. Never assume the camera is centered.

## Decisions made up front

- Language: Rust, shipped as a systemd user service plus a small CLI.
- Tracking: head pose only (yaw and pitch) from face landmarks. No eye gaze in
  v1.
- Landmarks: a MediaPipe face landmark model run through ONNX Runtime (the
  `ort` crate). Check the current best model and crate versions before
  choosing.
- Capture: about 15 fps at low resolution (`nokhwa` or `v4l`, picked after
  checking what works on Arch).
- Smoothing: One Euro filter on yaw and pitch.
- Monitor classification: nearest-centroid on (yaw, pitch) using per-monitor
  samples recorded during calibration, not fixed left/right thresholds.
  Hysteresis, so leaving a monitor needs a bigger change than entering one.
- Dwell: switch only after the pose stays in a new zone for about 300 ms
  (configurable).
- Action: through Hyprland IPC, focus the monitor and then move the cursor to
  the last known position on that monitor. Check the exact dispatchers against
  current Hyprland docs, and check whether focusing already warps the cursor,
  so it is not warped twice.
- Overrides:
  1. Physical mouse movement pauses tracking for about 2 s.
  2. A pause/toggle command (`lookfocus toggle`) that can be bound to a
     Hyprland keybind.
  3. Look-away pause: if the face is lost or pitch shows the user looking down
     at the keyboard or a phone, hold the current monitor.
  4. Typing lock: while typing, lock focus so a head turn mid-sentence does
     not steal it. **Dropped during the build**: lookfocus must never read
     keystrokes, so it reads no input devices at all.

## Guided calibration (core feature)

`lookfocus calibrate`:

1. Read monitors from Hyprland. Ask which monitor the camera is mounted on,
   defaulting to a best guess, and store it as a hint.
2. For each monitor, prompt the user to look at its center, capture about 2
   seconds of pose samples, and store the median yaw/pitch plus spread.
3. Quality checks with clear messages: warn if two monitors' centroids are too
   close to separate reliably, if a sample was noisy, if extreme yaw (beyond
   roughly 45 to 60 degrees) makes landmarks unreliable, or if lighting is
   poor. Suggest fixes (move the camera, recalibrate, future gaze assist).
4. Save to a TOML file under the XDG config directory.

`lookfocus recalibrate` runs it again. Recorded samples always win over the
camera-monitor hint.

## CLI

calibrate, recalibrate, run (daemon in the foreground), toggle, status, debug
(live yaw/pitch/zone in the terminal).

## Phases (in order, with a report between each)

- Phase 0: plan. Inspect the machine and the repository, check current crate
  and Hyprland docs, write a short plan with risks.
- Phase 1: a throwaway Python prototype (MediaPipe and OpenCV) that prints
  live yaw and pitch and records per-monitor samples, to check that the angle
  spread from an off-center camera is separable before the Rust build.
- Phase 2: Rust core: capture, landmarks, head pose, filter, classifier, with
  unit tests on recorded or synthetic pose data.
- Phase 3: Hyprland integration, calibration flow, config, dwell and
  hysteresis.
- Phase 4: overrides, systemd user service, CLI polish.
- Phase 5: open source release prep: README with a demo section, install
  instructions for Arch and Omarchy (AUR PKGBUILD), config reference,
  troubleshooting, license (MIT or Apache-2.0), CI that builds and runs tests,
  and docs for the event stream below.

Added during the build: adaptive centroids (switchable), and an Omarchy bar
button to control lookfocus.

## Designed for, not fully built in v1

An event stream on a Unix socket reporting the active monitor and zone
changes, so other tools can react to where the user is looking. The internal
architecture is event-based so this is easy to complete.

## Engineering requirements

- Light on CPU when idle. Release the camera when paused or when nobody has
  been in frame for a while.
- Privacy: never store or transmit frames. Process in memory only, and say so
  in the README.
- Mock the Hyprland IPC and the camera so the logic is testable without
  hardware.
- Handle failure cleanly: camera busy, Hyprland socket missing, model file
  missing, monitor hotplug (re-read monitors and prompt to recalibrate if the
  layout changed).
- Do not guess API details from memory. Check current docs for crates, ONNX
  model sources and Hyprland dispatchers.
- Small, meaningful commits, and a running `NOTES.md` of decisions and open
  questions.

## Style

Docs and comments in plain natural language, with no em dashes anywhere.
