# Notes

A running log of decisions, verified facts, and open questions. Newest phase at
the top of each section. The project brief is in `docs/SPEC.md`.

## Hand tracking (2026-10-09, not yet run on real models)

Written and tested with fakes only, without network access, so nothing in
this section has been checked against the model files or a camera yet.

- Models: OpenCV Zoo's ONNX conversions of MediaPipe's palm detector
  (`opencv/palm_detection_mediapipe@233e619`, saved as `hand_detector.onnx`,
  sha256 78ff51c3...) and hand landmark model
  (`opencv/handpose_estimation_mediapipe@4b2a0b4`, saved as
  `hand_landmarks.onnx`, sha256 db0898ae...). Apache-2.0. They are not on the
  `models-v1` release yet, so `scripts/fetch-models.sh` and the PKGBUILD fetch
  them from Hugging Face directly.
- Palm detector, as expected from OpenCV Zoo's code: input 1x192x192x3 RGB in
  [0, 1], letterboxed. Outputs 2016 x 18 (box center, size, then 7 keypoints)
  and 2016 x 1 score logits. Anchors are BlazeFace's scheme at 192 px (stride
  8 with 2 per cell, stride 16 with 6 per cell). The code tells the outputs
  apart by size, not by name.
- Landmark model: input 1x224x224x3 RGB in [0, 1]. Outputs read by position:
  21 x 3 landmarks in crop pixels, hand confidence, handedness, world
  landmarks (unused). A score outside [0, 1] is treated as a logit and passed
  through a sigmoid. Both hand models log their output names and shapes at
  load, so the first real run confirms or corrects this layout.
- To check on the first real run: the logged output layout, that an open hand
  gives landmarks that line up with it, and which side the handedness score
  means. MediaPipe's hand models assume a mirrored (selfie) image, and
  lookfocus does not mirror frames.

## Verified facts (Phases 4 and 5, 2026-10-08)

- Installed on the development machine with `scripts/install-local.sh`:
  binary in `~/.local/bin`, models in `~/.local/share/lookfocus/models`,
  `~/.config/systemd/user/lookfocus.service` enabled under
  `graphical-session.target` (this Omarchy session is managed by uwsm, which
  exports `HYPRLAND_INSTANCE_SIGNATURE`, `WAYLAND_DISPLAY` and the D-Bus
  address to user services).
- Under systemd: tracking works, `lookfocus pause` releases `/dev/video0`
  (nothing holds it afterwards), `resume` reopens it.
- CPU with the idle frame rate: about 20% of one core over 15 s, down from
  37% at a constant 20 fps.
- Bar widget id `sasukevibes.lookfocus`, placed in the right section with
  `omarchy bar put`. It renders as the eye glyph plus the focused monitor
  name. The shell logs only the usual per-bar IpcHandler warning for it.
- Keybind `SUPER + ALT + E` added to `~/.config/hypr/bindings.lua` with
  `o.bind` (Omarchy's helper), after a backup. `hyprctl configerrors` is
  empty.
- `set -o pipefail` with `ldconfig -p | grep -q` reports a false failure,
  because `grep -q` exits early and `ldconfig` gets SIGPIPE. The install
  script checks the library file directly instead.
- Omarchy's floating terminal helper runs its command with `bash -c` inside
  the terminal, so the bar resolves the full path of `lookfocus` before
  launching recalibration.

## Verified facts (Phase 3, 2026-10-08)

### First live run (2026-10-08)

- Calibration: DP-1 +40.4, DP-2 +28.3, eDP-2 +22.2 yaw, face found 100%.
  Adjacent gaps 12.1 and 7.5 degrees, auto hysteresis about 1.9. The strict
  calibration check flagged both pairs, and look-down was only 3.7 degrees
  below the screens, so it is not separable by pitch for this user.
- `lookfocus run` with default settings: the user reports it works really
  well. So the calibration check's "too close" threshold is conservative in
  practice, because the daemon smooths poses before deciding.

### Hyprland 0.56.2 behavior (from its source at tag v0.56.2 and live tests)

- Socket requests work as `hyprctl` sends them: `j/monitors`, `j/version`,
  `j/cursorpos`, `j/getoption <name>`, and `dispatch <lua>`. A batch is
  `[[BATCH]]cmd1 ; cmd2` and replies `ok` once per command, separated by
  blank lines.
- Errors are not signalled by the connection. They come back as text, for
  example `warning: =[C]:-1: hl.focus.monitor: monitor not found`. Anything
  other than `ok` lines is treated as a failure.
- `cursor:warp_on_monitor_change` does not exist in 0.56.2 ("no such option").
  The wiki documents a newer version. Do not rely on it.
- `hl.dsp.focus({ monitor })` calls `tryMoveFocusToMonitor`, which always
  moves the cursor: to the middle of the monitor's focus candidate window
  (or to the remembered spot in it if `cursor:persistent_warps` is set), or to
  the monitor's middle when it has no windows. Only `cursor:no_warps` affects
  this.
- `hl.dsp.cursor.move({ x, y })` warps and then simulates mouse movement, so
  with `follow_mouse = 1` focus follows the window under the new position.
- A batch is handled in one go before the next frame is drawn, so sending
  focus and cursor move together hides Hyprland's intermediate warp.
- `j/version` returns both `version` ("0.56.2") and `tag` ("v0.56.2").

## Verified facts (Phase 2, 2026-10-08)

### Models

- The ONNX models are pinned Hugging Face revisions of MediaPipe's own models:
  `fernandotonon/QtMeshEditor-blazeface-onnx@50f2c66` (`face_detector.onnx`,
  sha256 02a04d5d...) and `fernandotonon/QtMeshEditor-facemesh-onnx@3271408`
  (`face_landmarks.onnx`, sha256 d16e5a55...). Both are opset 18 and use only
  basic operators (Conv, PRelu, Add, Pad, MaxPool, Reshape and similar).
- Checked against Google's original TFLite files from `face_landmarker.task`,
  run with LiteRT on the same four inputs. The largest difference was 0.0005
  on landmarks measured in 256 px crop units, and about 0.0003 on detector
  outputs. They are the same models.
- Landmark model outputs: `Identity` (478 x 3 landmarks in crop pixels),
  `Identity_1` (face presence logit), `Identity_2` (an extra score we ignore).
- `face_landmarker.task` also contains `geometry_pipeline_metadata_landmarks.binarypb`:
  the 468-vertex canonical face and 33 weighted landmarks MediaPipe uses for
  its pose fit. `scripts/gen_canonical.py` turns it into `src/pose/canonical.rs`.

### Runtime

- `ort` 2.0.0-rc.13 with `load-dynamic` loads Arch's `onnxruntime-cpu` 1.29.0
  from `/usr/lib/libonnxruntime.so` without problems. ort only rejects
  libraries older than the version it was built for.
- Inference with one thread per model: 13.7 ms median, 18 ms at the 95th
  percentile, for the landmark model plus the pose fit.
- CPU: 3.7 s of user time over 10 s at 20 fps, about 37% of one core. Most of
  it is inference. Lowering the rate while the head is still is the obvious
  saving (Phase 4).

### Camera behavior

- This webcam has `exposure_dynamic_framerate` on, so in dim light it drops
  from 30 to about 20 fps by itself. Plain "skip every other frame" then gives
  10 fps. The frame picker now measures the real arrival rate and keeps the
  frame closest to each due time, which gives 19.8 fps here. lookfocus does not
  change camera controls, because that would affect other apps.

### Environment

- Shells started outside Hyprland (Claude Code's `!` commands, and likely
  systemd user services) have no `HYPRLAND_INSTANCE_SIGNATURE`, so `hyprctl`
  fails. Fall back to the newest directory in `$XDG_RUNTIME_DIR/hypr/`. The
  Phase 3 IPC code must do this too.

### Rust pipeline recording vs Python (same guided recording, same analysis)

The first Rust recording looked broken (all monitors near +38 yaw, 25%
accuracy). It was not a pipeline bug: the prompts named monitors only by
connector (DP-1) and did not say left, center or right. With position words in
the prompts:

| Monitor | Rust yaw | Python yaw | Rust pitch | Python pitch |
|---------|----------|------------|------------|--------------|
| DP-1 (left)   | +41.1 | +44.8 | +0.5 | +6.3 |
| DP-2 (center) | +30.4 | +27.3 | -0.8 | +3.7 |
| eDP-2 (right) | +25.2 | +10.2 | -3.6 | +1.9 |

- Rust: 99.3% raw and 100% smoothed nearest-centroid accuracy across rounds.
  Separation ratios 2.9 (left vs center), 3.5 (center vs right), 5.9.
- Python had ratios 5.2, 8.0 and 15.0. The Rust yaw range is about half as
  wide (16 degrees from right to left, against 35). Looking at the camera
  monitor reads +25 in Rust and +10 in Python.
- Suspected cause was the Rust fit compressing yaw. Ruled out: MediaPipe
  scales landmark z exactly as we do (`normalize_z` is 1, z is scaled by the
  crop width), and a recording that ran both fits on the same frames showed no
  compression:

  | Monitor | MediaPipe yaw | Rust fit yaw | MediaPipe pitch | Rust fit pitch |
  |---------|---------------|--------------|-----------------|----------------|
  | DP-1    | +43.0         | +45.3        | +4.6            | +1.9           |
  | DP-2    | +31.2         | +33.7        | +2.2            | -1.4           |
  | eDP-2   | +22.9         | +24.8        | +1.9            | -1.9           |

  The orthographic fit is offset by about +2 yaw and -3.5 pitch, with the same
  spread. Calibration is relative, so that is harmless.
- The real cause was behavior. The first Python session had deliberate, large
  head turns (camera monitor at +10). Every later session, Python or Rust, put
  the camera monitor at +23 to +25.
- Natural head turns put adjacent monitors only 8 to 12 degrees apart on this
  setup (separation ratios 2.2 to 2.4, raw accuracy 94.5%). The decision layer
  in Phase 3 (hysteresis, settle detection, dwell, adaptive centroids) has to
  carry this, not the pose fit.

### Pose numbers from the Rust pipeline

- Looking at DP-2 gave yaw +30.5 and pitch -5.3, with 1.1 and 0.9 degrees of
  jitter. Python MediaPipe gave about +27 and +4 for the same monitor. The
  pitch offset is expected: the Rust fit is orthographic while MediaPipe solves
  with perspective. Calibration is relative, so a constant offset is harmless.

## Verified facts (Phase 1, 2026-10-08)

### Recording on the development machine

Three rounds, 2 seconds per monitor, camera on eDP-2 (rightmost). Degrees.

| Monitor | Face found | Yaw   | Pitch | Yaw sd | Pitch sd | Yaw per round     |
|---------|------------|-------|-------|--------|----------|-------------------|
| DP-1    | 100%       | +44.8 | +6.3  | 1.81   | 0.83     | +46.8 +44.2 +44.3 |
| DP-2    | 100%       | +27.3 | +3.7  | 1.64   | 0.54     | +28.3 +24.1 +27.3 |
| eDP-2   | 100%       | +10.2 | +1.9  | 0.52   | 0.77     | +9.9 +10.2 +13.6  |

- Adjacent monitors are about 17 to 18 degrees apart. The separation ratio is
  5.2 for DP-1 vs DP-2 and 8.0 for DP-2 vs eDP-2. Nearest-centroid accuracy is
  100% when trained on one round and tested on the other two.
- The head turns far less than the eyes do. The far monitor is only 34.5
  degrees of head yaw from the camera monitor, well below the 45 degree range
  where landmarks were expected to degrade. Face detection never dropped.
- Looking at the camera monitor still gives +10 yaw, not 0, so the user does
  not face the camera squarely. Calibration must stay relative and never assume
  that the camera monitor is at zero.
- With this transformation matrix convention, positive yaw means turning toward
  the user's left. The code should not depend on the sign, since calibration
  learns it.
- The main noise is between visits, not while holding still. Coming back to the
  same monitor lands up to about 4 degrees away (DP-2 round 2, eDP-2 round 3),
  while the jitter within one hold is 0.5 to 1.8 degrees. So calibration should
  record each monitor more than once, and hysteresis margins should come from
  the spread between visits.
- All three monitors sit at the same height, so pitch barely differs between
  them (2 to 6 degrees). Pitch matters mostly for the look-down override, which
  this recording did not cover.

### Free-use classify run (30 s, natural work)

- Face found 100%. Predicted time: DP-2 87%, eDP-2 8%, DP-1 5%.
- 8 raw prediction changes, 4 after a 300ms dwell. The filtered ones were
  single-frame flips at the boundaries between monitors.
- Natural head turns are smaller than calibration turns. Yaw in use spanned
  +17.6 to +35.0 (5th to 95th percentile), while the calibrated centroids span
  +10.2 to +44.8. Looking at eDP-2 naturally gave +16.6 to +18.9, right next to
  the eDP-2/DP-2 midpoint at +18.75. Looking at DP-1 gave +37 to +40 against a
  centroid of +44.8. Calibration with deliberate head turns overshoots.
- A head sweep from DP-1 to eDP-2 spent 0.48s inside the DP-2 zone. A plain
  300ms dwell would have switched to DP-2 on the way past.
- Pitch stayed between -1.8 and +3.1 for the whole run, including any looks at
  the keyboard, so those were mostly eye movements. Head pitch alone may not
  catch looking down.

### Tooling and runtime

- MediaPipe 1.1.0 imports `sounddevice` while building a vision task. That
  opens PortAudio, then ALSA's JACK plugin, then PipeWire, and the process is
  killed with SIGKILL about 0.5s in. Found with gdb catching `tgkill` inside
  `pw_data_loop_stop` called from `jack_client_open`. Blocking the import with
  `sys.modules["sounddevice"] = None` fixes it, because MediaPipe catches the
  ImportError. This only affects the Python prototype. The Rust build does not
  use MediaPipe's runtime.
- MediaPipe depends on `opencv-contrib-python`. Adding `opencv-python` as well
  installs two packages into the same `cv2` folder, and removing one breaks the
  other. Use only the contrib build.
- The prototype runs at about 11 to 12 samples per second on this laptop with
  the Python MediaPipe pipeline, short of the 15 fps target. The Rust build
  should do better because it skips Python and the blendshape model.
- Camera auto exposure takes about 2 seconds to settle after opening (face
  brightness went from 31 to 98 out of 255). The daemon must ignore the first
  couple of seconds of frames after it reopens the camera.

## Verified facts (Phase 0, 2026-10-08)

Checked on the development machine and against current docs, not from memory.

### Hyprland

- Version 0.56.2. Since 0.55 the config is Lua and dispatchers are Lua calls.
  The old string form is gone: `hyprctl dispatch focusmonitor DP-2` now fails
  with a Lua parse error.
- Working forms, tested with no-op calls on this machine:
  - `hyprctl dispatch 'hl.dsp.focus({ monitor = "DP-2" })'`
  - `hyprctl dispatch 'hl.dsp.cursor.move({ x = 1449, y = 804 })'`
  - `hyprctl --batch "dispatch ... ; dispatch ..."` runs both in one request.
- Source: hyprland-wiki `content/configuring/core/dispatchers.md` and
  `advanced-configuration/using-hyprctl.md` at commit ede823b (2026-10-03).
- Command socket: `$XDG_RUNTIME_DIR/hypr/$HYPRLAND_INSTANCE_SIGNATURE/.socket.sock`.
  Hyprland handles it synchronously, so open, write, read, close every time.
- Event socket: `.socket2.sock`, lines of `EVENT>>DATA`. Useful events:
  `focusedmon`/`focusedmonv2`, `monitoradded(v2)`, `monitorremoved(v2)`.
- Cursor warping on monitor focus: `cursor:warp_on_monitor_change` (default -1,
  meaning "follow `cursor:warp_on_change_workspace`"). This machine has
  `warp_on_change_workspace = 1`, so focusing a monitor already warps the cursor
  to the last focused window there. `cursor:no_warps` is false. Read these at
  runtime with `getoption` instead of assuming them.
- `cursorpos` and `cursor.move` use global logical coordinates. Monitor logical
  size is pixels divided by scale, with width and height swapped for rotated
  transforms (1, 3, 5, 7).

### This machine's layout

| Name  | Pixels    | Position | Scale | Transform | Logical size |
|-------|-----------|----------|-------|-----------|--------------|
| DP-1  | 1360x768  | 0,0      | 1.0   | 1 (90°)   | 768x1360     |
| DP-2  | 1920x1080 | 768,0    | 1.0   | 0         | 1920x1080    |
| eDP-2 | 2560x1600 | 2688,0   | 1.6   | 0         | 1600x1000    |

The camera is the laptop's integrated UVC camera (`uvcvideo`,
`/dev/video0`), so it sits on eDP-2, the rightmost display.

### Camera

- Formats: MJPG up to 1920x1080, YUYV at 640x480, 320x240 and others. Every
  mode lists only 30 fps, so 15 fps means capturing at 30 and skipping frames.
- Device access works through a logind ACL. The user is not in the `video`
  group and does not need to be.

### Crates (crates.io, 2026-10-08)

- `ort` 2.0.0-rc.13 (still a release candidate). Targets ONNX Runtime 1.28,
  and ONNX Runtime is documented as forward compatible. Arch ships
  `onnxruntime-cpu` 1.29.0 in `extra`, which is not installed yet.
  `load-dynamic` plus `ORT_DYLIB_PATH` lets us use the system library.
- `nokhwa` 0.10.11. On Linux it is a wrapper over `v4l` 0.14 through
  `nokhwa-bindings-linux`, and it pulls in `image`, `wgpu` and others.
- `v4l` 0.14.0 (last release 2023-05). V4L2 is a stable kernel ABI, so its age
  matters less than it seems.
- `hyprland` (hyprland-rs) 0.3.13 stable, 0.4.0-beta.3, last updated 2025-09.
  That predates the Lua dispatcher syntax.
- `evdev` 0.13.2, `clap` 4.6.7, `toml` 1.1.7, `directories` 6.0.0,
  `crossterm` 0.29.0.

### Models

- Official MediaPipe Face Landmarker bundle:
  `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`
  (3.76 MB, last modified 2023-05-03). It contains a BlazeFace short-range
  detector and the Face Mesh V2 landmark model (478 points). Apache-2.0.
- No official ONNX export exists. A third-party conversion with a published
  script and a parity check against Python MediaPipe:
  `huggingface.co/fernandotonon/QtMeshEditor-facemesh-onnx` (landmarks, input
  `[N,256,256,3]` RGB in [0,1], output 478 x 3) and
  `huggingface.co/fernandotonon/QtMeshEditor-blazeface-onnx` (detector).

### Python prototype tooling

- System Python is 3.14.7. `mediapipe` 1.1.0 (2026-10-06) ships a
  `py3-none-manylinux_2_28_x86_64` wheel. `uv` is available.

### Permissions

- The user is not in the `input` group, so evdev features are unavailable until
  they opt in.

## Decisions

### Hand tracking (2026-10-09)

- The SSD anchor, decode and weighted NMS code is shared by faces and palms,
  generalized over input size and keypoint count.
- Hand crop from a palm, as MediaPipe does it: turned so the wrist to middle
  finger line points up, moved half the palm box toward the fingers, squared
  and grown 2.6x. From landmarks: same turn using the wrist and middle finger
  knuckle, the box of the steadier landmarks (no fingertips) measured along
  the turned axes, moved a tenth toward the fingers, squared and doubled.
- `HandTracker` tracks one hand. Unlike the face tracker, the palm detector
  runs at most once every 4 frames while no hand is tracked (`detect_every`),
  because most of the time there is no hand in view and the detector is the
  expensive part. A hand lost after tracking for a while is looked for again
  on the same frame.
- Gestures come from geometry on the 21 landmarks, not a model. A finger is
  extended when its tip is farther from the wrist than its PIP joint. The
  thumb also needs its tip farther than its IP joint from the little finger's
  base, which catches a thumb folded across the palm. Only x and y are used.
  The open palm (all five extended) is the only gesture so far.
- Hand tracking is not wired into the daemon yet, and the hand models are
  optional: `find_model_dir` still needs only the face models.

### Release (2026-10-08)

- The public repository is `sasukevibes/lookfocus`, created fresh with a
  single initial commit authored with the GitHub noreply address. The
  development repository, with its full history, stays private. This keeps
  personal details out of public history without force-pushing.
- Dev docs are kept in the public repository, scrubbed of personal details:
  this file, the design brief (`docs/SPEC.md`) and the Python prototype.
- The models are attached to a `models-v1` release, separate from app
  releases, so a new app version never changes them. The PKGBUILD downloads
  them from there.
- Version 0.1.0 is the first public release.

### Phases 4 and 5 (2026-10-08)

- The typing lock (override 4 in the spec) is dropped at the user's request:
  lookfocus must never read keystrokes. It reads no input devices at all, so
  there is no evdev dependency and no `input` group requirement. Mouse
  movement is detected by watching the cursor position through Hyprland's
  `cursorpos`, ignoring moves lookfocus made itself.
- Bar widget: left click pauses or resumes, right click opens a dropdown with
  pause, adaptive centroids, recalibrate and start/stop service. Icon and
  tooltip show the state and current screen.
- Keybind: SUPER + ALT + E runs `lookfocus toggle`.
- Adaptive centroids are optional and switchable at run time (bar, CLI,
  config). They learn from mouse use: while the mouse is moving on a monitor,
  the pose is a labelled sample for that monitor.
- Install on the development machine: binary in `~/.local/bin`, systemd user
  service, bar widget and keybind.

### Phase 3

- Hyprland IPC uses the sockets directly, with a 2 s timeout on every request.
  Instance discovery falls back to the newest instance directory.
- Settings (`config.toml`, hand-edited, unknown keys rejected) and calibration
  (`calibration.toml`, written by the tool) are separate files in
  `$XDG_CONFIG_HOME/lookfocus/`, so recalibrating never touches settings.
- Switching rules: the new monitor must be closer than the current one by the
  hysteresis margin, the head must move slower than the settle speed
  (default 25 deg/s), and both must hold for the dwell (default 300 ms). The
  default hysteresis is a quarter of the closest pair's gap, clamped to 1 to 4
  degrees (2.1 for this machine).
- Cursor: `restore` (default) puts the cursor back where it last was on the
  target monitor, or its center the first time, sent in the same batch as the
  focus. `hyprland` leaves Hyprland's own warp.
- The layout fingerprint is name, description, position and logical size per
  monitor. Any change pauses switching with a notification asking for
  `lookfocus recalibrate`, and switching resumes if the layout returns.
- Calibration defaults to 2 rounds plus a look-down step. The camera monitor
  question defaults to a built-in panel (eDP, LVDS, DSI). Without a terminal it
  takes the default, and `--camera-monitor` skips the question.
- A failed focus request leaves the switcher on the old monitor, so the next
  settled frames retry.

### Phase 2

- Models come from the pinned Hugging Face revisions above, fetched by
  `scripts/fetch-models.sh` with SHA-256 checks. Mirroring them to this repo's
  GitHub releases is still planned for packaging.
- Use `v4l` 0.14 directly. It works with this kernel and camera.
- Pose fit is an orthographic weighted Kabsch fit, not a full perspective
  solve. It is simpler and steady, and calibration only needs consistency.
- The tracker detects again on the same frame when tracking is lost, instead
  of waiting a frame as MediaPipe does.
- One thread per model by default. More threads would lower latency but raise
  CPU use, and latency is already well under one frame.
- `rustfmt.toml` sets a 120 column width.

### Phase 1 (from the prototype results)

- Dwell only counts while the head is settled. Use the One Euro filter's
  derivative to hold the dwell timer at zero while angular speed is above a
  threshold, so sweeping past a middle monitor does not select it.
- Calibration prompts should ask for natural glances, not full head turns, and
  should record each monitor at least twice.
- Add adaptive centroids as an option: when the mouse is used on a monitor,
  that is a labeled pose sample, and the centroid can drift slowly toward it.
  This corrects the gap between calibration poses and natural poses.
- Calibration gets a "look at your keyboard" step so the look-down threshold is
  measured. If it is not separable from the monitors, the look-down override
  falls back to face-lost and typing detection.

### Phase 0 (approved 2026-10-08)

- Name: `lookfocus`. It is free on crates.io and the AUR.
- Video calls: v1 releases the camera whenever lookfocus is paused, and the
  README tells users to run `lookfocus toggle` before a call. Automatic call
  detection is future work.
- Capture with `v4l` directly instead of `nokhwa`. nokhwa is a wrapper over the
  same crate on Linux and adds a large dependency tree. Use YUYV 640x480, which
  needs no JPEG decode.
- Talk to the Hyprland sockets directly instead of using hyprland-rs, because
  hyprland-rs builds the old dispatcher strings. Detect the Hyprland version
  and use the legacy `focusmonitor`/`movecursor` syntax below 0.55.
- Send focus and cursor move in one batch request so any warp from focus is
  replaced before the user sees it. Phase 3 will verify this visually.
- Head pose from a rigid fit (Kabsch/Procrustes) of the MediaPipe canonical face
  model to the 478 landmarks. Calibration is relative, so consistency matters
  more than absolute accuracy.
- Architecture: threads connected by channels, with one internal `Event` enum
  that the future event socket can rebroadcast.

## Open questions

- Can the landmark model stay reliable at the yaw needed to look at the far left
  monitor from a right-side camera? Phase 1 measures this.
- Model hosting is settled: the `models-v1` release of the public repository
  carries both files, with the pinned Hugging Face revisions as a fallback in
  `scripts/fetch-models.sh`.
- AUR submission needs the maintainer's AUR account. The PKGBUILD is ready
  apart from `updpkgsums` for the source tarball.
- A demo GIF for the README still needs recording (the README says how).
- Is about 20% of one core acceptable all day? Further savings would come from
  a lower capture resolution or skipping the landmark model on frames where
  the face has not moved.
