# lookfocus

Focus the Hyprland monitor you are facing. lookfocus watches your head through
a webcam, works out which monitor you are looking at, and after a short pause
moves Hyprland's focus and your cursor there. No more dragging the mouse across
three screens.

Built for [Omarchy](https://omarchy.org/) (Arch Linux with Hyprland), and works
on any Hyprland setup.

- Any layout: two monitors, a row of three, a vertical stack or a grid.
- Any camera position. The camera does not need to be in the middle.
- Mouse movement always wins, and a keybind or bar button pauses it.
- Runs on the CPU, with no GPU needed. Frames never leave memory.
- Optional hand gestures: hold up an open palm to run a command, such as
  toggling dictation.

## Privacy

lookfocus processes camera frames in memory only. It never stores a frame,
never sends anything over the network, and never reads your keyboard. The only
things it writes to disk are head angles: your calibration
(`~/.config/lookfocus/calibration.toml`) and, if you turn on adaptive
centroids, what it learned (`~/.local/state/lookfocus/state.json`), and the
adaptive and gestures switches. The camera is released whenever lookfocus is
paused or nobody is in front of it, except that it stays open while hand
gestures are on, since they need it.

## Demo

<!--
To add the recording: capture the screens with `omarchy capture`, convert it
to docs/demo.gif, and uncomment the next line.
![lookfocus switching focus between three monitors](docs/demo.gif)
-->

The setup: three monitors in a row with the camera on the rightmost one. Turning toward a screen moves focus and the cursor there about
a third of a second after your head settles. A quick sweep from the left
screen to the right one goes straight past the middle without selecting it.
Grabbing the mouse takes over immediately.

`lookfocus debug` shows the same thing as text:

```
yaw  +40.6  pitch  +0.4  speed    3°/s  zone DP-1   margin 11.8  stay
yaw  +33.9  pitch  +0.2  speed   61°/s  zone DP-2   margin  4.0  pending DP-2 0 ms
yaw  +23.1  pitch -3.2  speed   38°/s  zone eDP-2  margin  5.9  pending eDP-2 0 ms
yaw  +22.4  pitch -3.5  speed    6°/s  zone eDP-2  margin  6.4  pending eDP-2 264 ms
yaw  +22.3  pitch -3.6  speed    2°/s  zone eDP-2  margin  6.5  SWITCH -> eDP-2
```

## How it works

1. A face detector and MediaPipe's 478-point face mesh model run on each
   camera frame through ONNX Runtime.
2. A rigid fit of those points to MediaPipe's canonical face gives your head's
   yaw (left and right) and pitch (up and down).
3. A One Euro filter smooths the angles without adding much lag.
4. Calibration recorded where your head points for each monitor. The nearest
   recorded pose wins.
5. A new monitor takes over only when it beats the current one by a margin,
   your head has settled, and both have held for 300 ms.

Eye direction is not used in this version, only head pose.

## Requirements

- Hyprland 0.55 or newer (0.54 and older work too, through the legacy
  dispatcher syntax)
- A webcam that offers YUYV, which almost all UVC webcams do
- ONNX Runtime: `sudo pacman -S onnxruntime-cpu`
- Two or more monitors

## Install

### From the AUR (once published)

```sh
yay -S lookfocus
systemctl --user enable lookfocus
```

The package installs the binary, the models, the systemd user service and the
Omarchy bar widget.

### From source

```sh
git clone https://github.com/sasukevibes/lookfocus
cd lookfocus
scripts/install-local.sh
```

This builds lookfocus, puts it in `~/.local/bin`, downloads the models
(checked against pinned SHA-256 sums), and installs and enables the systemd
user service. `scripts/uninstall-local.sh` removes all of that again.

## Set up

### 1. Calibrate

```sh
lookfocus calibrate
```

lookfocus asks which monitor your camera sits on, then shows a notification
for each screen in turn: "Look at the LEFT screen", "Look at the CENTER
screen", and so on. Look at each one the way you normally would while
working, without exaggerating the head turn. It visits every screen twice and
then asks you to look down at your keyboard. The whole thing takes under a
minute, and it ends with a short report.

The report may add notes, for example when two screens are close together
from where your camera sits. Each note says what to try.

Run `lookfocus recalibrate` after moving the camera or a monitor. If
lookfocus is already running, calibration borrows the camera and hands it
back with the new calibration.

### 2. Start it

```sh
systemctl --user start lookfocus
```

The service starts with your graphical session from then on. To try it in a
terminal first, run `lookfocus run` and stop it with Ctrl+C.

### 3. Add the bar button (Omarchy)

The install puts the widget in `~/.config/omarchy/plugins/sasukevibes.lookfocus`
(the AUR package puts it in `/usr/share/lookfocus/omarchy/`, so copy that folder
there). Then add `"sasukevibes.lookfocus"` to a bar section in
`~/.config/omarchy/shell.json`, or use the shell's widget picker.

- Left click pauses or resumes. The icon shows the state, and the name of the
  focused monitor while tracking.
- Right click opens a menu with tracking, adaptive centroids, gestures,
  recalibrate and start/stop service.

### 4. Add a keybind

On Omarchy, in `~/.config/hypr/bindings.lua`:

```lua
o.bind("SUPER + ALT + E", "Toggle lookfocus", "lookfocus toggle")
```

On plain Hyprland 0.55 and newer:

```lua
hl.bind("SUPER + ALT + E", hl.dsp.exec_cmd("lookfocus toggle"))
```

On Hyprland 0.54 and older (`hyprland.conf`):

```
bind = SUPER ALT, E, exec, lookfocus toggle
```

Check the key is free first. On Omarchy, `omarchy menu keybindings --print`
lists what is taken.

## Overrides

| Situation | What lookfocus does |
|---|---|
| You move the mouse | Holds switching for 2 seconds after the last movement |
| You press the keybind or click the bar button | Pauses and releases the camera, or resumes. With gestures on, the camera stays open for them and only focus switching stops |
| No face in view | Holds the current monitor. After 30 seconds it releases the camera (gestures included), and comes back when you move the mouse |
| You look down at the keyboard or a phone | Holds the current monitor, if calibration could tell looking down apart from your screens |
| The monitor layout changes | Pauses with a notification asking you to recalibrate, and resumes on its own if the layout comes back |
| Another app needs the camera | Pause lookfocus first (for a video call, say). If the camera is busy when lookfocus starts, it retries every 5 seconds |

lookfocus does not watch your typing. It never reads input devices, so it
needs no special permissions.

## Adaptive centroids

People often turn their heads further when asked to look at a screen than
when they glance at it while working. With adaptive centroids on, lookfocus
learns from mouse use: while you move the mouse on a screen, you are almost
certainly looking at it, so that screen's calibrated pose moves a small step
toward your current one.

Learning is limited. A pose never drifts more than 6 degrees from its
calibrated value, two screens never get closer than 60% of their calibrated
distance, and samples far from the calibrated pose are ignored.

```sh
lookfocus adaptive on      # or off, toggle
lookfocus adaptive reset   # forget what was learned
```

It is off by default. The bar menu has a switch for it, which also shows how
far each screen has moved.

## Hand gestures

lookfocus can run a command when you hold up a hand. The only gesture so far
is the open palm (all five fingers spread, facing the camera), and by default
it runs `voxtype record toggle`, so one palm starts voxtype dictation and the
next one stops it.

Gestures are off by default. Switch them on with the bar menu, or:

```sh
lookfocus gestures on      # or off, toggle
```

The switch is kept between runs. To have them on from the start for good, set
`enabled = true` under `[gestures]` in `config.toml`. Gestures need the hand
models (`hand_detector.onnx` and `hand_landmarks.onnx`), which
`scripts/fetch-models.sh` downloads next to the face models. Without them
lookfocus logs "hand tracking is off" and focus tracking carries on as usual.

How it works:

- Gestures and focus tracking are separate switches. The camera is open when
  either one wants it. Pausing focus tracking, or a monitor layout change,
  stops focus switching but not gestures. Only turning both off releases the
  camera.
- A gesture counts after you hold it for 400 ms with your face in view. A hand
  waved across the picture, or raised while you are not there, does nothing.
- It fires once. To fire it again, lower your hand for half a second, then
  raise it. Two firings are at least 1.5 seconds apart.
- The hand models run only while gestures are on. While a hand is in view,
  lookfocus samples at the full rate so it does not miss the gesture.
- If nobody is in front of the camera for 30 seconds, the camera is released
  for gestures too, and comes back when you move the mouse.

Change the timing, or the command for a gesture, in `config.toml`:

```toml
[gestures]
enabled = true
hold_ms = 400
cooldown_ms = 1500
require_face = true

[gestures.actions]
open_palm = "voxtype record toggle"
```

Commands run through `sh -c`, so pipes and arguments work. lookfocus does not
wait for them, and logs one that cannot start or exits with an error (see
`journalctl --user -u lookfocus`). The service's PATH can be shorter than your
terminal's (it may lack `~/.local/bin`, for one), so give the full path to a
command that lives somewhere unusual.
Setting a gesture's command to `""` turns its action off. The `gesture` event
still goes out for other tools (see [docs/events.md](docs/events.md)).

`lookfocus debug` shows the hand and the gesture it sees, whatever the gestures
setting is, so you can check that the models work.

## Commands

| Command | What it does |
|---|---|
| `lookfocus calibrate` | Record where you look for each monitor |
| `lookfocus recalibrate` | The same, defaulting to your previous camera answer |
| `lookfocus run` | Run the daemon in the foreground (the service runs this) |
| `lookfocus toggle` | Pause or resume. Starts the service if it is stopped |
| `lookfocus pause`, `lookfocus resume` | Explicit versions of toggle |
| `lookfocus status [--json]` | Show the state, focused monitor and camera |
| `lookfocus adaptive on\|off\|toggle\|reset` | Control adaptive centroids |
| `lookfocus gestures on\|off\|toggle` | Switch hand gestures |
| `lookfocus debug [--json]` | Live pose, hand, zone and what the daemon would do |
| `lookfocus watch` | Stream events as JSON lines (experimental, see below) |

## Configuration

Settings go in `~/.config/lookfocus/config.toml`. Every setting is optional,
and a misspelled one is reported as an error instead of being ignored. See
[docs/configuration.md](docs/configuration.md) for all of them. The common
ones:

```toml
[switching]
dwell_ms = 300          # how long to hold a new pose before switching
hysteresis_deg = 2.5    # margin a new monitor needs (default: from calibration)
cursor = "restore"      # or "hyprland" to keep Hyprland's own cursor warp

[overrides]
mouse_hold_ms = 2000

[adaptive]
enabled = false

[gestures]
enabled = false         # or switch at run time: lookfocus gestures on
hold_ms = 400

[gestures.actions]
open_palm = "voxtype record toggle"
```

## Event stream (experimental)

`lookfocus watch` prints what the daemon does as JSON lines: the monitor you
are facing, switches, pauses and so on. Other tools can use it to react to
where you are looking. See [docs/events.md](docs/events.md). The format may
change before 1.0.

## Troubleshooting

**"model files not found"**: run `scripts/fetch-models.sh`, or point
`models_dir` in config.toml at a folder with `face_detector.onnx` and
`face_landmarks.onnx`.

**"hand tracking is off"**: gestures are on but the hand models are missing.
Run `scripts/fetch-models.sh`, or put `hand_detector.onnx` and
`hand_landmarks.onnx` next to the face models.

**A gesture does nothing**: run `lookfocus debug` and raise your hand. It
should show `hand open_palm`. If it does, check the log for the command's own
error, and that your face is in view (`require_face`).

**"could not load ONNX Runtime"**: `sudo pacman -S onnxruntime-cpu`, or set
`ORT_DYLIB_PATH` to your `libonnxruntime.so`.

**"camera /dev/video0 is busy"**: another app is using the camera. Close it,
or pause lookfocus before starting a call. lookfocus retries on its own.

**Two monitors keep swapping**: they are close together from where the camera
sits. Try one of these:

- recalibrate, turning your head a little more toward each screen
- raise `hysteresis_deg` or `dwell_ms`
- turn on adaptive centroids
- move the camera nearer the middle of your screens

**It switches while I am using the mouse on another screen**: it should not,
because mouse movement holds switching for two seconds. If your mouse moves
less often than that while you work, raise `mouse_hold_ms`.

**The service does not start**: run `journalctl --user -u lookfocus -e`.
Exit code 78 means something a restart cannot fix, usually a missing
calibration.

**Hyprland was updated and focus stopped working**: run `lookfocus debug` and
`lookfocus status`, and open an issue with the output of `hyprctl version`.

Run any command with `RUST_LOG=debug` for more detail.

## Building and testing

```sh
cargo test        # no camera, models or Hyprland needed
cargo clippy --all-targets
```

The tests use scripted poses, a fake Hyprland and fake time, so they cover
the whole daemon without hardware. `prototype/` holds the Python script used
to check that head pose can tell monitors apart before the Rust build, and
`NOTES.md` records the decisions and measurements along the way.

## Credits and license

The face detection and face mesh models are Google's
[MediaPipe](https://ai.google.dev/edge/mediapipe) models (Apache-2.0),
converted to ONNX. The canonical face geometry in `src/pose/canonical.rs`
comes from the same model bundle. The palm detection and hand landmark models
are MediaPipe's too, in the ONNX conversions published by
[OpenCV Zoo](https://github.com/opencv/opencv_zoo) (Apache-2.0).

lookfocus is licensed under either of [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE), at your option.
