# Configuration reference

lookfocus reads `~/.config/lookfocus/config.toml` (or
`$XDG_CONFIG_HOME/lookfocus/config.toml`). The file is optional, and so is
every setting in it. Unknown or misspelled keys are an error, so a typo never
goes unnoticed.

Restart the service after editing: `systemctl --user restart lookfocus`.

Two other files are written by lookfocus itself. Do not edit them by hand:

| File | Written by | Contents |
|---|---|---|
| `~/.config/lookfocus/calibration.toml` | `lookfocus calibrate` | Head pose per monitor, the monitor layout, look-down threshold |
| `~/.local/state/lookfocus/state.json` | the daemon | The adaptive centroids and gestures switches, and what adaptive learning learned |

## Full example with defaults

```toml
# Directory holding face_detector.onnx and face_landmarks.onnx (and, for
# hand tracking, hand_detector.onnx and hand_landmarks.onnx). Unset means
# the usual places: $LOOKFOCUS_MODEL_DIR, ~/.local/share/lookfocus/models,
# /usr/share/lookfocus/models.
# models_dir = "/path/to/models"

[camera]
device = "/dev/video0"
fps = 15.0        # samples per second while active
threads = 1       # CPU threads per model

[switching]
dwell_ms = 300    # how long a new pose must hold before switching
# hysteresis_deg = 2.0   # unset: a quarter of the gap between the two
                         # closest monitors, kept between 1 and 4 degrees
settle_speed = 25.0      # degrees per second; the dwell only counts while
                         # the head moves slower than this
cursor = "restore"       # "restore": back where it last was on that monitor
                         # "hyprland": wherever Hyprland's focus puts it

[overrides]
mouse_hold_ms = 2000     # mouse movement holds switching this long
look_down = true         # hold while looking down, if calibration measured it

[adaptive]
enabled = false          # the bar button and `lookfocus adaptive` override this
rate = 0.03              # fraction of the way to move per sample
max_drift_deg = 6.0      # furthest a monitor's pose may move from calibration

[power]
idle_fps = 6.0           # samples per second while the head is still
away_after_s = 30        # release the camera after this long without a face
probe_every_s = 20       # while away, look for a face this often

[gestures]
enabled = false          # the bar menu and `lookfocus gestures` override this
hold_ms = 400            # how long a gesture must be held before it counts
cooldown_ms = 1500       # the least time between two firings
require_face = true      # count a gesture only while your face is in view

[gestures.actions]       # gesture name = shell command, run with sh -c
open_palm = "voxtype record toggle"

[filter]                 # One Euro filter on yaw and pitch
min_cutoff = 1.0         # Hz. Lower is smoother when still, but laggier
beta = 0.05              # higher follows fast moves more closely
d_cutoff = 1.0           # Hz, for the speed estimate
```

## Gestures

`[gestures]` is described in the README. The details that matter for tuning:

- `enabled` is only the starting value. Once you use the bar menu or
  `lookfocus gestures on|off|toggle`, the choice is kept in `state.json` and
  wins over `enabled`. Remove the `gestures` entry from `state.json` to go back to
  the config value.
- A gesture fires after `hold_ms` of being held without a break. Once it has
  fired it can fire again only after it has been out of the picture for 500 ms
  (this is fixed), and at least `cooldown_ms` after the last firing. A gesture
  still held when the cooldown ends fires then.
- With `require_face = true`, the hold only counts while a face is in view. If
  your hand covers your face, the hold restarts. Set it to `false` to let
  gestures work with no face, for example when you sit back from the camera.
- `[gestures.actions]` maps a gesture name to a command. The only gesture name
  so far is `open_palm`. A name that does not exist is an error. Setting a
  command to an empty string turns the action off for that gesture. If you
  write the table, it replaces the default, so list every action you want.
- Commands run with `sh -c` as you, from the daemon's environment. They are not
  waited for. One that cannot start, or exits with a nonzero status, is logged.
- The camera stays open while gestures are on, including while focus tracking
  is paused. Only `away_after_s` releases it.

## Tuning tips

- **Switching feels slow**: lower `dwell_ms` (200 is still steady for most
  people) or raise `settle_speed` a little.
- **Two monitors swap back and forth**: raise `hysteresis_deg` by half a
  degree at a time, or raise `dwell_ms`. Recalibrating with slightly clearer
  head turns also helps.
- **CPU use matters more than speed**: lower `camera.fps` to 10 and
  `power.idle_fps` to 4. A turn is then noticed up to a quarter of a second
  later.
- **The cursor should stay where Hyprland puts it**: set `cursor = "hyprland"`.
- **A gesture fires by accident**: raise `hold_ms` (600 is still quick) or
  `cooldown_ms`.
- **A gesture is slow to fire**: lower `hold_ms`, and check that `fps` is not
  very low, since the hand is found on one frame in four while no hand is in
  view.

## Environment variables

| Variable | Effect |
|---|---|
| `RUST_LOG` | Log level: `error`, `warn`, `info` (default), `debug` |
| `ORT_DYLIB_PATH` | Path to `libonnxruntime.so` if it is not in `/usr/lib` |
| `LOOKFOCUS_MODEL_DIR` | Extra place to look for the models |
| `HYPRLAND_INSTANCE_SIGNATURE` | Which Hyprland to talk to. Without it, the newest running instance is used |
