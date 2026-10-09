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
| `~/.local/state/lookfocus/state.json` | the daemon | The adaptive centroids switch and what it learned |

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

[filter]                 # One Euro filter on yaw and pitch
min_cutoff = 1.0         # Hz. Lower is smoother when still, but laggier
beta = 0.05              # higher follows fast moves more closely
d_cutoff = 1.0           # Hz, for the speed estimate
```

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

## Environment variables

| Variable | Effect |
|---|---|
| `RUST_LOG` | Log level: `error`, `warn`, `info` (default), `debug` |
| `ORT_DYLIB_PATH` | Path to `libonnxruntime.so` if it is not in `/usr/lib` |
| `LOOKFOCUS_MODEL_DIR` | Extra place to look for the models |
| `HYPRLAND_INSTANCE_SIGNATURE` | Which Hyprland to talk to. Without it, the newest running instance is used |
