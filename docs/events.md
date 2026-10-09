# Event stream (experimental)

lookfocus knows which monitor you are facing. Other tools can use that, for
example a bar widget that highlights the screen you face, an app that moves
its notifications to where you are looking, or a script that dims the screens
you are not.

Inside the daemon, everything notable is already an event. `lookfocus watch`
streams them as JSON lines. The format is experimental and may change before
1.0.

## Reading the stream

From a shell:

```sh
lookfocus watch | jq -c 'select(.event == "zone")'
```

From any language, connect to the control socket, send `watch`, and read
lines:

```
$XDG_RUNTIME_DIR/lookfocus/control.sock
```

```python
import json, os, socket

path = os.path.join(os.environ["XDG_RUNTIME_DIR"], "lookfocus/control.sock")
s = socket.socket(socket.AF_UNIX)
s.connect(path)
s.sendall(b"watch\n")
for line in s.makefile():
    event = json.loads(line)
    if event["event"] == "zone":
        print("now facing", event["monitor"])
```

The stream ends when the daemon stops. Reconnect when it is back.

## Events

Each line is one JSON object with an `event` field.

| `event` | Fields | Meaning |
|---|---|---|
| `started` | `monitors` | The daemon started with these calibrated monitors |
| `zone` | `monitor` | The monitor your head points at changed. This can change without a switch, for a quick glance |
| `switched` | `from`, `to` | lookfocus moved focus |
| `focus_changed` | `monitor` | Focus moved for another reason (mouse, keybind) |
| `face_found`, `face_lost` | | A face came into view or left it |
| `away`, `back` | | No face for a while (camera released), and back again |
| `paused` | `reason` | Switching stopped. `reason` is `"paused"` when you paused it, or explains why (a layout change, for example) |
| `resumed` | | Switching started again |
| `camera_unavailable` | `reason` | The camera could not be opened |
| `adaptive_changed` | `enabled` | Adaptive centroids were switched on or off |
| `gestures_changed` | `enabled` | Hand gestures were switched on or off |
| `gesture` | `gesture`, `command` | A hand gesture was held long enough to count. `gesture` is its name (`open_palm`). `command` is the shell command lookfocus ran for it, or `null` if none is set |

Example:

```json
{"event":"zone","monitor":"eDP-2"}
{"event":"switched","from":"DP-2","to":"eDP-2"}
{"event":"paused","reason":"paused"}
{"event":"gesture","gesture":"open_palm","command":"voxtype record toggle"}
```

The `gesture` event is sent when the gesture fires, so a tool that only wants to
react to gestures can use it and leave `command` empty in `config.toml`:

```sh
lookfocus watch | jq -c 'select(.event == "gesture")'
```

## Other control commands

The same socket takes one command per connection and answers with one JSON
line. The CLI uses these:

| Command | Reply |
|---|---|
| `status` | Current status (see `lookfocus status --json`) |
| `pause`, `resume`, `toggle` | Status after the change |
| `adaptive on`, `adaptive off`, `adaptive toggle`, `adaptive reset` | Status after the change |
| `gestures on`, `gestures off`, `gestures toggle` | Status after the change |
| `reload` | Status after reading `calibration.toml` again |

## Design notes

- The stream carries decisions, never images or raw landmarks.
- Events are pushed as they happen, with no polling. A slow reader cannot
  slow the daemon down: each subscriber has its own queue, and a subscriber
  that disconnects is dropped.
- A future version may add a pose event at a low rate for tools that want
  continuous angles. It will be opt-in.
