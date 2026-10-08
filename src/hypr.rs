//! Talking to Hyprland over its IPC sockets.
//!
//! Two sockets live in `$XDG_RUNTIME_DIR/hypr/<instance>/`:
//!
//! - `.socket.sock` takes one request per connection, like `j/monitors`, and
//!   replies with text. Hyprland handles it synchronously, so we open, write,
//!   read and close every time, with timeouts so a stuck compositor cannot
//!   stall us.
//! - `.socket2.sock` streams events as `name>>data` lines.
//!
//! Since Hyprland 0.55 dispatchers are Lua calls such as
//! `hl.dsp.focus({ monitor = "DP-2" })`. Older versions use the string form
//! `focusmonitor DP-2`. We read the version once and build the right form.
//!
//! Focusing a monitor makes Hyprland move the cursor to that monitor's last
//! window (or its middle). We send the focus and our own cursor move as one
//! `[[BATCH]]` request. Hyprland runs both before drawing the next frame, so
//! its intermediate cursor position is never seen.

use std::env;
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use thiserror::Error;

const TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Debug, Error)]
pub enum HyprError {
    #[error("Hyprland is not running (no instance found in {0})")]
    NotRunning(PathBuf),
    #[error("XDG_RUNTIME_DIR is not set")]
    NoRuntimeDir,
    #[error("Hyprland socket {path}: {source}")]
    Socket { path: PathBuf, source: std::io::Error },
    #[error("Hyprland rejected {request:?}: {reply}")]
    Rejected { request: String, reply: String },
    #[error("could not parse Hyprland's reply to {request:?}: {detail}")]
    Parse { request: String, detail: String },
}

/// A monitor as lookfocus needs it. Sizes are logical (after scale and
/// rotation), which is the coordinate space cursor positions use.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Monitor {
    pub name: String,
    pub description: String,
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
    pub focused: bool,
}

impl Monitor {
    pub fn contains(&self, x: i32, y: i32) -> bool {
        x >= self.x && y >= self.y && x < self.x + self.width && y < self.y + self.height
    }

    pub fn center(&self) -> (i32, i32) {
        (self.x + self.width / 2, self.y + self.height / 2)
    }

    /// Moves a point inside this monitor, keeping it a few pixels from the
    /// edge so the cursor does not slip onto a neighbour.
    pub fn clamp(&self, x: i32, y: i32) -> (i32, i32) {
        let m = 2;
        (x.clamp(self.x + m, self.x + self.width - 1 - m), y.clamp(self.y + m, self.y + self.height - 1 - m))
    }
}

/// Events from `.socket2.sock` that lookfocus reacts to.
#[derive(Clone, Debug, PartialEq)]
pub enum HyprEvent {
    /// The focused monitor changed, by us or by anything else.
    FocusedMonitor(String),
    MonitorAdded(String),
    MonitorRemoved(String),
}

/// What the daemon needs from a compositor. The real one is `Hyprland`; tests
/// use a fake.
pub trait Compositor {
    fn monitors(&self) -> Result<Vec<Monitor>, HyprError>;
    fn cursor_pos(&self) -> Result<(i32, i32), HyprError>;
    /// Focuses `monitor` and, if given, moves the cursor to `cursor`.
    fn focus(&self, monitor: &str, cursor: Option<(i32, i32)>) -> Result<(), HyprError>;
}

pub struct Hyprland {
    dir: PathBuf,
    lua: bool,
}

impl Hyprland {
    /// Finds the running instance and checks which dispatcher syntax it uses.
    pub fn connect() -> Result<Self, HyprError> {
        let dir = instance_dir()?;
        let mut h = Self { dir, lua: true };
        let reply = h.request("j/version")?;
        h.lua = match parse_version(&reply) {
            Some(v) => uses_lua(v),
            None => {
                log::warn!("could not read the Hyprland version, assuming 0.55 or newer");
                true
            }
        };
        Ok(h)
    }

    pub fn uses_lua(&self) -> bool {
        self.lua
    }

    fn request(&self, req: &str) -> Result<String, HyprError> {
        let path = self.dir.join(".socket.sock");
        let err = |source| HyprError::Socket { path: path.clone(), source };
        let mut stream = UnixStream::connect(&path).map_err(err)?;
        stream.set_read_timeout(Some(TIMEOUT)).map_err(err)?;
        stream.set_write_timeout(Some(TIMEOUT)).map_err(err)?;
        stream.write_all(req.as_bytes()).map_err(err)?;
        let mut reply = String::new();
        stream.read_to_string(&mut reply).map_err(err)?;
        Ok(reply)
    }

    fn json<T: serde::de::DeserializeOwned>(&self, req: &str) -> Result<T, HyprError> {
        let reply = self.request(req)?;
        serde_json::from_str(&reply)
            .map_err(|e| HyprError::Parse { request: req.into(), detail: format!("{e}: {reply}") })
    }

    /// Reads an option, for example `cursor:no_warps`. Returns `None` if this
    /// Hyprland version does not have it.
    pub fn option_int(&self, name: &str) -> Result<Option<i64>, HyprError> {
        let reply = self.request(&format!("j/getoption {name}"))?;
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&reply) else { return Ok(None) };
        Ok(v.get("int").and_then(|i| i.as_i64()).or_else(|| v.get("bool").and_then(|b| b.as_bool()).map(i64::from)))
    }

    /// Opens the event stream. Blocks on read, so run it on its own thread.
    pub fn events(&self) -> Result<EventStream, HyprError> {
        let path = self.dir.join(".socket2.sock");
        let stream = UnixStream::connect(&path).map_err(|source| HyprError::Socket { path, source })?;
        Ok(EventStream { reader: BufReader::new(stream) })
    }
}

impl Compositor for Hyprland {
    fn monitors(&self) -> Result<Vec<Monitor>, HyprError> {
        let reply = self.request("j/monitors")?;
        parse_monitors(&reply).map_err(|detail| HyprError::Parse { request: "j/monitors".into(), detail })
    }

    fn cursor_pos(&self) -> Result<(i32, i32), HyprError> {
        #[derive(Deserialize)]
        struct Pos {
            x: f64,
            y: f64,
        }
        let p: Pos = self.json("j/cursorpos")?;
        Ok((p.x.round() as i32, p.y.round() as i32))
    }

    fn focus(&self, monitor: &str, cursor: Option<(i32, i32)>) -> Result<(), HyprError> {
        let req = focus_request(self.lua, monitor, cursor);
        let reply = self.request(&req)?;
        if reply_ok(&reply) { Ok(()) } else { Err(HyprError::Rejected { request: req, reply: reply.trim().into() }) }
    }
}

pub struct EventStream {
    reader: BufReader<UnixStream>,
}

impl Iterator for EventStream {
    type Item = HyprEvent;

    /// Returns the next event lookfocus cares about, skipping the rest. Ends
    /// when the socket closes.
    fn next(&mut self) -> Option<HyprEvent> {
        let mut line = String::new();
        loop {
            line.clear();
            match self.reader.read_line(&mut line) {
                Ok(0) | Err(_) => return None,
                Ok(_) => {
                    if let Some(e) = parse_event(line.trim_end()) {
                        return Some(e);
                    }
                }
            }
        }
    }
}

/// The directory of the running Hyprland instance. Uses
/// `HYPRLAND_INSTANCE_SIGNATURE` when set. Otherwise picks the newest
/// instance, because systemd services and some shells do not inherit it.
pub fn instance_dir() -> Result<PathBuf, HyprError> {
    let runtime = env::var_os("XDG_RUNTIME_DIR").ok_or(HyprError::NoRuntimeDir)?;
    let base = Path::new(&runtime).join("hypr");
    if let Some(sig) = env::var_os("HYPRLAND_INSTANCE_SIGNATURE") {
        let dir = base.join(sig);
        if dir.join(".socket.sock").exists() {
            return Ok(dir);
        }
    }
    newest_instance(&base).ok_or(HyprError::NotRunning(base))
}

fn newest_instance(base: &Path) -> Option<PathBuf> {
    std::fs::read_dir(base)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.join(".socket.sock").exists())
        .filter_map(|p| Some((p.metadata().ok()?.modified().ok()?, p)))
        .max()
        .map(|(_, p)| p)
}

/// Parses `j/version` into (major, minor, patch).
pub fn parse_version(reply: &str) -> Option<(u32, u32, u32)> {
    let v: serde_json::Value = serde_json::from_str(reply).ok()?;
    let s = v.get("version").or_else(|| v.get("tag"))?.as_str()?.trim_start_matches('v');
    let mut parts = s.split('.').map(|p| p.split(|c: char| !c.is_ascii_digit()).next().unwrap_or("").parse().ok());
    Some((parts.next()??, parts.next()??, parts.next().flatten().unwrap_or(0)))
}

/// Lua dispatchers arrived in Hyprland 0.55.
pub fn uses_lua((major, minor, _): (u32, u32, u32)) -> bool {
    major > 0 || minor >= 55
}

/// Parses `j/monitors`, converting sizes to logical pixels.
pub fn parse_monitors(reply: &str) -> Result<Vec<Monitor>, String> {
    #[derive(Deserialize)]
    struct Raw {
        name: String,
        #[serde(default)]
        description: String,
        x: i32,
        y: i32,
        width: i32,
        height: i32,
        #[serde(default = "one")]
        scale: f64,
        #[serde(default)]
        transform: i32,
        #[serde(default)]
        focused: bool,
        #[serde(default)]
        disabled: bool,
    }
    fn one() -> f64 {
        1.0
    }
    let raw: Vec<Raw> = serde_json::from_str(reply).map_err(|e| format!("{e}: {reply}"))?;
    Ok(raw
        .into_iter()
        .filter(|m| !m.disabled)
        .map(|m| {
            let scale = if m.scale > 0.0 { m.scale } else { 1.0 };
            let (mut w, mut h) = ((m.width as f64 / scale).round() as i32, (m.height as f64 / scale).round() as i32);
            // Transforms 1, 3, 5 and 7 rotate by 90 or 270 degrees.
            if m.transform % 2 == 1 {
                std::mem::swap(&mut w, &mut h);
            }
            Monitor {
                name: m.name,
                description: m.description,
                x: m.x,
                y: m.y,
                width: w,
                height: h,
                focused: m.focused,
            }
        })
        .collect())
}

/// Builds the request that focuses a monitor and optionally moves the cursor.
pub fn focus_request(lua: bool, monitor: &str, cursor: Option<(i32, i32)>) -> String {
    let focus = if lua {
        format!("dispatch hl.dsp.focus({{ monitor = \"{}\" }})", lua_escape(monitor))
    } else {
        format!("dispatch focusmonitor {monitor}")
    };
    match cursor {
        None => focus,
        Some((x, y)) => {
            let mv = if lua {
                format!("dispatch hl.dsp.cursor.move({{ x = {x}, y = {y} }})")
            } else {
                format!("dispatch movecursor {x} {y}")
            };
            format!("[[BATCH]]{focus} ; {mv}")
        }
    }
}

fn lua_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// True if every reply in a (possibly batched) response is "ok".
pub fn reply_ok(reply: &str) -> bool {
    let mut any = false;
    for line in reply.lines().map(str::trim).filter(|l| !l.is_empty()) {
        if line != "ok" {
            return false;
        }
        any = true;
    }
    any
}

/// Parses one `.socket2.sock` line.
pub fn parse_event(line: &str) -> Option<HyprEvent> {
    let (name, data) = line.split_once(">>")?;
    let first = data.split(',').next().unwrap_or("").to_string();
    match name {
        "focusedmon" => Some(HyprEvent::FocusedMonitor(first)),
        "monitoradded" => Some(HyprEvent::MonitorAdded(first)),
        "monitorremoved" => Some(HyprEvent::MonitorRemoved(first)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shaped like this machine's `hyprctl monitors -j`, trimmed to the
    /// fields we read: a rotated monitor, a plain one and a scaled laptop.
    const MONITORS: &str = r#"[
        {"id":0,"name":"eDP-2","description":"Laptop panel","width":2560,"height":1600,"x":2688,"y":0,"scale":1.6,"transform":0,"focused":false,"disabled":false},
        {"id":1,"name":"DP-1","description":"Side monitor","width":1360,"height":768,"x":0,"y":0,"scale":1.0,"transform":1,"focused":true,"disabled":false},
        {"id":2,"name":"DP-2","description":"Main monitor","width":1920,"height":1080,"x":768,"y":0,"scale":1.0,"transform":0,"focused":false,"disabled":false},
        {"id":3,"name":"HDMI-A-1","description":"Off","width":1920,"height":1080,"x":0,"y":0,"scale":1.0,"transform":0,"focused":false,"disabled":true}
    ]"#;

    #[test]
    fn parses_monitors_into_logical_sizes() {
        let m = parse_monitors(MONITORS).unwrap();
        assert_eq!(m.len(), 3, "disabled monitors are skipped");
        let by = |n: &str| m.iter().find(|x| x.name == n).unwrap().clone();
        assert_eq!((by("eDP-2").width, by("eDP-2").height), (1600, 1000));
        assert_eq!((by("DP-1").width, by("DP-1").height), (768, 1360));
        assert_eq!((by("DP-2").width, by("DP-2").height), (1920, 1080));
        assert!(by("DP-1").focused);
        assert_eq!(by("DP-2").description, "Main monitor");
    }

    #[test]
    fn monitor_geometry() {
        let m = Monitor {
            name: "A".into(),
            description: String::new(),
            x: 100,
            y: 0,
            width: 200,
            height: 100,
            focused: false,
        };
        assert!(m.contains(100, 0) && m.contains(299, 99));
        assert!(!m.contains(300, 50) && !m.contains(99, 50));
        assert_eq!(m.center(), (200, 50));
        assert_eq!(m.clamp(1000, -5), (297, 2));
        assert_eq!(m.clamp(150, 40), (150, 40));
    }

    #[test]
    fn version_and_syntax() {
        assert_eq!(parse_version(r#"{"version":"0.56.2","tag":"v0.56.2"}"#), Some((0, 56, 2)));
        assert_eq!(parse_version(r#"{"tag":"v0.54.0-12-gabc"}"#), Some((0, 54, 0)));
        assert_eq!(parse_version("not json"), None);
        assert!(uses_lua((0, 55, 0)) && uses_lua((0, 56, 2)) && uses_lua((1, 0, 0)));
        assert!(!uses_lua((0, 54, 9)));
    }

    #[test]
    fn focus_requests() {
        assert_eq!(focus_request(true, "DP-2", None), r#"dispatch hl.dsp.focus({ monitor = "DP-2" })"#);
        assert_eq!(
            focus_request(true, "DP-2", Some((1449, 804))),
            r#"[[BATCH]]dispatch hl.dsp.focus({ monitor = "DP-2" }) ; dispatch hl.dsp.cursor.move({ x = 1449, y = 804 })"#
        );
        assert_eq!(
            focus_request(false, "DP-2", Some((5, 6))),
            "[[BATCH]]dispatch focusmonitor DP-2 ; dispatch movecursor 5 6"
        );
        assert_eq!(focus_request(true, r#"a"b\c"#, None), r#"dispatch hl.dsp.focus({ monitor = "a\"b\\c" })"#);
    }

    #[test]
    fn replies() {
        assert!(reply_ok("ok"));
        assert!(reply_ok("ok\n\n\nok"));
        assert!(!reply_ok("warning: =[C]:-1: hl.focus.monitor: monitor not found"));
        assert!(!reply_ok("ok\n\n\nerror"));
        assert!(!reply_ok(""));
    }

    #[test]
    fn events() {
        assert_eq!(parse_event("focusedmon>>DP-2,3"), Some(HyprEvent::FocusedMonitor("DP-2".into())));
        assert_eq!(parse_event("monitoradded>>HDMI-A-1"), Some(HyprEvent::MonitorAdded("HDMI-A-1".into())));
        assert_eq!(parse_event("monitorremoved>>DP-1"), Some(HyprEvent::MonitorRemoved("DP-1".into())));
        assert_eq!(parse_event("workspace>>2"), None);
        assert_eq!(parse_event("garbage"), None);
    }
}
