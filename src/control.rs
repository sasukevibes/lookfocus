//! The control socket: how `lookfocus toggle`, `status`, the bar widget and
//! keybinds talk to the running daemon.
//!
//! The socket is `$XDG_RUNTIME_DIR/lookfocus/control.sock`. A client connects,
//! writes one command line, and reads one JSON reply. `watch` is the
//! exception: it keeps the connection open and streams events as JSON lines.
//!
//! Commands: `status`, `pause`, `resume`, `toggle`, `adaptive on|off|toggle|reset`,
//! `reload` (read calibration.toml again), `watch`.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::events::Event;

pub fn socket_path() -> PathBuf {
    let runtime = std::env::var_os("XDG_RUNTIME_DIR").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
    runtime.join("lookfocus/control.sock")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdaptiveCommand {
    On,
    Off,
    Toggle,
    Reset,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Command {
    Status,
    Pause,
    Resume,
    Toggle,
    Adaptive(AdaptiveCommand),
    /// Read the calibration file again, after recalibrating.
    Reload,
    Watch,
}

impl Command {
    pub fn parse(line: &str) -> Result<Self, String> {
        let words: Vec<&str> = line.split_whitespace().collect();
        Ok(match words.as_slice() {
            ["status"] => Command::Status,
            ["pause"] => Command::Pause,
            ["resume"] => Command::Resume,
            ["toggle"] => Command::Toggle,
            ["watch"] => Command::Watch,
            ["reload"] => Command::Reload,
            ["adaptive", "on"] => Command::Adaptive(AdaptiveCommand::On),
            ["adaptive", "off"] => Command::Adaptive(AdaptiveCommand::Off),
            ["adaptive", "toggle"] => Command::Adaptive(AdaptiveCommand::Toggle),
            ["adaptive", "reset"] => Command::Adaptive(AdaptiveCommand::Reset),
            _ => return Err(format!("unknown command {line:?}")),
        })
    }
}

/// What `status` reports.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Status {
    pub running: bool,
    /// One of: tracking, paused, away, layout_changed, camera_unavailable,
    /// stopped.
    pub state: String,
    /// Why the daemon is not tracking, in words.
    pub reason: Option<String>,
    /// The focused monitor as lookfocus knows it.
    pub monitor: Option<String>,
    /// The monitor the head points at right now.
    pub zone: Option<String>,
    pub face: bool,
    /// True while the camera is open.
    pub camera: bool,
    pub adaptive: bool,
    /// True while a recent mouse move is holding switching.
    pub mouse_hold: bool,
    /// Degrees each centroid has drifted through adaptive learning.
    pub drift: Vec<(String, f32)>,
    pub version: String,
}

/// A command from a client, with a way to answer it.
pub enum Request {
    Command {
        command: Command,
        reply: Sender<String>,
    },
    /// A `watch` client: the daemon should send it every event.
    Watch(Sender<Event>),
}

/// Binds the socket. Fails if another daemon is already listening, and
/// clears a stale socket left by a daemon that crashed.
pub fn bind(path: &Path) -> Result<UnixListener> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    if path.exists() {
        if UnixStream::connect(path).is_ok() {
            bail!("lookfocus is already running (socket {} is live)", path.display());
        }
        std::fs::remove_file(path).with_context(|| format!("removing stale {}", path.display()))?;
    }
    UnixListener::bind(path).with_context(|| format!("binding {}", path.display()))
}

/// Accepts clients on a background thread and hands their requests to the
/// daemon through the returned channel.
pub fn serve(listener: UnixListener) -> Receiver<Request> {
    let (tx, rx) = channel();
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let tx = tx.clone();
            std::thread::spawn(move || {
                if let Err(e) = handle_client(stream, &tx) {
                    log::debug!("control client: {e}");
                }
            });
        }
    });
    rx
}

fn handle_client(stream: UnixStream, tx: &Sender<Request>) -> Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(2)))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let mut out = stream;
    match Command::parse(line.trim()) {
        Err(e) => writeln!(out, "{}", serde_json::json!({ "error": e }))?,
        Ok(Command::Watch) => {
            let (etx, erx) = channel();
            tx.send(Request::Watch(etx))?;
            for event in erx {
                // Stops when the client hangs up.
                writeln!(out, "{}", serde_json::to_string(&event)?)?;
            }
        }
        Ok(command) => {
            let (rtx, rrx) = channel();
            tx.send(Request::Command { command, reply: rtx })?;
            let reply = rrx.recv_timeout(Duration::from_secs(3)).context("the daemon did not answer")?;
            writeln!(out, "{reply}")?;
        }
    }
    Ok(())
}

/// Sends one command to the running daemon and returns its JSON reply.
pub fn request(path: &Path, command: &str) -> Result<String> {
    let mut stream = UnixStream::connect(path).with_context(|| "lookfocus is not running".to_string())?;
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    writeln!(stream, "{command}")?;
    let mut reply = String::new();
    BufReader::new(stream).read_line(&mut reply)?;
    Ok(reply.trim().to_string())
}

/// Streams events from the running daemon, calling `f` for each JSON line.
pub fn watch(path: &Path, mut f: impl FnMut(&str) -> Result<()>) -> Result<()> {
    let mut stream = UnixStream::connect(path).with_context(|| "lookfocus is not running".to_string())?;
    writeln!(stream, "watch")?;
    for line in BufReader::new(stream).lines() {
        f(&line?)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_commands() {
        assert_eq!(Command::parse("status"), Ok(Command::Status));
        assert_eq!(Command::parse("  toggle \n"), Ok(Command::Toggle));
        assert_eq!(Command::parse("adaptive reset"), Ok(Command::Adaptive(AdaptiveCommand::Reset)));
        assert!(Command::parse("adaptive maybe").is_err());
        assert!(Command::parse("").is_err());
    }

    #[test]
    fn socket_round_trip_and_stale_socket_cleanup() {
        let dir = std::env::temp_dir().join(format!("lookfocus-ctl-{}", std::process::id()));
        let path = dir.join("control.sock");
        // A stale socket file from a crashed daemon is replaced.
        std::fs::create_dir_all(&dir).unwrap();
        drop(UnixListener::bind(&path).unwrap());
        let listener = bind(&path).unwrap();
        // A second daemon is refused while the first is live.
        assert!(bind(&path).unwrap_err().to_string().contains("already running"));

        let rx = serve(listener);
        std::thread::spawn(move || {
            for req in rx {
                match req {
                    Request::Command { command, reply } => {
                        let _ = reply.send(format!("{{\"got\":\"{command:?}\"}}"));
                    }
                    Request::Watch(tx) => {
                        let _ = tx.send(Event::FaceFound);
                        let _ = tx.send(Event::Resumed);
                    }
                }
            }
        });
        assert_eq!(request(&path, "toggle").unwrap(), r#"{"got":"Toggle"}"#);
        assert!(request(&path, "bogus").unwrap().contains("unknown command"));

        let mut lines = Vec::new();
        watch(&path, |l| {
            lines.push(l.to_string());
            Ok(())
        })
        .unwrap();
        assert_eq!(lines, vec![r#"{"event":"face_found"}"#, r#"{"event":"resumed"}"#]);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
