//! Running the command a gesture is mapped to.
//!
//! Commands run through `sh -c`, so a config line can use pipes, quotes and
//! arguments the way a terminal does. The daemon never waits for one: the shell
//! is started detached, in its own process group, with no input, and a small
//! thread waits for it so the finished process is reaped instead of left as a
//! zombie. A command that cannot start, or exits with an error, is logged and
//! otherwise ignored.
//!
//! The daemon talks to the `ActionRunner` trait, and tests give it a fake one.

use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::process::{Child, Command, ExitStatus, Stdio};

use anyhow::{Context, Result};

use crate::gesture::Gesture;

pub trait ActionRunner {
    /// Starts `command` for `gesture` and returns without waiting for it.
    /// An error means it could not be started.
    fn run(&mut self, gesture: Gesture, command: &str) -> Result<()>;
}

/// Runs commands with `sh -c`.
pub struct ShellRunner;

impl ActionRunner for ShellRunner {
    fn run(&mut self, gesture: Gesture, command: &str) -> Result<()> {
        let child = spawn(command)?;
        let owned = command.to_string();
        std::thread::Builder::new()
            .name("action".into())
            .spawn(move || {
                if let Some(problem) = wait_and_describe(child) {
                    log::warn!("the {gesture} command `{owned}` {problem}");
                }
            })
            .context("could not start a thread to wait for the command")?;
        Ok(())
    }
}

fn spawn(command: &str) -> Result<Child> {
    Command::new("sh")
        .arg("-c")
        .arg(command)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        // Left connected, so a command's complaints reach the journal.
        .stderr(Stdio::inherit())
        // Its own process group, so a Ctrl+C in the terminal running
        // `lookfocus run` does not reach it.
        .process_group(0)
        .spawn()
        .with_context(|| format!("could not start `sh -c {command}`"))
}

/// Waits for the child, which reaps it. Returns what went wrong, if anything,
/// in words that follow "the command".
fn wait_and_describe(mut child: Child) -> Option<String> {
    match child.wait() {
        Ok(status) => describe_failure(status),
        Err(e) => Some(format!("could not be waited for: {e}")),
    }
}

fn describe_failure(status: ExitStatus) -> Option<String> {
    if status.success() {
        None
    } else if let Some(code) = status.code() {
        Some(format!("exited with status {code}{}", hint(code)))
    } else {
        status.signal().map(|s| format!("was stopped by signal {s}"))
    }
}

/// The shell's own codes for a command it could not run.
fn hint(code: i32) -> &'static str {
    match code {
        126 => " (found, but not executable)",
        127 => " (not found on the daemon's PATH)",
        _ => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("lookfocus-actions-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn wait_for(path: &std::path::Path) -> bool {
        let until = Instant::now() + Duration::from_secs(10);
        while Instant::now() < until {
            if path.exists() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        false
    }

    #[test]
    fn runs_the_command_through_a_shell_without_waiting() {
        let dir = scratch("run");
        let marker = dir.join("marker");
        // A pipe and a variable only work through a shell. The sleep makes
        // sure `run` returns before the command is done.
        let command = format!("sleep 0.3; echo \"$((40 + 2))\" | cat > {}", marker.display());
        let started = Instant::now();
        ShellRunner.run(Gesture::OpenPalm, &command).unwrap();
        assert!(started.elapsed() < Duration::from_millis(250), "run waited for the command");
        assert!(wait_for(&marker));
        let until = Instant::now() + Duration::from_secs(5);
        while std::fs::read_to_string(&marker).unwrap_or_default().is_empty() && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert_eq!(std::fs::read_to_string(&marker).unwrap(), "42\n");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn reports_how_a_command_failed() {
        let child = spawn("exit 3").unwrap();
        assert_eq!(wait_and_describe(child).as_deref(), Some("exited with status 3"));
        let child = spawn("lookfocus-no-such-command-xyz 2>/dev/null").unwrap();
        let problem = wait_and_describe(child).unwrap();
        assert!(problem.contains("127") && problem.contains("PATH"), "{problem}");
        let child = spawn("kill -9 $$").unwrap();
        assert!(wait_and_describe(child).unwrap().contains("signal 9"));
        assert_eq!(wait_and_describe(spawn("true").unwrap()), None);
    }

    #[test]
    fn the_command_does_not_get_our_input() {
        // `cat` would block on an inherited terminal. With no input it ends.
        let child = spawn("cat").unwrap();
        assert_eq!(wait_and_describe(child), None);
    }
}
