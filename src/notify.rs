//! Desktop notifications through `notify-send`. Failures are ignored, since a
//! missing notification daemon should never stop tracking.

use std::process::Command;

pub fn send(summary: &str, body: &str, timeout_ms: u32) {
    let result = Command::new("notify-send")
        .args(["--app-name", "lookfocus", "--expire-time", &timeout_ms.to_string(), summary, body])
        .status();
    if let Err(e) = result {
        log::debug!("notify-send failed: {e}");
    }
}
