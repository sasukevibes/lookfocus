//! Runtime state that survives restarts: the adaptive centroids switch and
//! what adaptive learning has learned so far.
//!
//! Lives in `$XDG_STATE_HOME/lookfocus/state.json` (usually
//! `~/.local/state/lookfocus/`), apart from your settings, because the daemon
//! rewrites it on its own.

use std::env;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::classify::Centroid;

pub fn state_path() -> PathBuf {
    env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/state")))
        .unwrap_or_else(|| PathBuf::from("."))
        .join("lookfocus/state.json")
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct State {
    /// Adaptive centroids switched on or off at run time. `None` means use
    /// the setting in config.toml.
    pub adaptive: Option<bool>,
    /// The `created` stamp of the calibration the learned centroids belong
    /// to. A new calibration starts learning from scratch.
    pub calibration: String,
    pub learned: Option<Vec<Centroid>>,
}

impl State {
    /// Loads state. A missing or unreadable file gives the default, since
    /// this is only a cache of preferences and learning.
    pub fn load(path: &Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(text) => serde_json::from_str(&text).unwrap_or_else(|e| {
                log::warn!("ignoring unreadable {}: {e}", path.display());
                Self::default()
            }),
            Err(_) => Self::default(),
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_string_pretty(self)?)
            .with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, path).with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }

    /// Learned centroids, but only if they belong to this calibration.
    pub fn learned_for(&self, calibration_created: &str) -> Option<Vec<Centroid>> {
        if self.calibration == calibration_created { self.learned.clone() } else { None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_tolerates_missing_or_bad_files() {
        let dir = std::env::temp_dir().join(format!("lookfocus-state-{}", std::process::id()));
        let path = dir.join("state.json");
        assert_eq!(State::load(&path), State::default());
        let s = State {
            adaptive: Some(true),
            calibration: "2026-10-08T07:50:00".into(),
            learned: Some(vec![Centroid { monitor: "DP-1".into(), yaw: 38.0, pitch: 0.5 }]),
        };
        s.save(&path).unwrap();
        assert_eq!(State::load(&path), s);
        assert!(s.learned_for("2026-10-08T07:50:00").is_some());
        assert!(s.learned_for("another calibration").is_none());
        std::fs::write(&path, "{not json").unwrap();
        assert_eq!(State::load(&path), State::default());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
