//! Settings and calibration files.
//!
//! Both live in `$XDG_CONFIG_HOME/lookfocus/` (usually `~/.config/lookfocus/`):
//!
//! - `config.toml` holds settings you may edit by hand. Every field is
//!   optional and falls back to a default.
//! - `calibration.toml` is written by `lookfocus calibrate`. Keeping it
//!   separate means recalibrating never rewrites your settings or comments.

use std::env;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::classify::Centroid;
use crate::filter::OneEuroParams;
use crate::hypr::Monitor;

pub const SETTINGS_FILE: &str = "config.toml";
pub const CALIBRATION_FILE: &str = "calibration.toml";

pub fn config_dir() -> PathBuf {
    env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
        .unwrap_or_else(|| PathBuf::from("."))
        .join("lookfocus")
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Settings {
    pub camera: CameraSettings,
    pub switching: SwitchSettings,
    pub filter: OneEuroParams,
    pub overrides: OverrideSettings,
    pub adaptive: AdaptiveSettings,
    pub power: PowerSettings,
    /// Directory with the ONNX models. Unset means the usual search path.
    pub models_dir: Option<PathBuf>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OverrideSettings {
    /// Moving the mouse pauses switching for this long.
    pub mouse_hold_ms: u64,
    /// Hold the current monitor while looking down, if calibration could
    /// measure it.
    pub look_down: bool,
}

impl Default for OverrideSettings {
    fn default() -> Self {
        Self { mouse_hold_ms: 2000, look_down: true }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AdaptiveSettings {
    /// Learn from mouse use. The bar button and `lookfocus adaptive` can
    /// switch this at run time, which overrides this value.
    pub enabled: bool,
    /// Fraction of the way to move toward each labelled sample.
    pub rate: f32,
    /// Furthest a centroid may move from its calibrated position, in degrees.
    pub max_drift_deg: f32,
}

impl Default for AdaptiveSettings {
    fn default() -> Self {
        Self { enabled: false, rate: 0.03, max_drift_deg: 6.0 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PowerSettings {
    /// Samples per second while the head is still and nothing is pending.
    pub idle_fps: f32,
    /// Release the camera after this many seconds without a face.
    pub away_after_s: u64,
    /// While away, look for a face this often (seconds). Moving the mouse
    /// also brings tracking back right away.
    pub probe_every_s: u64,
}

impl Default for PowerSettings {
    fn default() -> Self {
        Self { idle_fps: 6.0, away_after_s: 30, probe_every_s: 20 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CameraSettings {
    pub device: PathBuf,
    /// Target samples per second.
    pub fps: f32,
    /// CPU threads per model.
    pub threads: usize,
}

impl Default for CameraSettings {
    fn default() -> Self {
        Self { device: PathBuf::from("/dev/video0"), fps: 15.0, threads: 1 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SwitchSettings {
    /// How long the pose must stay on a new monitor before switching.
    pub dwell_ms: u64,
    /// How much closer (in degrees) a new monitor must be than the current
    /// one before it counts. Unset means a quarter of the gap between the two
    /// closest calibrated monitors, kept between 1 and 4 degrees.
    pub hysteresis_deg: Option<f32>,
    /// The dwell timer only runs while the head moves slower than this, in
    /// degrees per second, so sweeping past a monitor does not select it.
    pub settle_speed: f32,
    /// Where to put the cursor after switching.
    pub cursor: CursorMode,
}

impl Default for SwitchSettings {
    fn default() -> Self {
        Self { dwell_ms: 300, hysteresis_deg: None, settle_speed: 25.0, cursor: CursorMode::Restore }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CursorMode {
    /// Put the cursor back where it last was on that monitor (its center the
    /// first time).
    Restore,
    /// Leave it where Hyprland puts it when focusing a monitor.
    Hyprland,
}

impl Settings {
    /// Loads settings, or defaults if the file does not exist.
    pub fn load(path: &Path) -> Result<Self> {
        match std::fs::read_to_string(path) {
            Ok(text) => toml::from_str(&text).with_context(|| format!("reading {}", path.display())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
        }
    }
}

/// One monitor as it was when calibrating, used to notice layout changes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayoutEntry {
    pub name: String,
    pub description: String,
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

impl From<&Monitor> for LayoutEntry {
    fn from(m: &Monitor) -> Self {
        Self {
            name: m.name.clone(),
            description: m.description.clone(),
            x: m.x,
            y: m.y,
            width: m.width,
            height: m.height,
        }
    }
}

/// Calibrated pose for one monitor.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibratedMonitor {
    pub name: String,
    /// Median pose over all visits, in degrees.
    pub yaw: f32,
    pub pitch: f32,
    /// Jitter while holding still (robust standard deviation), in degrees.
    pub hold_spread: f32,
    /// How far apart separate visits landed, in degrees.
    pub visit_spread: f32,
    /// Share of frames with a face, 0 to 1.
    pub face_rate: f32,
    pub samples: usize,
}

/// Looking down at the keyboard or a phone.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LookDown {
    /// Median pitch while looking down.
    pub pitch: f32,
    /// Pitch below which the pose counts as looking down. Halfway between the
    /// look-down pitch and the lowest monitor.
    pub threshold: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Calibration {
    /// File format version.
    pub version: u32,
    pub created: String,
    /// The monitor the camera sits on, as you answered. Only a hint: recorded
    /// poses always decide.
    pub camera_monitor: Option<String>,
    pub layout: Vec<LayoutEntry>,
    pub monitors: Vec<CalibratedMonitor>,
    /// Missing if looking down could not be told apart from the monitors.
    pub look_down: Option<LookDown>,
}

impl Calibration {
    pub const VERSION: u32 = 1;

    pub fn load(path: &Path) -> Result<Option<Self>> {
        match std::fs::read_to_string(path) {
            Ok(text) => Ok(Some(toml::from_str(&text).with_context(|| format!("reading {}", path.display()))?)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
        }
    }

    /// Writes the file atomically so a crash never leaves half a calibration.
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        let header = "# Written by `lookfocus calibrate`. Run `lookfocus recalibrate` instead of editing.\n\n";
        let text = format!("{header}{}", toml::to_string_pretty(self)?);
        let tmp = path.with_extension("toml.tmp");
        std::fs::write(&tmp, text).with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, path).with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }

    pub fn centroids(&self) -> Vec<Centroid> {
        self.monitors.iter().map(|m| Centroid { monitor: m.name.clone(), yaw: m.yaw, pitch: m.pitch }).collect()
    }

    /// Compares the calibrated layout with the current monitors. Monitors are
    /// matched by connector name and description. Moving a monitor in the
    /// Hyprland layout also counts, since it changes where you look.
    pub fn layout_change(&self, current: &[Monitor]) -> Option<LayoutChange> {
        let now: Vec<LayoutEntry> = current.iter().map(LayoutEntry::from).collect();
        let missing: Vec<String> = self.layout.iter().filter(|e| !now.contains(e)).map(|e| e.name.clone()).collect();
        let added: Vec<String> = now.iter().filter(|e| !self.layout.contains(e)).map(|e| e.name.clone()).collect();
        if missing.is_empty() && added.is_empty() { None } else { Some(LayoutChange { missing, added }) }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LayoutChange {
    /// Calibrated monitors that are gone or have moved.
    pub missing: Vec<String>,
    /// Monitors that are new or have moved.
    pub added: Vec<String>,
}

impl std::fmt::Display for LayoutChange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut parts = Vec::new();
        if !self.missing.is_empty() {
            parts.push(format!("changed or gone: {}", self.missing.join(", ")));
        }
        if !self.added.is_empty() {
            parts.push(format!("new or changed: {}", self.added.join(", ")));
        }
        write!(f, "monitor layout differs from calibration ({})", parts.join("; "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mon(name: &str, x: i32) -> Monitor {
        Monitor {
            name: name.into(),
            description: format!("{name} panel"),
            x,
            y: 0,
            width: 1920,
            height: 1080,
            focused: false,
        }
    }

    fn calibration() -> Calibration {
        Calibration {
            version: Calibration::VERSION,
            created: "2026-10-08T07:40:00".into(),
            camera_monitor: Some("C".into()),
            layout: vec![(&mon("A", 0)).into(), (&mon("B", 1920)).into(), (&mon("C", 3840)).into()],
            monitors: vec![
                CalibratedMonitor {
                    name: "A".into(),
                    yaw: 43.0,
                    pitch: 4.6,
                    hold_spread: 1.5,
                    visit_spread: 2.0,
                    face_rate: 1.0,
                    samples: 60,
                },
                CalibratedMonitor {
                    name: "B".into(),
                    yaw: 31.2,
                    pitch: 2.2,
                    hold_spread: 1.4,
                    visit_spread: 2.8,
                    face_rate: 1.0,
                    samples: 61,
                },
            ],
            look_down: Some(LookDown { pitch: -20.0, threshold: -9.0 }),
        }
    }

    #[test]
    fn settings_default_when_missing_and_partial_files_fill_in() {
        let dir = std::env::temp_dir().join(format!("lookfocus-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        let _ = std::fs::remove_file(&path);
        assert_eq!(Settings::load(&path).unwrap(), Settings::default());

        std::fs::write(&path, "[switching]\ndwell_ms = 450\ncursor = \"hyprland\"\n").unwrap();
        let s = Settings::load(&path).unwrap();
        assert_eq!(s.switching.dwell_ms, 450);
        assert_eq!(s.switching.cursor, CursorMode::Hyprland);
        assert_eq!(s.camera, CameraSettings::default());

        // A typo is an error, not a silently ignored setting.
        std::fs::write(&path, "[switching]\ndwel_ms = 450\n").unwrap();
        assert!(Settings::load(&path).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn calibration_round_trips() {
        let dir = std::env::temp_dir().join(format!("lookfocus-cal-{}", std::process::id()));
        let path = dir.join("calibration.toml");
        let cal = calibration();
        cal.save(&path).unwrap();
        assert_eq!(Calibration::load(&path).unwrap(), Some(cal));
        assert!(!path.with_extension("toml.tmp").exists());
        assert_eq!(Calibration::load(&dir.join("nope.toml")).unwrap(), None);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn detects_layout_changes() {
        let cal = calibration();
        assert_eq!(cal.layout_change(&[mon("A", 0), mon("B", 1920), mon("C", 3840)]), None);
        // Focus state does not matter.
        let mut focused = mon("B", 1920);
        focused.focused = true;
        assert_eq!(cal.layout_change(&[mon("A", 0), focused, mon("C", 3840)]), None);
        // Unplugged.
        let c = cal.layout_change(&[mon("A", 0), mon("B", 1920)]).unwrap();
        assert_eq!((c.missing, c.added), (vec!["C".to_string()], vec![]));
        // Plugged in.
        let c = cal.layout_change(&[mon("A", 0), mon("B", 1920), mon("C", 3840), mon("D", 5760)]).unwrap();
        assert_eq!(c.added, vec!["D".to_string()]);
        // Moved in the layout.
        let c = cal.layout_change(&[mon("A", 0), mon("B", 1920), mon("C", 4000)]).unwrap();
        assert!(c.to_string().contains("monitor layout differs"));
        assert_eq!((c.missing, c.added), (vec!["C".to_string()], vec!["C".to_string()]));
    }

    #[test]
    fn centroids_follow_monitors() {
        let c = calibration().centroids();
        assert_eq!(c.len(), 2);
        assert_eq!((c[0].monitor.as_str(), c[0].yaw), ("A", 43.0));
    }
}
