//! The tracking loop: poses in, monitor switches out.
//!
//! Each tick checks the mouse, decides whether the camera should be open,
//! reads one pose if it is, and asks the switcher whether to move focus.
//!
//! The camera is open only while it is useful. It is released when you pause
//! lookfocus, when the monitor layout no longer matches the calibration, and
//! when no face has been seen for a while ("away"). While away, moving the
//! mouse brings tracking back at once, and a short check runs every so often.
//!
//! Overrides, in the order they apply:
//!
//! 1. Paused by you (bar button, keybind, `lookfocus toggle`).
//! 2. Mouse movement holds switching for a moment (2 s by default). While the
//!    mouse moves on a monitor, the pose also teaches adaptive centroids.
//! 3. No face, or looking down, holds the current monitor.
//!
//! lookfocus never reads the keyboard. Mouse movement is seen only as cursor
//! position changes reported by Hyprland, ignoring moves lookfocus made.
//!
//! The daemon is generic over its pose source, compositor and clock, so the
//! whole loop is tested with scripted poses, a fake Hyprland and fake time.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};

use anyhow::Result;

use crate::adaptive::{Adaptive, AdaptiveParams};
use crate::classify::Classifier;
use crate::config::{Calibration, CursorMode, Settings};
use crate::control::{AdaptiveCommand, Command, Request, Status};
use crate::events::{Event, EventBus};
use crate::filter::PoseFilter;
use crate::hypr::{Compositor, HyprEvent, Monitor};
use crate::sampler::PoseSource;
use crate::state::State;
use crate::switcher::{Decision, SwitchParams, Switcher, auto_hysteresis};

/// Opens the pose source (camera plus models). Called again after each
/// release.
pub type Opener<P> = Box<dyn FnMut() -> Result<P>>;
pub type Clock = Box<dyn Fn() -> Instant>;

/// What a tick did, so the caller knows whether to sleep.
#[derive(Clone, Debug, PartialEq)]
pub enum Tick {
    /// The camera is closed. Sleep a little before the next tick.
    Idle,
    /// A sample was read (and maybe acted on).
    Sampled(Decision),
}

struct Timing {
    mouse_hold: Duration,
    away_after: Duration,
    probe_every: Duration,
    probe_length: Duration,
    camera_retry: Duration,
    /// Frames right after opening are skipped while auto exposure settles.
    warmup: Duration,
    fps: f32,
    idle_fps: f32,
    /// How long the head must be still before dropping to `idle_fps`.
    idle_after: Duration,
}

pub struct Daemon<P, C> {
    source: Option<P>,
    open: Opener<P>,
    clock: Clock,
    compositor: C,
    calibration: Calibration,
    switcher: Switcher,
    filter: PoseFilter,
    cursor_mode: CursorMode,
    look_down: Option<f32>,
    timing: Timing,

    adaptive: Adaptive,
    adaptive_on: bool,
    state_path: Option<PathBuf>,
    /// Where `reload` reads the calibration from.
    pub calibration_path: Option<PathBuf>,
    settings: Settings,
    state_dirty: bool,
    state_saved: Instant,

    monitors: Vec<Monitor>,
    cursor_memory: HashMap<String, (i32, i32)>,
    last_cursor: Option<(i32, i32)>,
    mouse_hold_until: Option<Instant>,

    user_paused: bool,
    layout_issue: Option<String>,
    camera_issue: Option<String>,
    camera_retry_at: Option<Instant>,
    away: bool,
    probe_until: Option<Instant>,
    next_probe_at: Option<Instant>,
    last_face_at: Instant,
    warmup_until: Option<Instant>,
    current_fps: f32,
    still_since: Option<Instant>,

    start: Instant,
    face: bool,
    zone: Option<usize>,
    pub events: EventBus,
}

impl<P: PoseSource, C: Compositor> Daemon<P, C> {
    /// `state_path` is where adaptive learning and the adaptive switch are
    /// kept between runs. `None` keeps them in memory only.
    pub fn new(
        open: Opener<P>,
        compositor: C,
        calibration: Calibration,
        settings: &Settings,
        state_path: Option<PathBuf>,
        clock: Clock,
    ) -> Result<Self> {
        let state = state_path.as_deref().map(State::load).unwrap_or_default();
        let base = calibration.centroids();
        let adaptive = Adaptive::new(
            AdaptiveParams {
                rate: settings.adaptive.rate,
                max_drift: settings.adaptive.max_drift_deg,
                ..AdaptiveParams::default()
            },
            base.clone(),
            state.learned_for(&calibration.created),
        );
        let adaptive_on = state.adaptive.unwrap_or(settings.adaptive.enabled);
        let classifier = Classifier::new(if adaptive_on { adaptive.centroids().to_vec() } else { base.clone() });
        // The margin comes from the calibrated gaps, not the learned ones, so
        // learning cannot shrink it.
        let hysteresis = settings.switching.hysteresis_deg.unwrap_or_else(|| auto_hysteresis(&Classifier::new(base)));
        let params = SwitchParams {
            dwell: Duration::from_millis(settings.switching.dwell_ms),
            hysteresis,
            settle_speed: settings.switching.settle_speed,
        };
        log::info!(
            "dwell {} ms, hysteresis {hysteresis:.1} degrees, settle speed {} deg/s, adaptive {}",
            settings.switching.dwell_ms,
            settings.switching.settle_speed,
            if adaptive_on { "on" } else { "off" }
        );
        let look_down =
            if settings.overrides.look_down { calibration.look_down.as_ref().map(|l| l.threshold) } else { None };
        let now = clock();
        let mut d = Self {
            source: None,
            open,
            clock,
            compositor,
            calibration,
            switcher: Switcher::new(classifier, params),
            filter: PoseFilter::new(settings.filter),
            cursor_mode: settings.switching.cursor,
            look_down,
            timing: Timing {
                mouse_hold: Duration::from_millis(settings.overrides.mouse_hold_ms),
                away_after: Duration::from_secs(settings.power.away_after_s),
                probe_every: Duration::from_secs(settings.power.probe_every_s),
                probe_length: Duration::from_secs(4),
                camera_retry: Duration::from_secs(5),
                warmup: Duration::from_millis(1000),
                fps: settings.camera.fps,
                idle_fps: settings.power.idle_fps.min(settings.camera.fps),
                idle_after: Duration::from_secs(1),
            },
            adaptive,
            adaptive_on,
            state_path,
            calibration_path: None,
            settings: settings.clone(),
            state_dirty: false,
            state_saved: now,
            monitors: Vec::new(),
            cursor_memory: HashMap::new(),
            last_cursor: None,
            mouse_hold_until: None,
            user_paused: false,
            layout_issue: None,
            camera_issue: None,
            camera_retry_at: None,
            away: false,
            probe_until: None,
            next_probe_at: None,
            last_face_at: now,
            warmup_until: None,
            current_fps: settings.camera.fps,
            still_since: None,
            start: now,
            face: false,
            zone: None,
            events: EventBus::default(),
        };
        d.refresh_monitors()?;
        Ok(d)
    }

    /// Announces the calibrated monitors. Call after subscribing to events.
    pub fn start(&mut self) {
        let monitors = self.calibration.monitors.iter().map(|m| m.name.clone()).collect();
        self.events.emit(Event::Started { monitors });
        if let Some(reason) = self.layout_issue.clone() {
            self.events.emit(Event::Paused { reason });
        }
    }

    /// Runs until an unrecoverable error. Hyprland events and control
    /// requests arrive on their own channels.
    pub fn run(&mut self, hypr: Receiver<HyprEvent>, control: Receiver<Request>) -> Result<()> {
        loop {
            for e in hypr.try_iter() {
                if let Err(err) = self.handle_hypr_event(e) {
                    log::warn!("handling a Hyprland event: {err}");
                }
            }
            for r in control.try_iter() {
                match r {
                    Request::Command { command, reply } => {
                        let _ = reply.send(self.handle_command(command));
                    }
                    Request::Watch(tx) => self.events.add(tx),
                }
            }
            if self.tick()? == Tick::Idle {
                std::thread::sleep(Duration::from_millis(200));
            }
        }
    }

    // ------------------------------------------------------------ state

    pub fn state_name(&self) -> &'static str {
        if self.user_paused {
            "paused"
        } else if self.layout_issue.is_some() {
            "layout_changed"
        } else if self.camera_issue.is_some() {
            "camera_unavailable"
        } else if self.away {
            "away"
        } else {
            "tracking"
        }
    }

    pub fn status(&self) -> Status {
        let reason = if self.user_paused {
            Some("Paused. Resume from the bar, with the keybind, or `lookfocus resume`.".to_string())
        } else {
            self.layout_issue.clone().or_else(|| self.camera_issue.clone()).or_else(|| {
                self.away.then(|| "No one in front of the camera, so it is released. Move the mouse to resume.".into())
            })
        };
        let now = (self.clock)();
        let names = self.adaptive.base().iter().map(|c| c.monitor.clone());
        Status {
            running: true,
            state: self.state_name().to_string(),
            reason,
            monitor: self.switcher.current().map(|i| self.name_of(i).to_string()),
            zone: self.zone.map(|i| self.name_of(i).to_string()),
            face: self.face,
            camera: self.source.is_some(),
            adaptive: self.adaptive_on,
            mouse_hold: self.mouse_hold_until.is_some_and(|t| now < t),
            drift: names.zip(self.adaptive.drift()).collect(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    pub fn handle_command(&mut self, command: Command) -> String {
        match command {
            Command::Status | Command::Watch => {}
            Command::Reload => {
                if let Err(e) = self.reload() {
                    log::warn!("reload failed: {e:#}");
                    return serde_json::json!({ "error": format!("{e:#}") }).to_string();
                }
            }
            Command::Pause => self.set_paused(true),
            Command::Resume => self.set_paused(false),
            Command::Toggle => self.set_paused(!self.user_paused),
            Command::Adaptive(a) => {
                match a {
                    AdaptiveCommand::On => self.set_adaptive(true),
                    AdaptiveCommand::Off => self.set_adaptive(false),
                    AdaptiveCommand::Toggle => self.set_adaptive(!self.adaptive_on),
                    AdaptiveCommand::Reset => {
                        self.adaptive.reset();
                        self.apply_centroids();
                        self.state_dirty = true;
                    }
                }
                self.save_state();
            }
        }
        serde_json::to_string(&self.status()).unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"))
    }

    /// Swaps in a new calibration, keeping the camera, pause state and
    /// adaptive switch. Learning restarts for the new calibration.
    pub fn reload(&mut self) -> Result<()> {
        let path = self.calibration_path.clone().ok_or_else(|| anyhow::anyhow!("no calibration path"))?;
        let calibration = Calibration::load(&path)?.ok_or_else(|| anyhow::anyhow!("{} is missing", path.display()))?;
        self.save_state();
        let base = calibration.centroids();
        self.adaptive = Adaptive::new(
            AdaptiveParams {
                rate: self.settings.adaptive.rate,
                max_drift: self.settings.adaptive.max_drift_deg,
                ..AdaptiveParams::default()
            },
            base.clone(),
            None,
        );
        let hysteresis =
            self.settings.switching.hysteresis_deg.unwrap_or_else(|| auto_hysteresis(&Classifier::new(base.clone())));
        let mut params = self.switcher.params();
        params.hysteresis = hysteresis;
        self.switcher = Switcher::new(Classifier::new(base), params);
        self.look_down =
            if self.settings.overrides.look_down { calibration.look_down.as_ref().map(|l| l.threshold) } else { None };
        self.calibration = calibration;
        self.apply_centroids();
        self.zone = None;
        self.state_dirty = true;
        self.save_state();
        self.refresh_monitors()?;
        log::info!("calibration reloaded");
        Ok(())
    }

    fn set_paused(&mut self, paused: bool) {
        if paused == self.user_paused {
            return;
        }
        self.user_paused = paused;
        if paused {
            log::info!("paused");
            // Release now, so the camera is free as soon as the reply arrives.
            self.release_camera();
            self.save_state();
            self.events.emit(Event::Paused { reason: "paused".into() });
        } else {
            log::info!("resumed");
            self.away = false;
            self.probe_until = None;
            self.next_probe_at = None;
            self.last_face_at = (self.clock)();
            self.events.emit(Event::Resumed);
        }
    }

    fn set_adaptive(&mut self, on: bool) {
        if on != self.adaptive_on {
            self.adaptive_on = on;
            self.apply_centroids();
            self.state_dirty = true;
            log::info!("adaptive centroids {}", if on { "on" } else { "off" });
            self.events.emit(Event::AdaptiveChanged { enabled: on });
        }
    }

    fn apply_centroids(&mut self) {
        let c = if self.adaptive_on { self.adaptive.centroids().to_vec() } else { self.adaptive.base().to_vec() };
        self.switcher.classifier_mut().set_centroids(c);
    }

    fn save_state(&mut self) {
        let Some(path) = &self.state_path else { return };
        if !self.state_dirty {
            return;
        }
        let state = State {
            adaptive: Some(self.adaptive_on),
            calibration: self.calibration.created.clone(),
            learned: Some(self.adaptive.centroids().to_vec()),
        };
        match state.save(path) {
            Ok(()) => self.state_dirty = false,
            Err(e) => log::warn!("could not save state: {e}"),
        }
        self.state_saved = (self.clock)();
    }

    fn index_of(&self, name: &str) -> Option<usize> {
        self.calibration.monitors.iter().position(|m| m.name == name)
    }

    fn name_of(&self, index: usize) -> &str {
        &self.calibration.monitors[index].name
    }

    // ------------------------------------------------------------ Hyprland

    /// Re-reads monitors, pauses switching if the layout no longer matches
    /// the calibration, and resumes when it matches again.
    pub fn refresh_monitors(&mut self) -> Result<()> {
        self.monitors = self.compositor.monitors()?;
        let focused = self.monitors.iter().find(|m| m.focused).and_then(|m| self.index_of(&m.name));
        self.switcher.set_current(focused);
        match self.calibration.layout_change(&self.monitors) {
            Some(change) => {
                let reason = format!("{change}. Run `lookfocus recalibrate`.");
                if self.layout_issue.as_ref() != Some(&reason) {
                    log::warn!("{reason}");
                    self.layout_issue = Some(reason.clone());
                    self.events.emit(Event::Paused { reason });
                }
            }
            None => {
                if self.layout_issue.take().is_some() {
                    log::info!("monitor layout matches the calibration again");
                    self.events.emit(Event::Resumed);
                }
            }
        }
        Ok(())
    }

    pub fn handle_hypr_event(&mut self, event: HyprEvent) -> Result<()> {
        match event {
            HyprEvent::FocusedMonitor(name) => {
                let index = self.index_of(&name);
                if index != self.switcher.current() {
                    self.switcher.set_current(index);
                    self.events.emit(Event::FocusChanged { monitor: name });
                }
            }
            HyprEvent::MonitorAdded(_) | HyprEvent::MonitorRemoved(_) => self.refresh_monitors()?,
        }
        Ok(())
    }

    /// Notices cursor moves that lookfocus did not make. Returns true if the
    /// mouse moved.
    fn check_mouse(&mut self, now: Instant) -> bool {
        let Ok(pos) = self.compositor.cursor_pos() else { return false };
        let moved = self.last_cursor.is_some_and(|(x, y)| (pos.0 - x).abs() + (pos.1 - y).abs() > 2);
        self.last_cursor = Some(pos);
        if moved {
            self.mouse_hold_until = Some(now + self.timing.mouse_hold);
        }
        moved
    }

    fn mouse_monitor(&self) -> Option<usize> {
        let (x, y) = self.last_cursor?;
        let m = self.monitors.iter().find(|m| m.contains(x, y))?;
        self.index_of(&m.name)
    }

    // ------------------------------------------------------------ camera

    fn release_camera(&mut self) {
        if self.source.take().is_some() {
            log::info!("camera released");
        }
        self.filter.reset();
        self.switcher.hold();
        self.warmup_until = None;
        self.still_since = None;
        if self.face {
            self.face = false;
            self.events.emit(Event::FaceLost);
        }
    }

    fn open_camera(&mut self, now: Instant) -> bool {
        if self.camera_retry_at.is_some_and(|t| now < t) {
            return false;
        }
        match (self.open)() {
            Ok(mut source) => {
                source.set_fps(self.timing.fps);
                self.current_fps = self.timing.fps;
                self.source = Some(source);
                self.warmup_until = Some(now + self.timing.warmup);
                self.last_face_at = now;
                self.camera_retry_at = None;
                if self.camera_issue.take().is_some() {
                    self.events.emit(Event::Resumed);
                }
                log::info!("camera open");
                true
            }
            Err(e) => {
                let reason = format!("{e:#}");
                self.camera_retry_at = Some(now + self.timing.camera_retry);
                if self.camera_issue.as_ref() != Some(&reason) {
                    log::warn!("camera unavailable: {reason}");
                    self.camera_issue = Some(reason.clone());
                    self.events.emit(Event::CameraUnavailable { reason });
                }
                false
            }
        }
    }

    // ------------------------------------------------------------ tick

    pub fn tick(&mut self) -> Result<Tick> {
        let now = (self.clock)();
        let moved = self.check_mouse(now);

        if self.away && moved {
            self.come_back(now);
        }
        if self.away && self.probe_until.is_none() && self.next_probe_at.is_some_and(|t| now >= t) {
            log::debug!("checking for a face");
            self.probe_until = Some(now + self.timing.probe_length);
        }
        let probing = self.probe_until.is_some();
        let want_camera = !self.user_paused && self.layout_issue.is_none() && (!self.away || probing);

        if !want_camera {
            self.release_camera();
            self.save_state_if_due(now);
            return Ok(Tick::Idle);
        }
        if self.source.is_none() && !self.open_camera(now) {
            return Ok(Tick::Idle);
        }

        let sample = match self.source.as_mut().map(|s| s.sample()) {
            Some(Ok(s)) => s,
            Some(Err(e)) => {
                // The camera went away mid-stream (unplugged, taken over).
                let reason = format!("{e:#}");
                log::warn!("camera stopped: {reason}");
                self.release_camera();
                self.camera_issue = Some(reason.clone());
                self.camera_retry_at = Some(now + self.timing.camera_retry);
                self.events.emit(Event::CameraUnavailable { reason });
                return Ok(Tick::Idle);
            }
            None => return Ok(Tick::Idle),
        };
        if self.warmup_until.is_some_and(|t| sample.time < t) {
            return Ok(Tick::Sampled(Decision::Stay));
        }
        self.warmup_until = None;

        let Some(face) = sample.face else {
            if self.face {
                self.face = false;
                self.events.emit(Event::FaceLost);
            }
            self.filter.reset();
            self.switcher.hold();
            self.set_fps(self.timing.fps);
            if self.probe_until.is_some_and(|t| now >= t) {
                // A check found nobody. Release again until the next one.
                self.probe_until = None;
                self.next_probe_at = Some(now + self.timing.probe_every);
                self.release_camera();
            } else if !self.away && now.duration_since(self.last_face_at) >= self.timing.away_after {
                log::info!("no face for {}s, releasing the camera", self.timing.away_after.as_secs());
                self.away = true;
                self.next_probe_at = Some(now + self.timing.probe_every);
                self.events.emit(Event::Away);
                self.release_camera();
            }
            return Ok(Tick::Sampled(Decision::Stay));
        };

        self.last_face_at = now;
        if self.away {
            self.come_back(now);
        }
        if !self.face {
            self.face = true;
            self.events.emit(Event::FaceFound);
        }
        let t = sample.time.duration_since(self.start).as_secs_f32();
        let (yaw, pitch) = self.filter.filter(t, face.pose.yaw, face.pose.pitch);
        let speed = self.filter.speed();

        let zone = self.switcher.classifier().classify(yaw, pitch).map(|c| c.best);
        if zone != self.zone {
            self.zone = zone;
            if let Some(z) = zone {
                self.events.emit(Event::Zone { monitor: self.name_of(z).to_string() });
            }
        }

        let decision = self.decide(now, sample.time, yaw, pitch, speed);
        self.adjust_rate(now, &decision, speed);
        self.save_state_if_due(now);
        Ok(Tick::Sampled(decision))
    }

    fn come_back(&mut self, now: Instant) {
        self.away = false;
        self.probe_until = None;
        self.next_probe_at = None;
        self.last_face_at = now;
        log::info!("back");
        self.events.emit(Event::Back);
    }

    fn decide(&mut self, now: Instant, time: Instant, yaw: f32, pitch: f32, speed: f32) -> Decision {
        // Mouse in use: hold, and learn where this monitor really is.
        if self.mouse_hold_until.is_some_and(|t| now < t) {
            self.switcher.hold();
            if self.adaptive_on
                && speed < self.switcher.params().settle_speed
                && let Some(i) = self.mouse_monitor()
                && self.adaptive.learn(now, i, yaw, pitch)
            {
                self.apply_centroids();
                self.state_dirty = true;
            }
            return Decision::Stay;
        }
        if self.look_down.is_some_and(|threshold| pitch < threshold) {
            self.switcher.hold();
            return Decision::Stay;
        }
        let decision = self.switcher.update(time, yaw, pitch, speed);
        if let Decision::Switch { from, to } = decision {
            let to_name = self.name_of(to).to_string();
            let from_name = from.map(|f| self.name_of(f).to_string());
            match self.switch_to(&to_name) {
                Ok(()) => {
                    log::info!(
                        "switched {} -> {to_name} (yaw {yaw:+.1}, pitch {pitch:+.1})",
                        from_name.as_deref().unwrap_or("?")
                    );
                    self.events.emit(Event::Switched { from: from_name, to: to_name });
                }
                Err(e) => {
                    log::warn!("could not focus {to_name}: {e}");
                    // Undo, so the switcher does not believe the switch happened.
                    self.switcher.set_current(from);
                }
            }
        }
        decision
    }

    fn switch_to(&mut self, name: &str) -> Result<()> {
        // Remember where the cursor is now, on whichever monitor holds it.
        if let Ok(pos) = self.compositor.cursor_pos()
            && let Some(m) = self.monitors.iter().find(|m| m.contains(pos.0, pos.1))
        {
            self.cursor_memory.insert(m.name.clone(), pos);
        }
        let cursor = match self.cursor_mode {
            CursorMode::Hyprland => None,
            CursorMode::Restore => self.monitors.iter().find(|m| m.name == name).map(|m| {
                let (x, y) = self.cursor_memory.get(name).copied().unwrap_or_else(|| m.center());
                m.clamp(x, y)
            }),
        };
        self.compositor.focus(name, cursor)?;
        // Our own move is not the user moving the mouse.
        self.last_cursor = self.compositor.cursor_pos().ok().or(cursor);
        Ok(())
    }

    /// Drops to the idle rate while the head is still and nothing is
    /// pending, and goes back to the full rate as soon as anything happens.
    fn adjust_rate(&mut self, now: Instant, decision: &Decision, speed: f32) {
        let still = matches!(decision, Decision::Stay)
            && speed < self.switcher.params().settle_speed / 2.0
            && !self.mouse_hold_until.is_some_and(|t| now < t);
        if still {
            let since = *self.still_since.get_or_insert(now);
            if now.duration_since(since) >= self.timing.idle_after {
                self.set_fps(self.timing.idle_fps);
            }
        } else {
            self.still_since = None;
            self.set_fps(self.timing.fps);
        }
    }

    fn set_fps(&mut self, fps: f32) {
        if (fps - self.current_fps).abs() > f32::EPSILON {
            self.current_fps = fps;
            if let Some(s) = self.source.as_mut() {
                s.set_fps(fps);
            }
        }
    }

    fn save_state_if_due(&mut self, now: Instant) {
        if self.state_dirty && now.duration_since(self.state_saved) >= Duration::from_secs(60) {
            self.save_state();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CalibratedMonitor, LayoutEntry, LookDown};
    use crate::hypr::HyprError;
    use crate::pose::HeadPose;
    use crate::sampler::{FaceSample, Sample};
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;

    const FRAME: Duration = Duration::from_millis(66);

    fn monitors() -> Vec<Monitor> {
        let m = |name: &str, x: i32, w: i32, h: i32| Monitor {
            name: name.into(),
            description: format!("{name} screen"),
            x,
            y: 0,
            width: w,
            height: h,
            focused: name == "DP-2",
        };
        vec![m("DP-1", 0, 768, 1360), m("DP-2", 768, 1920, 1080), m("eDP-2", 2688, 1600, 1000)]
    }

    fn calibration() -> Calibration {
        let cal = |name: &str, yaw: f32, pitch: f32| CalibratedMonitor {
            name: name.into(),
            yaw,
            pitch,
            hold_spread: 1.5,
            visit_spread: 2.0,
            face_rate: 1.0,
            samples: 60,
        };
        Calibration {
            version: 1,
            created: "now".into(),
            camera_monitor: Some("eDP-2".into()),
            layout: monitors().iter().map(LayoutEntry::from).collect(),
            monitors: vec![cal("DP-1", 43.0, 4.6), cal("DP-2", 31.2, 2.2), cal("eDP-2", 22.9, 1.9)],
            look_down: None,
        }
    }

    /// A recorded focus request: monitor name and cursor target.
    type FocusCall = (String, Option<(i32, i32)>);

    #[derive(Default)]
    struct FakeHypr {
        monitors: RefCell<Vec<Monitor>>,
        cursor: Cell<(i32, i32)>,
        focus_calls: RefCell<Vec<FocusCall>>,
        fail: Cell<bool>,
    }

    impl Compositor for Rc<FakeHypr> {
        fn monitors(&self) -> Result<Vec<Monitor>, HyprError> {
            Ok(self.monitors.borrow().clone())
        }
        fn cursor_pos(&self) -> Result<(i32, i32), HyprError> {
            Ok(self.cursor.get())
        }
        fn focus(&self, monitor: &str, cursor: Option<(i32, i32)>) -> Result<(), HyprError> {
            if self.fail.get() {
                return Err(HyprError::Rejected { request: "focus".into(), reply: "nope".into() });
            }
            self.focus_calls.borrow_mut().push((monitor.into(), cursor));
            if let Some(c) = cursor {
                self.cursor.set(c);
            }
            Ok(())
        }
    }

    /// Poses shared by every source the opener creates, so reopening the
    /// camera continues the script. Each sample advances the fake clock.
    struct Script {
        poses: Vec<Option<(f32, f32)>>,
        i: usize,
        clock: Rc<Cell<Instant>>,
        opens: usize,
        fail_open: bool,
        fps: Vec<f32>,
    }

    struct Source(Rc<RefCell<Script>>);

    impl PoseSource for Source {
        fn sample(&mut self) -> Result<Sample> {
            let mut s = self.0.borrow_mut();
            let pose = s.poses.get(s.i).copied().flatten();
            s.i += 1;
            let t = s.clock.get() + FRAME;
            s.clock.set(t);
            let face = pose.map(|(yaw, pitch)| FaceSample {
                pose: HeadPose { yaw, pitch, roll: 0.0 },
                presence: 1.0,
                tracked: true,
                luma: 100.0,
            });
            Ok(Sample { time: t, face, capture: Duration::ZERO, inference: Duration::ZERO })
        }
        fn set_fps(&mut self, fps: f32) {
            self.0.borrow_mut().fps.push(fps);
        }
    }

    struct Rig {
        d: Daemon<Source, Rc<FakeHypr>>,
        hypr: Rc<FakeHypr>,
        script: Rc<RefCell<Script>>,
        clock: Rc<Cell<Instant>>,
        events: std::sync::mpsc::Receiver<Event>,
    }

    impl Rig {
        fn new(poses: Vec<Option<(f32, f32)>>) -> Self {
            Self::with(poses, calibration(), Settings::default())
        }

        fn with(poses: Vec<Option<(f32, f32)>>, cal: Calibration, settings: Settings) -> Self {
            let clock = Rc::new(Cell::new(Instant::now()));
            let hypr = Rc::new(FakeHypr::default());
            *hypr.monitors.borrow_mut() = monitors();
            hypr.cursor.set((1500, 500)); // on DP-2
            let script = Rc::new(RefCell::new(Script {
                poses,
                i: 0,
                clock: clock.clone(),
                opens: 0,
                fail_open: false,
                fps: Vec::new(),
            }));
            let s2 = script.clone();
            let opener: Opener<Source> = Box::new(move || {
                let mut s = s2.borrow_mut();
                if s.fail_open {
                    anyhow::bail!("camera /dev/video0 is busy");
                }
                s.opens += 1;
                drop(s);
                Ok(Source(s2.clone()))
            });
            let c2 = clock.clone();
            let mut d = Daemon::new(opener, hypr.clone(), cal, &settings, None, Box::new(move || c2.get())).unwrap();
            d.timing.warmup = Duration::ZERO;
            let events = d.events.subscribe();
            d.start();
            Rig { d, hypr, script, clock, events }
        }

        fn done(&self) -> bool {
            let s = self.script.borrow();
            s.i >= s.poses.len()
        }

        fn run_all(&mut self) {
            while !self.done() {
                if self.d.tick().unwrap() == Tick::Idle {
                    self.clock.set(self.clock.get() + Duration::from_millis(200));
                }
            }
        }

        fn events(&self) -> Vec<Event> {
            self.events.try_iter().collect()
        }

        fn focus_calls(&self) -> Vec<FocusCall> {
            self.hypr.focus_calls.borrow().clone()
        }
    }

    fn hold(yaw: f32, pitch: f32, frames: usize) -> Vec<Option<(f32, f32)>> {
        vec![Some((yaw, pitch)); frames]
    }

    #[test]
    fn looking_at_a_monitor_focuses_it_and_restores_the_cursor() {
        let mut poses = hold(31.0, 2.0, 15);
        poses.extend(hold(23.0, 2.0, 20));
        poses.extend(hold(31.0, 2.0, 20));
        let mut r = Rig::new(poses);
        r.run_all();
        let calls = r.focus_calls();
        assert_eq!(calls.len(), 2, "{calls:?}");
        assert_eq!(calls[0], ("eDP-2".to_string(), Some((3488, 500))));
        assert_eq!(calls[1], ("DP-2".to_string(), Some((1500, 500))));
        let events = r.events();
        assert!(matches!(events[0], Event::Started { .. }));
        assert!(events.contains(&Event::Switched { from: Some("DP-2".into()), to: "eDP-2".into() }));
        assert!(events.contains(&Event::Zone { monitor: "eDP-2".into() }));
    }

    #[test]
    fn our_own_cursor_moves_do_not_count_as_mouse_use() {
        let mut poses = hold(23.0, 2.0, 20);
        poses.extend(hold(31.0, 2.0, 20));
        let mut r = Rig::new(poses);
        r.run_all();
        // Both switches happen, with no mouse hold between them.
        assert_eq!(r.focus_calls().len(), 2);
    }

    #[test]
    fn hyprland_cursor_mode_sends_no_cursor() {
        let mut settings = Settings::default();
        settings.switching.cursor = CursorMode::Hyprland;
        let mut r = Rig::with(hold(23.0, 2.0, 20), calibration(), settings);
        r.run_all();
        assert_eq!(r.focus_calls(), vec![("eDP-2".to_string(), None)]);
    }

    #[test]
    fn moving_the_mouse_holds_switching() {
        let mut r = Rig::new(hold(23.0, 2.0, 80));
        // Wiggle the mouse on DP-2 for the first 20 frames.
        for i in 0..20 {
            r.hypr.cursor.set((1500 + (i % 2) * 10, 500));
            r.d.tick().unwrap();
        }
        assert!(r.focus_calls().is_empty());
        assert!(r.d.status().mouse_hold);
        // Then let go: after the 2 s hold and the dwell, it switches.
        r.run_all();
        assert_eq!(r.focus_calls().len(), 1);
    }

    #[test]
    fn pausing_releases_the_camera_and_resuming_reopens_it() {
        let mut r = Rig::new(hold(23.0, 2.0, 40));
        r.d.tick().unwrap();
        assert!(r.d.status().camera);
        let reply = r.d.handle_command(Command::Toggle);
        assert!(reply.contains("\"state\":\"paused\""), "{reply}");
        assert!(reply.contains("\"camera\":false"), "released before replying: {reply}");
        assert_eq!(r.d.tick().unwrap(), Tick::Idle);
        assert!(!r.d.status().camera);
        assert!(r.focus_calls().is_empty());
        r.d.handle_command(Command::Resume);
        r.run_all();
        assert_eq!(r.script.borrow().opens, 2);
        assert_eq!(r.focus_calls().len(), 1);
        let events = r.events();
        assert!(events.contains(&Event::Paused { reason: "paused".into() }));
        assert!(events.contains(&Event::Resumed));
    }

    #[test]
    fn losing_the_face_holds_the_monitor() {
        let mut poses = hold(23.0, 2.0, 3);
        poses.extend(vec![None; 10]);
        poses.extend(hold(23.0, 2.0, 3));
        let mut r = Rig::new(poses);
        r.run_all();
        assert!(r.focus_calls().is_empty());
        let events = r.events();
        assert!(events.contains(&Event::FaceLost) && events.contains(&Event::FaceFound));
    }

    #[test]
    fn away_releases_the_camera_and_the_mouse_brings_it_back() {
        let mut settings = Settings::default();
        settings.power.away_after_s = 2;
        settings.power.probe_every_s = 60;
        let mut poses = vec![None; 40];
        poses.extend(hold(23.0, 2.0, 20));
        let mut r = Rig::with(poses, calibration(), settings);
        // 2 s of no face (about 31 frames) releases the camera.
        for _ in 0..40 {
            r.d.tick().unwrap();
            if r.d.status().state == "away" {
                break;
            }
        }
        assert_eq!(r.d.status().state, "away");
        assert!(!r.d.status().camera);
        assert_eq!(r.d.tick().unwrap(), Tick::Idle);
        // The mouse moves: tracking is back at once.
        r.hypr.cursor.set((1600, 520));
        r.script.borrow_mut().i = 40;
        r.d.tick().unwrap();
        assert_eq!(r.d.status().state, "tracking");
        assert!(r.d.status().camera);
        let events = r.events();
        assert!(events.contains(&Event::Away) && events.contains(&Event::Back));
    }

    #[test]
    fn away_probes_now_and_then() {
        let mut settings = Settings::default();
        settings.power.away_after_s = 1;
        settings.power.probe_every_s = 3;
        let mut r = Rig::with(vec![None; 200], calibration(), settings);
        for _ in 0..200 {
            if r.d.tick().unwrap() == Tick::Idle {
                r.clock.set(r.clock.get() + Duration::from_millis(200));
            }
        }
        // Opened at start, then again for each probe.
        assert!(r.script.borrow().opens >= 3, "{}", r.script.borrow().opens);
        assert_eq!(r.d.status().state, "away");
    }

    #[test]
    fn busy_camera_is_retried() {
        let mut r = Rig::new(hold(23.0, 2.0, 30));
        r.script.borrow_mut().fail_open = true;
        assert_eq!(r.d.tick().unwrap(), Tick::Idle);
        let s = r.d.status();
        assert_eq!(s.state, "camera_unavailable");
        assert!(s.reason.unwrap().contains("busy"));
        // No retry before 5 s.
        r.script.borrow_mut().fail_open = false;
        assert_eq!(r.d.tick().unwrap(), Tick::Idle);
        r.clock.set(r.clock.get() + Duration::from_secs(6));
        r.run_all();
        assert_eq!(r.d.status().state, "tracking");
        assert_eq!(r.focus_calls().len(), 1);
        assert!(r.events().iter().any(|e| matches!(e, Event::CameraUnavailable { .. })));
    }

    #[test]
    fn looking_down_holds_the_monitor() {
        let mut cal = calibration();
        cal.look_down = Some(LookDown { pitch: -20.0, threshold: -9.0 });
        // Looking down and a little right, as when glancing at a phone.
        let mut r = Rig::with(hold(24.0, -18.0, 30), cal, Settings::default());
        r.run_all();
        assert!(r.focus_calls().is_empty());
    }

    #[test]
    fn adaptive_learns_from_mouse_use_and_can_be_switched() {
        let mut r = Rig::new(hold(38.0, 4.0, 200));
        r.d.handle_command(Command::Adaptive(AdaptiveCommand::On));
        // Mouse in use on DP-1 while the head sits at 38 (calibrated 43).
        r.hypr.cursor.set((300, 600));
        for i in 0..150 {
            r.hypr.cursor.set((300 + (i % 2) * 5, 600));
            r.d.tick().unwrap();
        }
        let s = r.d.status();
        assert!(s.adaptive);
        let drift = s.drift.iter().find(|(m, _)| m == "DP-1").unwrap().1;
        assert!(drift > 1.0, "{:?}", s.drift);
        // Switching off restores the calibrated centroid for decisions.
        r.d.handle_command(Command::Adaptive(AdaptiveCommand::Off));
        assert_eq!(r.d.switcher.classifier().centroids()[0].yaw, 43.0);
        r.d.handle_command(Command::Adaptive(AdaptiveCommand::Reset));
        assert!(r.d.status().drift.iter().all(|(_, d)| *d == 0.0));
        assert!(r.events().contains(&Event::AdaptiveChanged { enabled: true }));
    }

    #[test]
    fn idles_at_a_lower_rate_when_still() {
        let mut poses = hold(31.0, 2.0, 40);
        poses.extend(hold(23.0, 2.0, 10));
        let mut r = Rig::new(poses);
        r.run_all();
        let fps = r.script.borrow().fps.clone();
        // Full rate on open, idle once still, full again when the head turns.
        assert_eq!(fps.first(), Some(&15.0));
        assert!(fps.contains(&6.0), "{fps:?}");
        assert_eq!(fps.last(), Some(&15.0), "{fps:?}");
    }

    #[test]
    fn external_focus_changes_are_followed() {
        let mut r = Rig::new(hold(23.0, 2.0, 20));
        r.d.tick().unwrap();
        r.d.handle_hypr_event(HyprEvent::FocusedMonitor("eDP-2".into())).unwrap();
        r.run_all();
        assert!(r.focus_calls().is_empty());
    }

    #[test]
    fn layout_change_pauses_and_resumes() {
        let mut r = Rig::new(hold(23.0, 2.0, 40));
        r.hypr.monitors.borrow_mut().remove(0);
        r.d.handle_hypr_event(HyprEvent::MonitorRemoved("DP-1".into())).unwrap();
        assert_eq!(r.d.status().state, "layout_changed");
        assert!(r.d.status().reason.unwrap().contains("recalibrate"));
        for _ in 0..20 {
            assert_eq!(r.d.tick().unwrap(), Tick::Idle);
        }
        assert!(r.focus_calls().is_empty());
        *r.hypr.monitors.borrow_mut() = monitors();
        r.d.handle_hypr_event(HyprEvent::MonitorAdded("DP-1".into())).unwrap();
        r.run_all();
        assert_eq!(r.focus_calls().len(), 1);
        let events = r.events();
        assert!(events.iter().any(|e| matches!(e, Event::Paused { .. })));
        assert!(events.contains(&Event::Resumed));
    }

    #[test]
    fn a_failed_focus_is_retried() {
        let mut r = Rig::new(hold(23.0, 2.0, 30));
        r.hypr.fail.set(true);
        for _ in 0..8 {
            r.d.tick().unwrap();
        }
        assert_eq!(r.d.switcher.current(), Some(1));
        r.hypr.fail.set(false);
        r.run_all();
        assert_eq!(r.focus_calls().len(), 1);
    }

    #[test]
    fn reload_picks_up_a_new_calibration() {
        let dir = std::env::temp_dir().join(format!("lookfocus-reload-{}", std::process::id()));
        let path = dir.join("calibration.toml");
        let mut cal = calibration();
        cal.monitors[2].yaw = 10.0; // the right screen moved
        cal.created = "later".into();
        cal.save(&path).unwrap();
        let mut r = Rig::new(hold(12.0, 2.0, 30));
        r.d.calibration_path = Some(path);
        r.d.handle_command(Command::Reload);
        assert_eq!(r.d.switcher.classifier().centroids()[2].yaw, 10.0);
        r.run_all();
        assert_eq!(r.focus_calls().len(), 1);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn status_reports_the_basics() {
        let mut r = Rig::new(hold(23.0, 2.0, 30));
        r.run_all();
        let s = r.d.status();
        assert_eq!(s.state, "tracking");
        assert_eq!(s.monitor.as_deref(), Some("eDP-2"));
        assert_eq!(s.zone.as_deref(), Some("eDP-2"));
        assert!(s.face && s.camera && !s.adaptive);
        let json: serde_json::Value = serde_json::from_str(&r.d.handle_command(Command::Status)).unwrap();
        assert_eq!(json["state"], "tracking");
    }
}
