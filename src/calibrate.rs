//! Guided calibration.
//!
//! For each monitor, in a few rounds with alternating order, the user glances
//! at it naturally while about two seconds of poses are recorded. A final step
//! records looking down at the keyboard. The result is one centroid per
//! monitor plus quality warnings with suggested fixes.
//!
//! Recording follows the timestamps of the samples rather than the wall
//! clock, so tests can feed scripted poses without waiting.

use std::fmt;
use std::time::Duration;

use anyhow::{Result, bail};

use crate::config::{CalibratedMonitor, Calibration, LayoutEntry, LookDown};
use crate::hypr::Monitor;
use crate::sampler::PoseSource;

/// Poses beyond this yaw (from facing the camera) make landmarks less reliable.
pub const EXTREME_YAW: f32 = 45.0;
/// Separation ratio below which two monitors are likely to be confused.
pub const MIN_RATIO: f32 = 2.0;
/// Face brightness (0 to 255) below which lighting is called poor.
pub const DIM_LUMA: f32 = 60.0;
/// Looking down must be at least this many degrees below the lowest monitor.
pub const LOOK_DOWN_GAP: f32 = 6.0;

#[derive(Clone, Debug)]
pub struct Plan {
    /// Monitors in the order to visit them in the first round.
    pub monitors: Vec<Monitor>,
    pub rounds: usize,
    /// Time to turn toward the target before recording.
    pub lead: Duration,
    pub record: Duration,
    pub look_down: bool,
}

impl Plan {
    pub fn new(mut monitors: Vec<Monitor>, rounds: usize) -> Self {
        monitors.sort_by_key(|m| (m.x, m.y));
        Self { monitors, rounds, lead: Duration::from_secs(3), record: Duration::from_secs(2), look_down: true }
    }
}

/// How calibration talks to the user. The CLI prints and sends desktop
/// notifications; tests record the messages.
pub trait Prompter {
    /// A new instruction, like "Look at the LEFT screen".
    fn instruct(&mut self, text: &str);
    /// A status line, like "60/60 frames had a face".
    fn report(&mut self, text: &str);
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Target {
    Monitor(usize),
    LookDown,
}

#[derive(Clone, Debug)]
pub struct PoseRecord {
    pub yaw: f32,
    pub pitch: f32,
    pub luma: f32,
}

#[derive(Clone, Debug)]
pub struct Visit {
    pub target: Target,
    pub round: usize,
    /// One entry per frame. `None` means no face.
    pub poses: Vec<Option<PoseRecord>>,
}

/// Plain position words for each monitor, so prompts say where to look.
/// Expects monitors sorted left to right.
pub fn position_labels(monitors: &[Monitor]) -> Vec<String> {
    let n = monitors.len();
    if n == 1 {
        return vec!["only screen".into()];
    }
    let row = monitors.iter().all(|m| m.y == monitors[0].y);
    let column = monitors.iter().all(|m| m.x == monitors[0].x);
    let line = |first: &str, middle: &str, last: &str, order: &[usize]| -> Vec<String> {
        let mut out = vec![String::new(); n];
        for (rank, &i) in order.iter().enumerate() {
            out[i] = match rank {
                0 => first.to_string(),
                r if r == n - 1 => last.to_string(),
                _ if n == 3 => middle.to_string(),
                r => format!("{middle} {r}"),
            };
        }
        out
    };
    let mut order: Vec<usize> = (0..n).collect();
    if row {
        order.sort_by_key(|&i| monitors[i].x);
        line("LEFT", "CENTER", "RIGHT", &order)
    } else if column {
        order.sort_by_key(|&i| monitors[i].y);
        line("TOP", "MIDDLE", "BOTTOM", &order)
    } else {
        monitors.iter().map(|m| format!("screen at {},{}", m.x, m.y)).collect()
    }
}

/// The monitor the camera is probably on: a built-in laptop panel if there is
/// one. Otherwise there is no good guess, so the first monitor.
pub fn guess_camera_monitor(monitors: &[Monitor]) -> usize {
    monitors.iter().position(|m| ["eDP", "LVDS", "DSI"].iter().any(|p| m.name.starts_with(p))).unwrap_or(0)
}

/// Runs the guided recording. Returns one visit per step.
pub fn record(source: &mut impl PoseSource, plan: &Plan, ui: &mut impl Prompter) -> Result<Vec<Visit>> {
    let labels = position_labels(&plan.monitors);
    let mut steps = Vec::new();
    for round in 0..plan.rounds {
        let mut order: Vec<usize> = (0..plan.monitors.len()).collect();
        if round % 2 == 1 {
            order.reverse();
        }
        steps.extend(order.into_iter().map(|i| (Target::Monitor(i), round)));
    }
    if plan.look_down {
        steps.push((Target::LookDown, 0));
    }

    // Let the camera's auto exposure settle first.
    wait(source, Duration::from_secs(3))?;

    let mut visits = Vec::new();
    for (target, round) in steps {
        let text = match target {
            Target::Monitor(i) => format!("Look at the {} screen ({})", labels[i], plan.monitors[i].name),
            Target::LookDown => "Look down at your keyboard".to_string(),
        };
        ui.instruct(&text);
        wait(source, plan.lead)?;
        let mut poses = Vec::new();
        let start = source.sample()?;
        let begin = start.time;
        let mut push = |s: crate::sampler::Sample| {
            poses.push(s.face.map(|f| PoseRecord { yaw: f.pose.yaw, pitch: f.pose.pitch, luma: f.luma }))
        };
        push(start);
        loop {
            let s = source.sample()?;
            let done = s.time.duration_since(begin) >= plan.record;
            push(s);
            if done {
                break;
            }
        }
        let seen = poses.iter().filter(|p| p.is_some()).count();
        ui.report(&format!("{seen}/{} frames had a face", poses.len()));
        visits.push(Visit { target, round, poses });
    }
    ui.instruct("Done. You can look anywhere now.");
    Ok(visits)
}

fn wait(source: &mut impl PoseSource, d: Duration) -> Result<()> {
    let start = source.sample()?.time;
    while source.sample()?.time.duration_since(start) < d {}
    Ok(())
}

/// A quality problem found in a calibration, with advice.
#[derive(Clone, Debug, PartialEq)]
pub enum Warning {
    TooClose { a: String, b: String, distance: f32, ratio: f32 },
    FaceLost { monitor: String, rate: f32 },
    Jittery { monitor: String, spread: f32 },
    Inconsistent { monitor: String, spread: f32 },
    ExtremeYaw { monitor: String, yaw: f32 },
    Dim { luma: f32 },
    LookDownUnclear { gap: f32 },
    CameraHint { hinted: String, nearest: String },
}

impl fmt::Display for Warning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Warning::TooClose { a, b, distance, ratio } => write!(
                f,
                "{a} and {b} are only {distance:.1} degrees apart (separation {ratio:.1}, under {MIN_RATIO}), so they \
                 may be confused. Turn your head a little more toward each when calibrating, or move the camera \
                 closer to the middle of your screens, then recalibrate. Eye gaze could help here in a future \
                 version."
            ),
            Warning::FaceLost { monitor, rate } => write!(
                f,
                "Your face was found in only {:.0}% of frames while looking at {monitor}. Check that the camera \
                 can see your whole face from that angle.",
                rate * 100.0
            ),
            Warning::Jittery { monitor, spread } => write!(
                f,
                "The pose for {monitor} was jittery ({spread:.1} degrees). Hold your head still during each \
                 recording, and check the lighting."
            ),
            Warning::Inconsistent { monitor, spread } => write!(
                f,
                "Your visits to {monitor} landed up to {spread:.1} degrees apart. Look at it the way you normally \
                 would each time, then recalibrate."
            ),
            Warning::ExtremeYaw { monitor, yaw } => write!(
                f,
                "Looking at {monitor} turns your head {:.0} degrees from the camera. Beyond about {EXTREME_YAW:.0} \
                 degrees face landmarks get less reliable. Moving the camera toward the middle of your screens \
                 would help.",
                yaw.abs()
            ),
            Warning::Dim { luma } => write!(
                f,
                "Lighting is poor (face brightness {luma:.0} of 255). More light on your face will make tracking \
                 steadier."
            ),
            Warning::LookDownUnclear { gap } => write!(
                f,
                "Looking down was only {gap:.1} degrees below your lowest screen, so it cannot be told apart \
                 reliably. Look-down will be detected only when your face is lost or you are typing."
            ),
            Warning::CameraHint { hinted, nearest } => write!(
                f,
                "You said the camera is on {hinted}, but you faced the camera most squarely when looking at \
                 {nearest}. This is only a note: the recorded poses are what lookfocus uses."
            ),
        }
    }
}

fn median(v: &mut [f32]) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    v.sort_by(f32::total_cmp);
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

/// Robust standard deviation (1.4826 times the median absolute deviation).
fn robust_spread(v: &[f32]) -> f32 {
    let mut c = v.to_vec();
    let m = median(&mut c);
    let mut dev: Vec<f32> = v.iter().map(|x| (x - m).abs()).collect();
    1.4826 * median(&mut dev)
}

/// Turns recorded visits into a calibration plus quality warnings.
pub fn summarize(
    plan: &Plan,
    visits: &[Visit],
    camera_hint: Option<usize>,
    created: String,
) -> Result<(Calibration, Vec<Warning>)> {
    let mut warnings = Vec::new();
    let mut monitors = Vec::new();
    let mut lumas = Vec::new();

    for (i, mon) in plan.monitors.iter().enumerate() {
        let mine: Vec<&Visit> = visits.iter().filter(|v| v.target == Target::Monitor(i)).collect();
        let frames: usize = mine.iter().map(|v| v.poses.len()).sum();
        let poses: Vec<&PoseRecord> = mine.iter().flat_map(|v| v.poses.iter().flatten()).collect();
        if poses.len() < 5 {
            bail!("no usable face poses while looking at {}. Check the camera can see you, then try again.", mon.name);
        }
        let mut yaws: Vec<f32> = poses.iter().map(|p| p.yaw).collect();
        let mut pitches: Vec<f32> = poses.iter().map(|p| p.pitch).collect();
        lumas.extend(poses.iter().map(|p| p.luma));
        let (yaw, pitch) = (median(&mut yaws), median(&mut pitches));

        let mut holds = Vec::new();
        let mut visit_spread: f32 = 0.0;
        for v in &mine {
            let vy: Vec<f32> = v.poses.iter().flatten().map(|p| p.yaw).collect();
            let vp: Vec<f32> = v.poses.iter().flatten().map(|p| p.pitch).collect();
            if vy.len() < 3 {
                continue;
            }
            holds.push(robust_spread(&vy).hypot(robust_spread(&vp)));
            let (my, mp) = (median(&mut vy.clone()), median(&mut vp.clone()));
            visit_spread = visit_spread.max((my - yaw).hypot(mp - pitch));
        }
        let hold_spread = median(&mut holds);
        let face_rate = poses.len() as f32 / frames.max(1) as f32;

        if face_rate < 0.8 {
            warnings.push(Warning::FaceLost { monitor: mon.name.clone(), rate: face_rate });
        }
        if hold_spread > 3.0 {
            warnings.push(Warning::Jittery { monitor: mon.name.clone(), spread: hold_spread });
        }
        if visit_spread > 5.0 {
            warnings.push(Warning::Inconsistent { monitor: mon.name.clone(), spread: visit_spread });
        }
        if yaw.abs() > EXTREME_YAW {
            warnings.push(Warning::ExtremeYaw { monitor: mon.name.clone(), yaw });
        }
        monitors.push(CalibratedMonitor {
            name: mon.name.clone(),
            yaw,
            pitch,
            hold_spread,
            visit_spread,
            face_rate,
            samples: poses.len(),
        });
    }

    for i in 0..monitors.len() {
        for j in i + 1..monitors.len() {
            let (a, b) = (&monitors[i], &monitors[j]);
            let distance = (a.yaw - b.yaw).hypot(a.pitch - b.pitch);
            let spread = a.hold_spread + a.visit_spread + b.hold_spread + b.visit_spread;
            let ratio = if spread > 0.0 { distance / spread } else { f32::INFINITY };
            if ratio < MIN_RATIO {
                warnings.push(Warning::TooClose { a: a.name.clone(), b: b.name.clone(), distance, ratio });
            }
        }
    }

    let luma = median(&mut lumas);
    if luma < DIM_LUMA {
        warnings.push(Warning::Dim { luma });
    }

    let look_down = visits.iter().find(|v| v.target == Target::LookDown).and_then(|v| {
        let mut p: Vec<f32> = v.poses.iter().flatten().map(|p| p.pitch).collect();
        if p.len() < 5 {
            return None;
        }
        let pitch = median(&mut p);
        let lowest = monitors.iter().map(|m| m.pitch).fold(f32::INFINITY, f32::min);
        let gap = lowest - pitch;
        if gap < LOOK_DOWN_GAP {
            warnings.push(Warning::LookDownUnclear { gap });
            None
        } else {
            Some(LookDown { pitch, threshold: (pitch + lowest) / 2.0 })
        }
    });

    // The camera hint never changes the result. It only gets a note if the
    // poses disagree with it.
    let nearest = monitors.iter().enumerate().min_by(|a, b| a.1.yaw.abs().total_cmp(&b.1.yaw.abs())).map(|(i, _)| i);
    if let (Some(h), Some(n)) = (camera_hint, nearest)
        && h != n
        && monitors.len() > 1
    {
        warnings.push(Warning::CameraHint { hinted: monitors[h].name.clone(), nearest: monitors[n].name.clone() });
    }

    let calibration = Calibration {
        version: Calibration::VERSION,
        created,
        camera_monitor: camera_hint.map(|h| plan.monitors[h].name.clone()),
        layout: plan.monitors.iter().map(LayoutEntry::from).collect(),
        monitors,
        look_down,
    };
    Ok((calibration, warnings))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pose::HeadPose;
    use crate::sampler::{FaceSample, Sample};
    use std::time::Instant;

    fn mon(name: &str, x: i32, y: i32) -> Monitor {
        Monitor { name: name.into(), description: String::new(), x, y, width: 1920, height: 1080, focused: false }
    }

    #[test]
    fn labels_for_common_layouts() {
        let row = [mon("a", 0, 0), mon("b", 1920, 0), mon("c", 3840, 0)];
        assert_eq!(position_labels(&row), ["LEFT", "CENTER", "RIGHT"]);
        assert_eq!(position_labels(&row[..2]), ["LEFT", "RIGHT"]);
        let four = [mon("a", 0, 0), mon("b", 1, 0), mon("c", 2, 0), mon("d", 3, 0)];
        assert_eq!(position_labels(&four), ["LEFT", "CENTER 1", "CENTER 2", "RIGHT"]);
        let stack = [mon("top", 0, 0), mon("bottom", 0, 1080)];
        assert_eq!(position_labels(&stack), ["TOP", "BOTTOM"]);
        let grid = [mon("a", 0, 0), mon("b", 0, 1080), mon("c", 1920, 0)];
        assert_eq!(position_labels(&grid)[2], "screen at 1920,0");
        assert_eq!(position_labels(&row[..1]), ["only screen"]);
    }

    #[test]
    fn guesses_the_laptop_panel() {
        assert_eq!(guess_camera_monitor(&[mon("DP-1", 0, 0), mon("eDP-1", 1920, 0)]), 1);
        assert_eq!(guess_camera_monitor(&[mon("DP-1", 0, 0), mon("HDMI-A-1", 1920, 0)]), 0);
    }

    /// Plays back a pose per target, switching whenever the recorder moves on.
    /// Time advances 66 ms per sample, so nothing waits on the clock.
    struct Scripted {
        t: Instant,
        pose: Box<dyn FnMut(usize) -> Option<(f32, f32)>>,
        n: usize,
    }

    impl PoseSource for Scripted {
        fn sample(&mut self) -> Result<Sample> {
            self.t += Duration::from_millis(66);
            self.n += 1;
            let face = (self.pose)(self.n).map(|(yaw, pitch)| FaceSample {
                pose: HeadPose { yaw, pitch, roll: 0.0 },
                presence: 1.0,
                tracked: true,
                luma: 120.0,
            });
            Ok(Sample { time: self.t, face, capture: Duration::ZERO, inference: Duration::ZERO })
        }
    }

    #[derive(Default)]
    struct Log(Vec<String>);

    impl Prompter for Log {
        fn instruct(&mut self, text: &str) {
            self.0.push(text.to_string());
        }
        fn report(&mut self, _: &str) {}
    }

    fn noise(n: usize) -> f32 {
        ((n * 7919) % 13) as f32 / 13.0 - 0.5
    }

    #[test]
    fn records_every_step_in_order() {
        let plan = Plan::new(vec![mon("DP-2", 768, 0), mon("DP-1", 0, 0), mon("eDP-2", 2688, 0)], 2);
        let mut src = Scripted { t: Instant::now(), pose: Box::new(|_| Some((30.0, 0.0))), n: 0 };
        let mut log = Log::default();
        let visits = record(&mut src, &plan, &mut log).unwrap();
        assert_eq!(visits.len(), 7);
        assert_eq!(log.0[0], "Look at the LEFT screen (DP-1)");
        assert_eq!(log.0[2], "Look at the RIGHT screen (eDP-2)");
        assert_eq!(log.0[3], "Look at the RIGHT screen (eDP-2)", "second round runs in reverse");
        assert_eq!(log.0[6], "Look down at your keyboard");
        assert_eq!(log.0[7], "Done. You can look anywhere now.");
        // About two seconds of frames per visit.
        assert!(
            visits.iter().all(|v| (30..=33).contains(&v.poses.len())),
            "{:?}",
            visits.iter().map(|v| v.poses.len()).collect::<Vec<_>>()
        );
    }

    /// Builds visits directly from per-target poses with a little noise.
    fn visits(targets: &[(Target, f32, f32)], rounds: usize, drop_face: impl Fn(usize) -> bool) -> Vec<Visit> {
        let mut out = Vec::new();
        let mut n = 0;
        for round in 0..rounds {
            for &(target, yaw, pitch) in targets {
                if target == Target::LookDown && round > 0 {
                    continue;
                }
                let poses = (0..30)
                    .map(|_| {
                        n += 1;
                        (!drop_face(n)).then(|| PoseRecord {
                            yaw: yaw + noise(n) * 2.0,
                            pitch: pitch + noise(n + 3),
                            luma: 110.0,
                        })
                    })
                    .collect();
                out.push(Visit { target, round, poses });
            }
        }
        out
    }

    fn row_plan() -> Plan {
        Plan::new(vec![mon("DP-1", 0, 0), mon("DP-2", 768, 0), mon("eDP-2", 2688, 0)], 2)
    }

    #[test]
    fn a_clean_calibration_has_no_warnings() {
        let plan = row_plan();
        let v = visits(
            &[
                (Target::Monitor(0), 43.0, 4.6),
                (Target::Monitor(1), 31.2, 2.2),
                (Target::Monitor(2), 10.0, 1.9),
                (Target::LookDown, 0.0, -20.0),
            ],
            2,
            |_| false,
        );
        let (cal, warnings) = summarize(&plan, &v, Some(2), "now".into()).unwrap();
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(cal.monitors.len(), 3);
        assert!((cal.monitors[0].yaw - 43.0).abs() < 1.0);
        assert_eq!(cal.camera_monitor.as_deref(), Some("eDP-2"));
        let ld = cal.look_down.unwrap();
        assert!((ld.pitch + 20.0).abs() < 1.0);
        // Halfway between looking down and the lowest monitor (pitch about 1.9).
        assert!((ld.threshold + 9.0).abs() < 1.0, "{ld:?}");
        assert_eq!(cal.layout.len(), 3);
    }

    #[test]
    fn warns_when_monitors_are_too_close() {
        let plan = row_plan();
        let v = visits(
            &[(Target::Monitor(0), 30.0, 2.0), (Target::Monitor(1), 27.0, 2.0), (Target::Monitor(2), 10.0, 2.0)],
            2,
            |_| false,
        );
        let (_, warnings) = summarize(&plan, &v, None, "now".into()).unwrap();
        assert!(
            warnings.iter().any(|w| matches!(w, Warning::TooClose { a, b, .. } if a == "DP-1" && b == "DP-2")),
            "{warnings:?}"
        );
        assert!(warnings.iter().all(|w| !matches!(w, Warning::TooClose { b, .. } if b == "eDP-2")));
    }

    #[test]
    fn warns_about_extreme_yaw_lost_faces_dim_light_and_unclear_look_down() {
        let plan = row_plan();
        let mut v = visits(
            &[
                (Target::Monitor(0), 60.0, 2.0),
                (Target::Monitor(1), 30.0, 2.0),
                (Target::Monitor(2), 5.0, 2.0),
                (Target::LookDown, 0.0, -1.0),
            ],
            2,
            |n| n <= 30 && n % 3 == 0, // lose a third of the first visit
        );
        for visit in &mut v {
            for p in visit.poses.iter_mut().flatten() {
                p.luma = 40.0;
            }
        }
        let (cal, warnings) = summarize(&plan, &v, Some(0), "now".into()).unwrap();
        let has = |f: &dyn Fn(&Warning) -> bool| warnings.iter().any(f);
        assert!(has(&|w| matches!(w, Warning::ExtremeYaw { monitor, .. } if monitor == "DP-1")));
        // Losing a third of one of two visits stays above the 80% threshold.
        assert!(!has(&|w| matches!(w, Warning::FaceLost { .. })));
        assert!(has(&|w| matches!(w, Warning::Dim { .. })));
        assert!(has(&|w| matches!(w, Warning::LookDownUnclear { .. })));
        assert!(has(
            &|w| matches!(w, Warning::CameraHint { hinted, nearest } if hinted == "DP-1" && nearest == "eDP-2")
        ));
        assert!(cal.look_down.is_none());
        // Every warning has a readable message.
        assert!(warnings.iter().all(|w| w.to_string().len() > 40));
    }

    #[test]
    fn warns_when_the_face_is_often_lost() {
        let plan = row_plan();
        let v = visits(
            &[(Target::Monitor(0), 43.0, 2.0), (Target::Monitor(1), 30.0, 2.0), (Target::Monitor(2), 10.0, 2.0)],
            2,
            |n| n % 2 == 0, // every other frame, everywhere
        );
        let (_, warnings) = summarize(&plan, &v, None, "now".into()).unwrap();
        assert_eq!(warnings.iter().filter(|w| matches!(w, Warning::FaceLost { .. })).count(), 3, "{warnings:?}");
    }

    #[test]
    fn warns_about_inconsistent_visits() {
        let plan = row_plan();
        let mut v = visits(
            &[(Target::Monitor(0), 43.0, 2.0), (Target::Monitor(1), 30.0, 2.0), (Target::Monitor(2), 10.0, 2.0)],
            2,
            |_| false,
        );
        // Shift the second visit to DP-2 by 12 degrees.
        for visit in v.iter_mut().filter(|v| v.target == Target::Monitor(1) && v.round == 1) {
            for p in visit.poses.iter_mut().flatten() {
                p.yaw += 12.0;
            }
        }
        let (_, warnings) = summarize(&plan, &v, None, "now".into()).unwrap();
        assert!(
            warnings.iter().any(|w| matches!(w, Warning::Inconsistent { monitor, .. } if monitor == "DP-2")),
            "{warnings:?}"
        );
    }

    #[test]
    fn fails_without_a_face() {
        let plan = row_plan();
        let mut v = visits(
            &[(Target::Monitor(0), 43.0, 2.0), (Target::Monitor(1), 30.0, 2.0), (Target::Monitor(2), 10.0, 2.0)],
            2,
            |_| false,
        );
        for visit in v.iter_mut().filter(|v| v.target == Target::Monitor(1)) {
            visit.poses.iter_mut().for_each(|p| *p = None);
        }
        let err = summarize(&plan, &v, None, "now".into()).unwrap_err();
        assert!(err.to_string().contains("DP-2"), "{err}");
    }
}
