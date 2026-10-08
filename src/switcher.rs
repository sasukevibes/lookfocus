//! Deciding when to switch monitors.
//!
//! A new monitor wins only when all three hold:
//!
//! 1. Hysteresis: it is closer than the current monitor by at least a margin.
//!    Leaving a monitor takes a bigger change than staying on it, so a pose
//!    near the boundary does not flip back and forth.
//! 2. Settled: the head is moving slower than the settle speed. While the head
//!    sweeps (say from the left monitor to the right one), the dwell timer
//!    keeps restarting, so the monitor it passes is never chosen.
//! 3. Dwell: conditions 1 and 2 have held for the dwell time.
//!
//! The switcher only decides. The daemon carries out the switch.

use std::time::{Duration, Instant};

use crate::classify::Classifier;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SwitchParams {
    pub dwell: Duration,
    /// Degrees by which the new monitor must be closer than the current one.
    pub hysteresis: f32,
    /// Degrees per second below which the head counts as settled.
    pub settle_speed: f32,
}

/// What the switcher concluded for one sample.
#[derive(Clone, Debug, PartialEq)]
pub enum Decision {
    /// Stay on the current monitor (or there is none yet and no clear choice).
    Stay,
    /// Another monitor is winning but has not held long enough.
    Pending { target: usize, held: Duration },
    /// Switch now.
    Switch { from: Option<usize>, to: usize },
}

pub struct Switcher {
    classifier: Classifier,
    params: SwitchParams,
    current: Option<usize>,
    candidate: Option<(usize, Instant)>,
}

impl Switcher {
    pub fn new(classifier: Classifier, params: SwitchParams) -> Self {
        Self { classifier, params, current: None, candidate: None }
    }

    pub fn classifier(&self) -> &Classifier {
        &self.classifier
    }

    pub fn classifier_mut(&mut self) -> &mut Classifier {
        &mut self.classifier
    }

    pub fn params(&self) -> SwitchParams {
        self.params
    }

    pub fn current(&self) -> Option<usize> {
        self.current
    }

    /// Records the focused monitor when it changes for reasons other than our
    /// own switch, for example a mouse move or a keybind.
    pub fn set_current(&mut self, index: Option<usize>) {
        if self.current != index {
            self.current = index;
            self.candidate = None;
        }
    }

    /// Drops any half-finished dwell, for example when the face is lost.
    pub fn hold(&mut self) {
        self.candidate = None;
    }

    pub fn update(&mut self, now: Instant, yaw: f32, pitch: f32, speed: f32) -> Decision {
        let Some(c) = self.classifier.classify(yaw, pitch) else {
            self.candidate = None;
            return Decision::Stay;
        };
        let target = match self.current {
            Some(cur) if c.best == cur => None,
            Some(cur) => {
                let advantage = self.classifier.distance(cur, yaw, pitch) - c.best_distance;
                (advantage >= self.params.hysteresis).then_some(c.best)
            }
            // Nothing focused that we know of: take the nearest once it has
            // the usual margin over the runner-up.
            None => (c.margin() >= self.params.hysteresis).then_some(c.best),
        };
        let Some(target) = target else {
            self.candidate = None;
            return Decision::Stay;
        };
        let since = match self.candidate {
            Some((t, since)) if t == target && speed <= self.params.settle_speed => since,
            // A new target, or the head is still moving: restart the timer.
            _ => now,
        };
        self.candidate = Some((target, since));
        let held = now.duration_since(since);
        if held >= self.params.dwell {
            let from = self.current;
            self.current = Some(target);
            self.candidate = None;
            Decision::Switch { from, to: target }
        } else {
            Decision::Pending { target, held }
        }
    }
}

/// The default hysteresis: a quarter of the smallest gap between calibrated
/// monitors, kept between 1 and 4 degrees.
pub fn auto_hysteresis(classifier: &Classifier) -> f32 {
    let c = classifier.centroids();
    let mut gap = f32::INFINITY;
    for i in 0..c.len() {
        for j in i + 1..c.len() {
            gap = gap.min((c[i].yaw - c[j].yaw).hypot(c[i].pitch - c[j].pitch));
        }
    }
    if gap.is_finite() { (gap / 4.0).clamp(1.0, 4.0) } else { 2.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classify::Centroid;

    const LEFT: usize = 0;
    const CENTER: usize = 1;
    const RIGHT: usize = 2;
    const FRAME: Duration = Duration::from_millis(66);

    /// The natural-turn centroids from the Phase 2 check on the dev machine.
    fn classifier() -> Classifier {
        Classifier::new(vec![
            Centroid { monitor: "left".into(), yaw: 43.0, pitch: 4.6 },
            Centroid { monitor: "center".into(), yaw: 31.2, pitch: 2.2 },
            Centroid { monitor: "right".into(), yaw: 22.9, pitch: 1.9 },
        ])
    }

    fn switcher() -> Switcher {
        let c = classifier();
        let hysteresis = auto_hysteresis(&c);
        let mut s =
            Switcher::new(c, SwitchParams { dwell: Duration::from_millis(300), hysteresis, settle_speed: 25.0 });
        s.set_current(Some(CENTER));
        s
    }

    /// Feeds poses one frame apart and returns every decision.
    fn run(s: &mut Switcher, start: Instant, poses: &[(f32, f32, f32)]) -> Vec<Decision> {
        poses.iter().enumerate().map(|(i, &(y, p, v))| s.update(start + FRAME * i as u32, y, p, v)).collect()
    }

    fn switches(d: &[Decision]) -> Vec<usize> {
        d.iter().filter_map(|d| if let Decision::Switch { to, .. } = d { Some(*to) } else { None }).collect()
    }

    #[test]
    fn auto_hysteresis_scales_with_the_tightest_gap() {
        // Center to right is 8.3 degrees apart, so a quarter is about 2.1.
        let h = auto_hysteresis(&classifier());
        assert!((h - 2.08).abs() < 0.05, "{h}");
        let wide = Classifier::new(vec![
            Centroid { monitor: "a".into(), yaw: 0.0, pitch: 0.0 },
            Centroid { monitor: "b".into(), yaw: 40.0, pitch: 0.0 },
        ]);
        assert_eq!(auto_hysteresis(&wide), 4.0);
        assert_eq!(auto_hysteresis(&Classifier::default()), 2.0);
    }

    #[test]
    fn switches_after_the_dwell_when_settled() {
        let mut s = switcher();
        let d = run(&mut s, Instant::now(), &[(23.0, 2.0, 5.0); 8]);
        // 300 ms at 66 ms per frame: the switch lands on the sixth sample.
        assert_eq!(switches(&d), vec![RIGHT]);
        assert!(matches!(d[5], Decision::Switch { from: Some(CENTER), to: RIGHT }), "{:?}", d);
        assert!(matches!(d[1], Decision::Pending { target: RIGHT, .. }));
        assert_eq!(s.current(), Some(RIGHT));
        // Once there, staying is quiet.
        assert!(matches!(d[7], Decision::Stay));
    }

    #[test]
    fn a_short_glance_does_not_switch() {
        let mut s = switcher();
        let mut poses = vec![(23.0, 2.0, 5.0); 3]; // 200 ms on the right
        poses.extend([(31.0, 2.0, 5.0); 6]); // back to center
        assert!(switches(&run(&mut s, Instant::now(), &poses)).is_empty());
        assert_eq!(s.current(), Some(CENTER));
    }

    #[test]
    fn hysteresis_holds_near_the_boundary() {
        let mut s = switcher();
        // The center/right midpoint is about yaw 27. Just past it, the right
        // monitor is closer, but not by the 2.1 degree margin.
        let d = run(&mut s, Instant::now(), &[(26.5, 2.0, 2.0); 20]);
        assert!(switches(&d).is_empty());
        // Clearly past it, the switch happens.
        let d = run(&mut s, Instant::now(), &[(25.0, 2.0, 2.0); 20]);
        assert_eq!(switches(&d), vec![RIGHT]);
        // And coming back needs the same margin the other way.
        let d = run(&mut s, Instant::now(), &[(27.5, 2.0, 2.0); 20]);
        assert!(switches(&d).is_empty());
    }

    #[test]
    fn sweeping_past_a_monitor_does_not_select_it() {
        let mut s = switcher();
        s.set_current(Some(LEFT));
        // A 0.5 s sweep from left to right through the center at 40 deg/s,
        // then settling on the right.
        let mut poses = Vec::new();
        for i in 0..8 {
            poses.push((43.0 - 2.6 * i as f32, 3.0, 40.0));
        }
        poses.extend([(23.0, 2.0, 4.0); 8]);
        let d = run(&mut s, Instant::now(), &poses);
        assert_eq!(switches(&d), vec![RIGHT], "{d:?}");
    }

    #[test]
    fn moving_resets_the_dwell() {
        let mut s = switcher();
        // On the right but still moving fast: never settles, never switches.
        let d = run(&mut s, Instant::now(), &[(23.0, 2.0, 60.0); 20]);
        assert!(switches(&d).is_empty());
    }

    #[test]
    fn external_focus_change_is_respected() {
        let mut s = switcher();
        run(&mut s, Instant::now(), &[(23.0, 2.0, 5.0); 3]); // pending right
        s.set_current(Some(RIGHT)); // the mouse moved there anyway
        let d = run(&mut s, Instant::now(), &[(23.0, 2.0, 5.0); 10]);
        assert!(switches(&d).is_empty());
    }

    #[test]
    fn hold_cancels_a_pending_switch() {
        let mut s = switcher();
        let start = Instant::now();
        run(&mut s, start, &[(23.0, 2.0, 5.0); 4]);
        s.hold();
        // The timer restarts, so three more frames are not enough.
        let later = start + FRAME * 4;
        let d = run(&mut s, later, &[(23.0, 2.0, 5.0); 3]);
        assert!(switches(&d).is_empty());
    }

    #[test]
    fn no_known_current_picks_a_clear_winner() {
        let mut s = switcher();
        s.set_current(None);
        let d = run(&mut s, Instant::now(), &[(43.0, 4.0, 2.0); 8]);
        assert_eq!(switches(&d), vec![LEFT]);
        assert!(matches!(d[5], Decision::Switch { from: None, to: LEFT }));
    }
}
