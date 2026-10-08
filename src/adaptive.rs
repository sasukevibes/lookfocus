//! Adaptive centroids: slowly moving each monitor's centroid toward how you
//! actually look at it.
//!
//! Calibration poses can differ from everyday ones (people tend to turn their
//! head further when asked to look at a screen). Mouse use gives free labels:
//! while you move the mouse on a monitor, you are almost always looking at it.
//! Each such sample nudges that monitor's centroid a small step toward your
//! current pose.
//!
//! Three guards keep learning from going wrong:
//!
//! - A centroid never drifts more than `max_drift` degrees from its
//!   calibrated position.
//! - Two centroids never get closer than `min_gap_fraction` of their
//!   calibrated distance, so monitors stay separable.
//! - A sample far from the monitor's calibrated pose is ignored. That is the
//!   case where the mouse is on one screen while you read another.

use std::time::{Duration, Instant};

use crate::classify::Centroid;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaptiveParams {
    /// Fraction of the way to move toward each sample.
    pub rate: f32,
    pub max_drift: f32,
    pub min_gap_fraction: f32,
    /// At most one update per monitor in this interval.
    pub interval: Duration,
}

impl Default for AdaptiveParams {
    fn default() -> Self {
        Self { rate: 0.03, max_drift: 6.0, min_gap_fraction: 0.6, interval: Duration::from_millis(250) }
    }
}

pub struct Adaptive {
    params: AdaptiveParams,
    base: Vec<Centroid>,
    current: Vec<Centroid>,
    last: Vec<Option<Instant>>,
}

fn dist(a: &Centroid, yaw: f32, pitch: f32) -> f32 {
    (a.yaw - yaw).hypot(a.pitch - pitch)
}

impl Adaptive {
    /// Starts from the calibrated centroids, or from previously learned ones
    /// if they match the same monitors.
    pub fn new(params: AdaptiveParams, base: Vec<Centroid>, learned: Option<Vec<Centroid>>) -> Self {
        let current = match learned {
            Some(l) if l.len() == base.len() && l.iter().zip(&base).all(|(a, b)| a.monitor == b.monitor) => l,
            _ => base.clone(),
        };
        let last = vec![None; base.len()];
        let mut a = Self { params, base, current, last };
        // Re-apply the guards in case the settings got stricter.
        for i in 0..a.current.len() {
            if !a.allowed(i, a.current[i].yaw, a.current[i].pitch) {
                a.current[i] = a.base[i].clone();
            }
        }
        a
    }

    pub fn centroids(&self) -> &[Centroid] {
        &self.current
    }

    pub fn base(&self) -> &[Centroid] {
        &self.base
    }

    pub fn reset(&mut self) {
        self.current = self.base.clone();
    }

    /// How far each centroid has moved from calibration, in degrees.
    pub fn drift(&self) -> Vec<f32> {
        self.current.iter().zip(&self.base).map(|(c, b)| dist(b, c.yaw, c.pitch)).collect()
    }

    fn allowed(&self, i: usize, yaw: f32, pitch: f32) -> bool {
        if dist(&self.base[i], yaw, pitch) > self.params.max_drift + 1e-4 {
            return false;
        }
        (0..self.current.len()).filter(|&j| j != i).all(|j| {
            let calibrated = dist(&self.base[i], self.base[j].yaw, self.base[j].pitch);
            dist(&self.current[j], yaw, pitch) >= self.params.min_gap_fraction * calibrated
        })
    }

    /// Learns from one labelled sample: you were looking at monitor `i` with
    /// this pose. Returns true if the centroid moved.
    pub fn learn(&mut self, now: Instant, i: usize, yaw: f32, pitch: f32) -> bool {
        if i >= self.current.len() {
            return false;
        }
        if self.last[i].is_some_and(|t| now.duration_since(t) < self.params.interval) {
            return false;
        }
        if dist(&self.base[i], yaw, pitch) > 2.0 * self.params.max_drift {
            return false;
        }
        self.last[i] = Some(now);
        let c = &self.current[i];
        let mut ny = c.yaw + self.params.rate * (yaw - c.yaw);
        let mut np = c.pitch + self.params.rate * (pitch - c.pitch);
        // Pull back inside the drift limit if needed.
        let b = &self.base[i];
        let d = dist(b, ny, np);
        if d > self.params.max_drift {
            let k = self.params.max_drift / d;
            ny = b.yaw + (ny - b.yaw) * k;
            np = b.pitch + (np - b.pitch) * k;
        }
        if !self.allowed(i, ny, np) {
            return false;
        }
        self.current[i].yaw = ny;
        self.current[i].pitch = np;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> Vec<Centroid> {
        vec![
            Centroid { monitor: "left".into(), yaw: 40.4, pitch: 0.2 },
            Centroid { monitor: "center".into(), yaw: 28.3, pitch: 0.7 },
            Centroid { monitor: "right".into(), yaw: 22.2, pitch: -3.6 },
        ]
    }

    fn feed(a: &mut Adaptive, i: usize, yaw: f32, pitch: f32, n: u32) -> Instant {
        let start = Instant::now();
        for k in 0..n {
            a.learn(start + Duration::from_millis(300) * k, i, yaw, pitch);
        }
        start
    }

    #[test]
    fn moves_toward_natural_poses() {
        let mut a = Adaptive::new(AdaptiveParams::default(), base(), None);
        // Naturally the left screen is looked at with less head turn.
        feed(&mut a, 0, 37.0, 0.5, 200);
        let c = &a.centroids()[0];
        assert!((c.yaw - 37.0).abs() < 0.3, "{c:?}");
        assert!(a.drift()[0] > 3.0);
        assert_eq!(a.drift()[1], 0.0);
    }

    #[test]
    fn respects_the_drift_limit() {
        let mut a = Adaptive::new(AdaptiveParams { max_drift: 2.0, ..Default::default() }, base(), None);
        feed(&mut a, 0, 44.0, 0.2, 500);
        assert!(a.drift()[0] <= 2.0 + 1e-3, "{:?}", a.drift());
    }

    #[test]
    fn keeps_monitors_apart() {
        let mut a = Adaptive::new(AdaptiveParams::default(), base(), None);
        // Center and right start 7.5 degrees apart. Pull right toward center.
        feed(&mut a, 2, 26.5, -1.0, 1000);
        let (c, r) = (&a.centroids()[1], &a.centroids()[2]);
        let gap = (c.yaw - r.yaw).hypot(c.pitch - r.pitch);
        assert!(gap >= 0.6 * 7.43 - 1e-3, "gap {gap}");
    }

    #[test]
    fn ignores_outliers_and_rate_limits() {
        let mut a = Adaptive::new(AdaptiveParams::default(), base(), None);
        let now = Instant::now();
        // Mouse on the right screen while clearly looking at the left one.
        assert!(!a.learn(now, 2, 40.0, 0.0));
        assert!(a.learn(now, 2, 23.0, -3.0));
        // Too soon for the same monitor, fine for another.
        assert!(!a.learn(now + Duration::from_millis(100), 2, 23.0, -3.0));
        assert!(a.learn(now + Duration::from_millis(100), 1, 28.0, 0.5));
        assert!(!a.learn(now, 9, 0.0, 0.0));
    }

    #[test]
    fn restores_learned_centroids_for_the_same_monitors() {
        let mut learned = base();
        learned[0].yaw = 38.0;
        let a = Adaptive::new(AdaptiveParams::default(), base(), Some(learned.clone()));
        assert_eq!(a.centroids()[0].yaw, 38.0);
        // A different monitor set is ignored.
        learned[1].monitor = "other".into();
        let b = Adaptive::new(AdaptiveParams::default(), base(), Some(learned));
        assert_eq!(b.centroids(), &base()[..]);
        // Learned values beyond the current limits are dropped.
        let mut far = base();
        far[0].yaw = 30.0;
        let c = Adaptive::new(AdaptiveParams::default(), base(), Some(far));
        assert_eq!(c.centroids()[0].yaw, 40.4);
    }

    #[test]
    fn reset_returns_to_calibration() {
        let mut a = Adaptive::new(AdaptiveParams::default(), base(), None);
        feed(&mut a, 0, 37.0, 0.5, 50);
        a.reset();
        assert_eq!(a.centroids(), &base()[..]);
    }
}
