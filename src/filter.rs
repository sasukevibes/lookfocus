//! One Euro filter for smoothing yaw and pitch.
//!
//! The filter smooths hard when the signal is still and follows quickly when
//! it moves, which suits head pose: jitter disappears while holding still, and
//! a real head turn is not delayed much. See Casiez, Roussel and Vogel, "1 Euro
//! Filter", CHI 2012.
//!
//! The filter also exposes its smoothed derivative. The dwell logic uses that
//! to tell a settled head from one that is sweeping past a monitor. Unlike the
//! reference implementation, the derivative is taken between raw samples, not
//! from the smoothed value. The smoothed value lags during steady motion, which
//! would make the speed read far too high.

use std::f32::consts::PI;

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OneEuroParams {
    /// Cutoff frequency in Hz when the signal is still. Lower is smoother.
    pub min_cutoff: f32,
    /// How much the cutoff rises with speed. Higher follows fast moves better.
    pub beta: f32,
    /// Cutoff frequency in Hz for the derivative estimate.
    pub d_cutoff: f32,
}

impl Default for OneEuroParams {
    fn default() -> Self {
        // Angles in degrees at about 15 samples per second.
        Self { min_cutoff: 1.0, beta: 0.05, d_cutoff: 1.0 }
    }
}

#[derive(Clone, Debug)]
pub struct OneEuro {
    params: OneEuroParams,
    last: Option<State>,
}

#[derive(Clone, Copy, Debug)]
struct State {
    t: f32,
    raw: f32,
    smooth: f32,
    derivative: f32,
}

impl OneEuro {
    pub fn new(params: OneEuroParams) -> Self {
        Self { params, last: None }
    }

    /// Feeds one sample at time `t` (seconds, increasing) and returns the
    /// smoothed value.
    pub fn filter(&mut self, t: f32, x: f32) -> f32 {
        let Some(prev) = self.last else {
            self.last = Some(State { t, raw: x, smooth: x, derivative: 0.0 });
            return x;
        };
        let dt = t - prev.t;
        if dt <= 0.0 {
            return prev.smooth;
        }
        let dx = (x - prev.raw) / dt;
        let dx_hat = lerp(prev.derivative, dx, alpha(dt, self.params.d_cutoff));
        let cutoff = self.params.min_cutoff + self.params.beta * dx_hat.abs();
        let x_hat = lerp(prev.smooth, x, alpha(dt, cutoff));
        self.last = Some(State { t, raw: x, smooth: x_hat, derivative: dx_hat });
        x_hat
    }

    /// Smoothed rate of change in units per second, zero before two samples.
    pub fn derivative(&self) -> f32 {
        self.last.map_or(0.0, |s| s.derivative)
    }

    pub fn reset(&mut self) {
        self.last = None;
    }
}

fn alpha(dt: f32, cutoff: f32) -> f32 {
    let tau = 1.0 / (2.0 * PI * cutoff);
    1.0 / (1.0 + tau / dt)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

/// A smoothed (yaw, pitch) pair with a combined angular speed.
#[derive(Clone, Debug)]
pub struct PoseFilter {
    yaw: OneEuro,
    pitch: OneEuro,
}

impl PoseFilter {
    pub fn new(params: OneEuroParams) -> Self {
        Self { yaw: OneEuro::new(params), pitch: OneEuro::new(params) }
    }

    pub fn filter(&mut self, t: f32, yaw: f32, pitch: f32) -> (f32, f32) {
        (self.yaw.filter(t, yaw), self.pitch.filter(t, pitch))
    }

    /// Angular speed of the head in degrees per second.
    pub fn speed(&self) -> f32 {
        self.yaw.derivative().hypot(self.pitch.derivative())
    }

    pub fn reset(&mut self) {
        self.yaw.reset();
        self.pitch.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 1.0 / 15.0;

    #[test]
    fn first_sample_passes_through() {
        let mut f = OneEuro::new(OneEuroParams::default());
        assert_eq!(f.filter(0.0, 12.5), 12.5);
        assert_eq!(f.derivative(), 0.0);
    }

    #[test]
    fn reduces_jitter_while_still() {
        let mut f = OneEuro::new(OneEuroParams::default());
        // Alternating plus and minus 2 degrees around 30.
        let mut worst: f32 = 0.0;
        for i in 0..150 {
            let x = 30.0 + if i % 2 == 0 { 2.0 } else { -2.0 };
            let y = f.filter(i as f32 * DT, x);
            if i > 30 {
                worst = worst.max((y - 30.0).abs());
            }
        }
        assert!(worst < 1.0, "smoothed jitter {worst}");
    }

    #[test]
    fn follows_a_step_and_reports_speed() {
        let mut f = OneEuro::new(OneEuroParams::default());
        for i in 0..30 {
            f.filter(i as f32 * DT, 10.0);
        }
        assert!(f.derivative().abs() < 1e-3);
        let mut y = 0.0;
        let mut peak_speed: f32 = 0.0;
        for i in 30..60 {
            y = f.filter(i as f32 * DT, 40.0);
            peak_speed = peak_speed.max(f.derivative());
        }
        // Two seconds after a 30 degree step the output has caught up.
        assert!((y - 40.0).abs() < 1.0, "y = {y}");
        assert!(peak_speed > 50.0, "peak speed {peak_speed}");
        // And it settles back to still.
        for i in 60..120 {
            f.filter(i as f32 * DT, 40.0);
        }
        assert!(f.derivative().abs() < 1.0);
    }

    #[test]
    fn ignores_non_increasing_time() {
        let mut f = OneEuro::new(OneEuroParams::default());
        f.filter(1.0, 5.0);
        assert_eq!(f.filter(1.0, 50.0), 5.0);
        assert_eq!(f.filter(0.5, 50.0), 5.0);
    }

    #[test]
    fn pose_filter_speed_combines_axes() {
        let mut f = PoseFilter::new(OneEuroParams::default());
        for i in 0..20 {
            let t = i as f32 * DT;
            f.filter(t, 30.0 * t, 40.0 * t);
        }
        // Constant motion of 30 and 40 degrees per second is 50 combined.
        assert!((f.speed() - 50.0).abs() < 3.0, "speed {}", f.speed());
    }
}
