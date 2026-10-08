//! Nearest-centroid monitor classification on (yaw, pitch).
//!
//! Each monitor has a centroid learned during calibration. A pose belongs to
//! the monitor whose centroid is closest. There are no fixed left/right
//! thresholds, so any layout and camera position works as long as the
//! calibrated poses are far enough apart.
//!
//! This module only answers "which centroid is nearest, and by how much". The
//! hysteresis and dwell that decide when to actually switch live elsewhere.

use serde::{Deserialize, Serialize};

/// A calibrated pose for one monitor, in degrees.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Centroid {
    pub monitor: String,
    pub yaw: f32,
    pub pitch: f32,
}

/// The outcome of classifying one pose.
#[derive(Clone, Debug, PartialEq)]
pub struct Classification {
    /// Index into the classifier's centroids of the nearest monitor.
    pub best: usize,
    pub best_distance: f32,
    /// Distance to the runner-up, or infinity with only one monitor.
    pub second_distance: f32,
}

impl Classification {
    /// How much closer the winner is than the runner-up, in degrees. Small
    /// margins mean the pose sits between two monitors.
    pub fn margin(&self) -> f32 {
        self.second_distance - self.best_distance
    }
}

#[derive(Clone, Debug, Default)]
pub struct Classifier {
    centroids: Vec<Centroid>,
}

impl Classifier {
    pub fn new(centroids: Vec<Centroid>) -> Self {
        Self { centroids }
    }

    pub fn centroids(&self) -> &[Centroid] {
        &self.centroids
    }

    /// Replaces the centroids, for adaptive learning. Monitors keep their
    /// order, so indices stay valid.
    pub fn set_centroids(&mut self, centroids: Vec<Centroid>) {
        debug_assert_eq!(centroids.len(), self.centroids.len());
        self.centroids = centroids;
    }

    /// Distance from a pose to one centroid, in degrees.
    pub fn distance(&self, index: usize, yaw: f32, pitch: f32) -> f32 {
        let c = &self.centroids[index];
        (yaw - c.yaw).hypot(pitch - c.pitch)
    }

    /// Returns the nearest centroid, or `None` when nothing is calibrated.
    pub fn classify(&self, yaw: f32, pitch: f32) -> Option<Classification> {
        let mut best: Option<(usize, f32)> = None;
        let mut second = f32::INFINITY;
        for i in 0..self.centroids.len() {
            let d = self.distance(i, yaw, pitch);
            match best {
                Some((_, bd)) if d >= bd => second = second.min(d),
                Some((_, bd)) => {
                    second = bd;
                    best = Some((i, d));
                }
                None => best = Some((i, d)),
            }
        }
        best.map(|(best, best_distance)| Classification { best, best_distance, second_distance: second })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Centroids shaped like the Phase 1 recording: a row of three monitors
    /// with the camera on the rightmost one.
    fn row_of_three() -> Classifier {
        Classifier::new(vec![
            Centroid { monitor: "left".into(), yaw: 44.8, pitch: 6.3 },
            Centroid { monitor: "middle".into(), yaw: 27.3, pitch: 3.7 },
            Centroid { monitor: "right".into(), yaw: 10.2, pitch: 1.9 },
        ])
    }

    fn name(c: &Classifier, yaw: f32, pitch: f32) -> &str {
        &c.centroids()[c.classify(yaw, pitch).unwrap().best].monitor
    }

    #[test]
    fn picks_the_nearest_monitor() {
        let c = row_of_three();
        assert_eq!(name(&c, 46.0, 5.0), "left");
        assert_eq!(name(&c, 26.0, 4.0), "middle");
        assert_eq!(name(&c, 9.0, 2.0), "right");
        // Far outside the calibrated range still picks the closest edge.
        assert_eq!(name(&c, 90.0, 0.0), "left");
        assert_eq!(name(&c, -40.0, 0.0), "right");
    }

    #[test]
    fn works_for_a_vertical_stack() {
        // Two monitors stacked, told apart by pitch alone.
        let c = Classifier::new(vec![
            Centroid { monitor: "top".into(), yaw: 2.0, pitch: 18.0 },
            Centroid { monitor: "bottom".into(), yaw: 1.0, pitch: -6.0 },
        ]);
        assert_eq!(name(&c, 0.0, 12.0), "top");
        assert_eq!(name(&c, 3.0, -2.0), "bottom");
    }

    #[test]
    fn works_for_a_grid() {
        let mut v = Vec::new();
        for (row, pitch) in [("top", 15.0), ("bottom", -5.0)] {
            for (col, yaw) in [("left", 25.0), ("right", -5.0)] {
                v.push(Centroid { monitor: format!("{row}-{col}"), yaw, pitch });
            }
        }
        let c = Classifier::new(v);
        assert_eq!(name(&c, 22.0, 13.0), "top-left");
        assert_eq!(name(&c, -3.0, 16.0), "top-right");
        assert_eq!(name(&c, 27.0, -4.0), "bottom-left");
        assert_eq!(name(&c, -8.0, -7.0), "bottom-right");
    }

    #[test]
    fn margin_is_small_between_monitors() {
        let c = row_of_three();
        let near_middle = c.classify(27.0, 3.7).unwrap();
        let between = c.classify(18.75, 2.8).unwrap();
        assert!(near_middle.margin() > 15.0, "{near_middle:?}");
        assert!(between.margin() < 1.0, "{between:?}");
    }

    #[test]
    fn single_and_empty() {
        let one = Classifier::new(vec![Centroid { monitor: "only".into(), yaw: 0.0, pitch: 0.0 }]);
        let r = one.classify(10.0, 0.0).unwrap();
        assert_eq!(r.best, 0);
        assert!(r.second_distance.is_infinite());
        assert!(Classifier::default().classify(0.0, 0.0).is_none());
    }

    #[test]
    fn noisy_samples_classify_correctly() {
        // Gaussian-ish noise with the spread seen in Phase 1 (about 2 degrees)
        // should essentially never cross a 17 degree gap.
        let c = row_of_three();
        let mut seed = 7u32;
        let mut noise = || {
            let mut s = 0.0;
            for _ in 0..6 {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                s += (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
            }
            s * 2.0 * 2.0
        };
        let mut wrong = 0;
        for (i, cent) in c.centroids().iter().enumerate() {
            for _ in 0..500 {
                if c.classify(cent.yaw + noise(), cent.pitch + noise()).unwrap().best != i {
                    wrong += 1;
                }
            }
        }
        assert_eq!(wrong, 0);
    }
}
