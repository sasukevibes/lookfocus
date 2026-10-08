//! Camera frame in, head pose sample out.
//!
//! This joins the camera, the face tracker and the pose fit into one step so
//! the CLI commands and the daemon all read poses the same way.

use std::time::{Duration, Instant};

use anyhow::Result;

use crate::camera::FrameSource;
use crate::pose::{self, HeadPose};
use crate::vision::{FaceDetector, FaceMesh, FaceTracker};

/// What one frame produced.
#[derive(Clone, Debug)]
pub struct Sample {
    pub time: Instant,
    pub face: Option<FaceSample>,
    /// Time spent waiting for and converting the frame.
    pub capture: Duration,
    /// Time spent in the models and the pose fit.
    pub inference: Duration,
}

#[derive(Clone, Debug)]
pub struct FaceSample {
    pub pose: HeadPose,
    pub presence: f32,
    /// False when the detector had to run this frame.
    pub tracked: bool,
    /// Mean brightness of the face region, 0 to 255.
    pub luma: f32,
}

/// Anything that produces pose samples: the live sampler, or scripted poses
/// in tests.
pub trait PoseSource {
    fn sample(&mut self) -> Result<Sample>;

    /// Changes the target sample rate, where the source supports it.
    fn set_fps(&mut self, _fps: f32) {}
}

pub struct Sampler<C, D, M> {
    camera: C,
    tracker: FaceTracker<D, M>,
}

impl<C: FrameSource, D: FaceDetector, M: FaceMesh> Sampler<C, D, M> {
    pub fn new(camera: C, tracker: FaceTracker<D, M>) -> Self {
        Self { camera, tracker }
    }

    /// Forgets the tracked face, for example after a pause.
    pub fn reset(&mut self) {
        self.tracker.reset();
    }
}

impl<C: FrameSource, D: FaceDetector, M: FaceMesh> PoseSource for Sampler<C, D, M> {
    fn set_fps(&mut self, fps: f32) {
        self.camera.set_fps(fps);
    }

    fn sample(&mut self) -> Result<Sample> {
        let t0 = Instant::now();
        let frame = self.camera.next_frame()?;
        let t1 = Instant::now();
        let face = self.tracker.process(&frame.image)?.and_then(|face| {
            let pose = pose::estimate(&face.landmarks)?;
            let r = &face.roi;
            let half = r.width / 3.0; // The ROI is 1.5x the face, so this is about the face itself.
            let luma = frame.image.mean_luma(r.cx - half, r.cy - half, r.cx + half, r.cy + half);
            Some(FaceSample { pose, presence: face.presence, tracked: face.tracked, luma })
        });
        Ok(Sample { time: frame.time, face, capture: t1 - t0, inference: t1.elapsed() })
    }
}
