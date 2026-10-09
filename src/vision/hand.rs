//! Hand detection and landmark tracking.
//!
//! This works like face tracking (see `mod.rs`): a detector finds a palm, the
//! landmark model reads 21 points from a crop around it, and those points give
//! the crop for the next frame. Two things differ:
//!
//! - The palm detector costs more than BlazeFace, and most of the time there
//!   is no hand in view. So while no hand is tracked it runs at most once
//!   every `detect_every` frames instead of on every frame.
//! - Only one hand is tracked: the strongest palm.
//!
//! The models sit behind traits, as for faces, so the logic can be tested with
//! fakes.

use std::f32::consts::FRAC_PI_2;

use anyhow::Result;

use super::detector::sigmoid;
use super::palm::{PalmDetection, roi_from_palm};
use super::project;
use super::roi::{Roi, rotation};
use crate::image::RgbImage;

/// Side length of the hand landmark model's square input.
pub const HAND_INPUT_SIZE: usize = 224;
/// Number of landmarks the hand model produces.
pub const NUM_HAND_LANDMARKS: usize = 21;

/// Landmark indices, in MediaPipe's order. Each finger has four points from
/// the palm out: the knuckle (MCP), the middle joint (PIP), the last joint
/// (DIP) and the tip. The thumb's are named CMC, MCP, IP and tip.
pub const WRIST: usize = 0;
pub const THUMB_CMC: usize = 1;
pub const THUMB_MCP: usize = 2;
pub const THUMB_IP: usize = 3;
pub const THUMB_TIP: usize = 4;
pub const INDEX_MCP: usize = 5;
pub const INDEX_PIP: usize = 6;
pub const INDEX_TIP: usize = 8;
pub const MIDDLE_MCP: usize = 9;
pub const MIDDLE_PIP: usize = 10;
pub const MIDDLE_TIP: usize = 12;
pub const RING_MCP: usize = 13;
pub const RING_PIP: usize = 14;
pub const RING_TIP: usize = 16;
pub const PINKY_MCP: usize = 17;
pub const PINKY_PIP: usize = 18;
pub const PINKY_TIP: usize = 20;

/// Landmarks that move little from frame to frame: the wrist, the thumb's
/// base joints and the two lower joints of each finger. Fingertips are left
/// out so a curling finger does not shrink the next crop. MediaPipe uses the
/// same set.
const STABLE_LANDMARKS: [usize; 12] = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18];

/// The next frame's crop is the stable landmarks' box, moved a tenth of its
/// size toward the fingers and doubled.
const ROI_SHIFT_Y: f32 = -0.1;
const ROI_SCALE: f32 = 2.0;

pub trait PalmDetector {
    /// Palms in normalized frame coordinates, strongest first.
    fn detect(&mut self, frame: &RgbImage) -> Result<Vec<PalmDetection>>;
}

/// Raw landmark model output for one crop.
#[derive(Clone, Debug, PartialEq)]
pub struct HandOutput {
    /// NUM_HAND_LANDMARKS points as (x, y, z) in crop pixels (0 to
    /// HAND_INPUT_SIZE).
    pub points: Vec<[f32; 3]>,
    /// Probability (0 to 1) that the crop really contains a hand.
    pub confidence: f32,
    /// The model's handedness score, 0 to 1, as it reports it. Not yet
    /// checked against a live frame, so nothing relies on which side it means.
    pub handedness: f32,
}

pub trait HandLandmarker {
    fn run(&mut self, frame: &RgbImage, roi: &Roi) -> Result<HandOutput>;
}

/// A tracked hand for one frame.
#[derive(Clone, Debug, PartialEq)]
pub struct Hand {
    /// Landmarks in frame pixels, with z in the same scale (smaller is closer).
    pub landmarks: Vec<[f32; 3]>,
    pub confidence: f32,
    pub handedness: f32,
    /// The region the landmarks were read from.
    pub roi: Roi,
    /// True if this frame reused the previous frame's region instead of
    /// running the detector.
    pub tracked: bool,
}

/// Anything that finds a hand in a frame: the live `HandTracker`, or a fake in
/// tests.
pub trait HandSource {
    fn process(&mut self, frame: &RgbImage) -> Result<Option<Hand>>;

    /// Forgets any tracked hand.
    fn reset(&mut self);
}

pub struct HandTracker<D, L> {
    detector: D,
    landmarker: L,
    roi: Option<Roi>,
    /// Frames since the palm detector last ran.
    since_detect: usize,
    /// Hands whose confidence falls below this are treated as lost.
    pub min_confidence: f32,
    /// While no hand is tracked, run the palm detector on at most one frame in
    /// this many. 1 runs it on every frame.
    pub detect_every: usize,
}

impl<D: PalmDetector, L: HandLandmarker> HandTracker<D, L> {
    pub fn new(detector: D, landmarker: L) -> Self {
        Self { detector, landmarker, roi: None, since_detect: usize::MAX, min_confidence: 0.5, detect_every: 4 }
    }

    /// Forgets the tracked hand so the next frame runs the detector.
    pub fn reset(&mut self) {
        self.roi = None;
        self.since_detect = usize::MAX;
    }

    pub fn process(&mut self, frame: &RgbImage) -> Result<Option<Hand>> {
        self.since_detect = self.since_detect.saturating_add(1);
        if let Some(roi) = self.roi
            && let Some(hand) = self.read_landmarks(frame, roi, true)?
        {
            return Ok(Some(hand));
        }
        // A hand lost after tracking for a while is looked for right away,
        // since the detector has not run for a while either.
        if self.since_detect < self.detect_every {
            return Ok(None);
        }
        self.since_detect = 0;
        let palms = self.detector.detect(frame)?;
        let Some(best) = palms.first() else {
            return Ok(None);
        };
        let roi = roi_from_palm(best, frame.width, frame.height);
        self.read_landmarks(frame, roi, false)
    }

    fn read_landmarks(&mut self, frame: &RgbImage, roi: Roi, tracked: bool) -> Result<Option<Hand>> {
        let out = self.landmarker.run(frame, &roi)?;
        if out.confidence < self.min_confidence || out.points.len() < NUM_HAND_LANDMARKS {
            self.roi = None;
            return Ok(None);
        }
        let landmarks = project(&out.points, &roi, HAND_INPUT_SIZE);
        self.roi = Some(roi_from_hand_landmarks(&landmarks));
        Ok(Some(Hand { landmarks, confidence: out.confidence, handedness: out.handedness, roi, tracked }))
    }
}

impl<D: PalmDetector, L: HandLandmarker> HandSource for HandTracker<D, L> {
    fn process(&mut self, frame: &RgbImage) -> Result<Option<Hand>> {
        HandTracker::process(self, frame)
    }

    fn reset(&mut self) {
        HandTracker::reset(self);
    }
}

/// Builds the next frame's crop region from this frame's hand landmarks. The
/// crop is turned so the line from the wrist to the middle finger knuckle
/// points straight up, and the box is measured along the turned axes, so a
/// tilted hand does not get a needlessly large crop.
pub fn roi_from_hand_landmarks(points: &[[f32; 3]]) -> Roi {
    let p = |i: usize| [points[i][0], points[i][1]];
    let angle = rotation(p(WRIST), p(MIDDLE_MCP), FRAC_PI_2);
    let (sin, cos) = angle.sin_cos();
    // Measure the box along the crop's axes: u along its x and v along its y.
    let (mut u0, mut v0, mut u1, mut v1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for &i in &STABLE_LANDMARKS {
        let [x, y] = p(i);
        let (u, v) = (cos * x + sin * y, -sin * x + cos * y);
        u0 = u0.min(u);
        v0 = v0.min(v);
        u1 = u1.max(u);
        v1 = v1.max(v);
    }
    let (u, v) = ((u0 + u1) / 2.0, (v0 + v1) / 2.0);
    let center = Roi { cx: cos * u - sin * v, cy: sin * u + cos * v, width: u1 - u0, height: v1 - v0, angle };
    center.transform(0.0, ROI_SHIFT_Y, ROI_SCALE)
}

/// Reads a model score as a probability. The converted hand model may or may
/// not end with a sigmoid, so a value already between 0 and 1 is taken as a
/// probability and anything else as a raw logit.
pub fn as_probability(x: f32) -> f32 {
    if (0.0..=1.0).contains(&x) { x } else { sigmoid(x) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::detector::Detection;
    use crate::vision::palm::{self, NUM_KEYPOINTS};
    use std::cell::RefCell;
    use std::rc::Rc;

    struct FakeDetector {
        result: Vec<PalmDetection>,
        calls: Rc<RefCell<usize>>,
    }

    impl PalmDetector for FakeDetector {
        fn detect(&mut self, _: &RgbImage) -> Result<Vec<PalmDetection>> {
            *self.calls.borrow_mut() += 1;
            Ok(self.result.clone())
        }
    }

    /// Returns an upright hand in the middle of the crop with the given
    /// confidence values in turn.
    struct FakeLandmarker {
        confidence: Vec<f32>,
        rois: Rc<RefCell<Vec<Roi>>>,
    }

    impl HandLandmarker for FakeLandmarker {
        fn run(&mut self, _: &RgbImage, roi: &Roi) -> Result<HandOutput> {
            self.rois.borrow_mut().push(*roi);
            let confidence = if self.confidence.len() > 1 { self.confidence.remove(0) } else { self.confidence[0] };
            // The stable landmarks span 56 px wide (84 to 140) and 56 px high
            // (112 to 168), centered at (112, 140), with the wrist straight
            // below the middle finger.
            let mut points = vec![[112.0, 140.0, 0.0]; NUM_HAND_LANDMARKS];
            points[WRIST] = [112.0, 168.0, 0.0];
            points[MIDDLE_MCP] = [112.0, 112.0, 0.0];
            points[THUMB_CMC] = [84.0, 150.0, 0.0];
            points[PINKY_MCP] = [140.0, 130.0, 0.0];
            // A fingertip far outside the box, which must not count.
            points[MIDDLE_TIP] = [112.0, 0.0, 0.0];
            Ok(HandOutput { points, confidence, handedness: 0.7 })
        }
    }

    fn palm_detection() -> PalmDetection {
        let mut keypoints = [[0.5, 0.5]; NUM_KEYPOINTS];
        keypoints[palm::WRIST] = [0.5, 0.6];
        keypoints[palm::MIDDLE_MCP] = [0.5, 0.45];
        Detection { score: 0.9, xmin: 0.45, ymin: 0.4, xmax: 0.55, ymax: 0.6, keypoints }
    }

    /// A tracker with fakes, plus handles to the detector call count and the
    /// ROIs the landmark model was asked to read.
    type Rig = (HandTracker<FakeDetector, FakeLandmarker>, Rc<RefCell<usize>>, Rc<RefCell<Vec<Roi>>>);

    fn tracker(palms: Vec<PalmDetection>, confidence: Vec<f32>) -> Rig {
        let calls = Rc::new(RefCell::new(0));
        let rois = Rc::new(RefCell::new(Vec::new()));
        let t = HandTracker::new(
            FakeDetector { result: palms, calls: calls.clone() },
            FakeLandmarker { confidence, rois: rois.clone() },
        );
        (t, calls, rois)
    }

    #[test]
    fn detects_once_then_tracks() {
        let frame = RgbImage::new(640, 480);
        let (mut t, calls, rois) = tracker(vec![palm_detection()], vec![0.95]);
        let first = t.process(&frame).unwrap().unwrap();
        assert!(!first.tracked);
        assert_eq!(first.landmarks.len(), NUM_HAND_LANDMARKS);
        assert_eq!(first.handedness, 0.7);
        for _ in 0..5 {
            assert!(t.process(&frame).unwrap().unwrap().tracked);
        }
        assert_eq!(*calls.borrow(), 1);
        // The palm ROI: the 96 px tall palm box grown 2.6x, moved up 48 px
        // from its center at (320, 240).
        let r0 = rois.borrow()[0];
        assert!((r0.width - 249.6).abs() < 1e-3 && r0.angle.abs() < 1e-6, "{r0:?}");
        assert!((r0.cx - 320.0).abs() < 1e-3 && (r0.cy - 192.0).abs() < 1e-3, "{r0:?}");
        // While tracking, the stable box is a quarter of the crop, so doubled
        // it gives a region half the size, moved up a tenth of the box. The
        // box is centered 28 px below the crop's middle in crop pixels.
        let r1 = rois.borrow()[1];
        let px = r0.width / HAND_INPUT_SIZE as f32;
        assert!((r1.width - r0.width / 2.0).abs() < 1e-2, "{r1:?}");
        assert!((r1.cy - (r0.cy + (28.0 - 5.6) * px)).abs() < 1e-2, "{r1:?}");
        assert!((r1.cx - r0.cx).abs() < 1e-3 && r1.angle.abs() < 1e-6);
    }

    #[test]
    fn rate_limits_the_detector_while_nothing_is_tracked() {
        let frame = RgbImage::new(640, 480);
        let (mut t, calls, rois) = tracker(vec![], vec![0.95]);
        t.detect_every = 3;
        for _ in 0..7 {
            assert!(t.process(&frame).unwrap().is_none());
        }
        // Frames 1, 4 and 7.
        assert_eq!(*calls.borrow(), 3);
        assert!(rois.borrow().is_empty());
    }

    #[test]
    fn lost_hand_waits_for_the_detector_interval() {
        let frame = RgbImage::new(640, 480);
        // Found, lost on the next frame, then good again.
        let (mut t, calls, _) = tracker(vec![palm_detection()], vec![0.95, 0.1, 0.95]);
        t.detect_every = 3;
        assert!(t.process(&frame).unwrap().is_some());
        assert!(t.process(&frame).unwrap().is_none());
        assert!(t.process(&frame).unwrap().is_none());
        let hand = t.process(&frame).unwrap().unwrap();
        assert!(!hand.tracked);
        assert_eq!(*calls.borrow(), 2);
    }

    #[test]
    fn hand_lost_after_long_tracking_redetects_on_the_same_frame() {
        let frame = RgbImage::new(640, 480);
        let mut confidence = vec![0.95; 10];
        confidence.extend([0.1, 0.95]);
        let (mut t, calls, _) = tracker(vec![palm_detection()], confidence);
        t.detect_every = 3;
        for _ in 0..10 {
            assert!(t.process(&frame).unwrap().is_some());
        }
        // Tracking fails (0.1), and the detector has been idle long enough to
        // run again at once.
        let hand = t.process(&frame).unwrap().unwrap();
        assert!(!hand.tracked);
        assert_eq!(*calls.borrow(), 2);
    }

    #[test]
    fn reset_detects_on_the_next_frame() {
        let frame = RgbImage::new(640, 480);
        let (mut t, calls, _) = tracker(vec![], vec![0.95]);
        t.detect_every = 10;
        t.process(&frame).unwrap();
        t.reset();
        t.process(&frame).unwrap();
        assert_eq!(*calls.borrow(), 2);
    }

    #[test]
    fn landmark_roi_follows_a_tilted_hand() {
        // A hand lying on its side with the fingers toward image -x: the box
        // is measured along the turned axes, so its size does not change.
        let mut points = vec![[200.0, 100.0, 0.0]; NUM_HAND_LANDMARKS];
        points[WRIST] = [228.0, 100.0, 0.0];
        points[MIDDLE_MCP] = [172.0, 100.0, 0.0];
        points[THUMB_CMC] = [190.0, 128.0, 0.0];
        points[PINKY_MCP] = [210.0, 72.0, 0.0];
        let roi = roi_from_hand_landmarks(&points);
        assert!((roi.angle + FRAC_PI_2).abs() < 1e-5, "{}", roi.angle);
        assert!((roi.width - 112.0).abs() < 1e-3, "{roi:?}");
        // Center of the box (200, 100), moved 5.6 px toward the fingers.
        assert!((roi.cx - 194.4).abs() < 1e-3 && (roi.cy - 100.0).abs() < 1e-3, "{roi:?}");
    }

    #[test]
    fn scores_read_as_probabilities() {
        assert_eq!(as_probability(0.8), 0.8);
        assert_eq!(as_probability(0.0), 0.0);
        assert!((as_probability(3.0) - sigmoid(3.0)).abs() < 1e-6);
        assert!(as_probability(-4.0) < 0.05);
    }
}
