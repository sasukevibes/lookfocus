//! Face detection and landmark tracking.
//!
//! Like MediaPipe's video mode, the detector only runs when there is no face
//! being tracked. Once the landmark model finds a face, its landmarks give the
//! region to crop on the next frame, which is cheaper and steadier than
//! detecting every time.
//!
//! The two models sit behind traits so the tracking logic can be tested with
//! fakes, without model files or a camera.

pub mod detector;
pub mod roi;

use anyhow::Result;

use crate::image::RgbImage;
use detector::{Detection, LEFT_EYE, RIGHT_EYE};
pub use roi::Roi;

/// Side length of the landmark model's square input.
pub const MESH_INPUT_SIZE: usize = 256;
/// Number of landmarks the face mesh model produces (468 face plus 10 iris).
pub const NUM_LANDMARKS: usize = 478;
/// Landmark indices for the outer eye corners. MediaPipe uses these to level
/// the crop while tracking.
const RIGHT_EYE_OUTER: usize = 33;
const LEFT_EYE_OUTER: usize = 263;

pub trait FaceDetector {
    /// Detections in normalized frame coordinates, strongest first.
    fn detect(&mut self, frame: &RgbImage) -> Result<Vec<Detection>>;
}

/// Raw landmark model output for one crop.
#[derive(Clone, Debug, PartialEq)]
pub struct MeshOutput {
    /// NUM_LANDMARKS points as (x, y, z) in crop pixels (0 to MESH_INPUT_SIZE).
    pub points: Vec<[f32; 3]>,
    /// Probability (0 to 1) that the crop really contains a face.
    pub presence: f32,
}

pub trait FaceMesh {
    fn run(&mut self, frame: &RgbImage, roi: &Roi) -> Result<MeshOutput>;
}

/// A tracked face for one frame.
#[derive(Clone, Debug, PartialEq)]
pub struct Face {
    /// Landmarks in frame pixels, with z in the same scale (smaller is closer).
    pub landmarks: Vec<[f32; 3]>,
    pub presence: f32,
    /// The region the landmarks were read from.
    pub roi: Roi,
    /// True if this frame reused the previous frame's region instead of
    /// running the detector.
    pub tracked: bool,
}

pub struct FaceTracker<D, M> {
    detector: D,
    mesh: M,
    roi: Option<Roi>,
    /// Faces whose presence falls below this are treated as lost.
    pub min_presence: f32,
}

impl<D: FaceDetector, M: FaceMesh> FaceTracker<D, M> {
    pub fn new(detector: D, mesh: M) -> Self {
        Self { detector, mesh, roi: None, min_presence: 0.5 }
    }

    /// Forgets the tracked face so the next frame runs the detector.
    pub fn reset(&mut self) {
        self.roi = None;
    }

    pub fn process(&mut self, frame: &RgbImage) -> Result<Option<Face>> {
        if let Some(roi) = self.roi
            && let Some(face) = self.read_landmarks(frame, roi, true)?
        {
            return Ok(Some(face));
        }
        // Nothing tracked, or the face was lost while tracking. Detect right
        // away rather than waiting for the next frame.
        let detections = self.detector.detect(frame)?;
        let Some(best) = detections.first() else {
            self.roi = None;
            return Ok(None);
        };
        let roi = roi_from_detection(best, frame.width, frame.height);
        self.read_landmarks(frame, roi, false)
    }

    fn read_landmarks(&mut self, frame: &RgbImage, roi: Roi, tracked: bool) -> Result<Option<Face>> {
        let out = self.mesh.run(frame, &roi)?;
        if out.presence < self.min_presence || out.points.len() < NUM_LANDMARKS {
            self.roi = None;
            return Ok(None);
        }
        let landmarks = project_landmarks(&out.points, &roi);
        self.roi = Some(roi_from_landmarks(&landmarks));
        Ok(Some(Face { landmarks, presence: out.presence, roi, tracked }))
    }
}

/// Builds the landmark crop region from a detection.
pub fn roi_from_detection(d: &Detection, width: usize, height: usize) -> Roi {
    let (w, h) = (width as f32, height as f32);
    let px = |p: [f32; 2]| [p[0] * w, p[1] * h];
    Roi::from_box_and_eyes(
        d.xmin * w,
        d.ymin * h,
        d.xmax * w,
        d.ymax * h,
        px(d.keypoints[RIGHT_EYE]),
        px(d.keypoints[LEFT_EYE]),
    )
}

/// Builds the next frame's crop region from this frame's landmarks.
pub fn roi_from_landmarks(points: &[[f32; 3]]) -> Roi {
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for p in points {
        x0 = x0.min(p[0]);
        y0 = y0.min(p[1]);
        x1 = x1.max(p[0]);
        y1 = y1.max(p[1]);
    }
    let eye = |i: usize| [points[i][0], points[i][1]];
    Roi::from_box_and_eyes(x0, y0, x1, y1, eye(RIGHT_EYE_OUTER), eye(LEFT_EYE_OUTER))
}

/// Maps landmarks from crop pixels back to frame pixels. Depth is scaled by
/// the crop width so it stays in the same units as x and y.
pub fn project_landmarks(points: &[[f32; 3]], roi: &Roi) -> Vec<[f32; 3]> {
    let size = MESH_INPUT_SIZE as f32;
    points
        .iter()
        .map(|p| {
            let [x, y] = roi.to_image(p[0] / size, p[1] / size);
            [x, y, p[2] / size * roi.width]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    struct FakeDetector {
        result: Vec<Detection>,
        calls: Rc<RefCell<usize>>,
    }

    impl FaceDetector for FakeDetector {
        fn detect(&mut self, _: &RgbImage) -> Result<Vec<Detection>> {
            *self.calls.borrow_mut() += 1;
            Ok(self.result.clone())
        }
    }

    /// Returns a face spread across the middle of the crop with the given
    /// presence values in turn.
    struct FakeMesh {
        presence: Vec<f32>,
        rois: Rc<RefCell<Vec<Roi>>>,
    }

    impl FaceMesh for FakeMesh {
        fn run(&mut self, _: &RgbImage, roi: &Roi) -> Result<MeshOutput> {
            self.rois.borrow_mut().push(*roi);
            let presence = if self.presence.len() > 1 { self.presence.remove(0) } else { self.presence[0] };
            let mut points = vec![[128.0, 128.0, 0.0]; NUM_LANDMARKS];
            // Spread a box from 64 to 192 so the next ROI has a known size.
            points[0] = [64.0, 64.0, 0.0];
            points[1] = [192.0, 192.0, 0.0];
            points[RIGHT_EYE_OUTER] = [96.0, 110.0, 0.0];
            points[LEFT_EYE_OUTER] = [160.0, 110.0, 0.0];
            Ok(MeshOutput { points, presence })
        }
    }

    fn detection() -> Detection {
        Detection {
            score: 0.9,
            xmin: 0.4,
            ymin: 0.3,
            xmax: 0.6,
            ymax: 0.6,
            keypoints: [[0.45, 0.4], [0.55, 0.4], [0.5, 0.5], [0.5, 0.55], [0.4, 0.42], [0.6, 0.42]],
        }
    }

    /// A tracker with fakes, plus handles to the detector call count and the
    /// ROIs the mesh was asked to read.
    type Rig = (FaceTracker<FakeDetector, FakeMesh>, Rc<RefCell<usize>>, Rc<RefCell<Vec<Roi>>>);

    fn tracker(dets: Vec<Detection>, presence: Vec<f32>) -> Rig {
        let calls = Rc::new(RefCell::new(0));
        let rois = Rc::new(RefCell::new(Vec::new()));
        let t = FaceTracker::new(
            FakeDetector { result: dets, calls: calls.clone() },
            FakeMesh { presence, rois: rois.clone() },
        );
        (t, calls, rois)
    }

    #[test]
    fn detects_once_then_tracks() {
        let frame = RgbImage::new(640, 480);
        let (mut t, calls, rois) = tracker(vec![detection()], vec![0.99]);
        let first = t.process(&frame).unwrap().unwrap();
        assert!(!first.tracked);
        for _ in 0..5 {
            assert!(t.process(&frame).unwrap().unwrap().tracked);
        }
        assert_eq!(*calls.borrow(), 1);
        // The detection ROI: a 128 x 144 px box made square on 144 and grown 1.5x.
        let r0 = rois.borrow()[0];
        assert!((r0.width - 216.0).abs() < 1e-3 && r0.angle.abs() < 1e-6);
        assert!((r0.cx - 320.0).abs() < 1e-3 && (r0.cy - 216.0).abs() < 1e-3);
        // While tracking, the landmark box (half the crop) times 1.5 gives a
        // region 0.75 the size of the one before it.
        let r1 = rois.borrow()[1];
        assert!((r1.width - 0.75 * r0.width).abs() < 1e-2, "{r1:?}");
    }

    #[test]
    fn redetects_on_the_same_frame_when_tracking_is_lost() {
        let frame = RgbImage::new(640, 480);
        let (mut t, calls, _) = tracker(vec![detection()], vec![0.99, 0.1, 0.99]);
        assert!(t.process(&frame).unwrap().is_some());
        // Tracking fails (0.1), detection runs again and finds the face.
        let face = t.process(&frame).unwrap().unwrap();
        assert!(!face.tracked);
        assert_eq!(*calls.borrow(), 2);
    }

    #[test]
    fn no_detection_means_no_face() {
        let frame = RgbImage::new(640, 480);
        let (mut t, calls, rois) = tracker(vec![], vec![0.99]);
        assert!(t.process(&frame).unwrap().is_none());
        assert!(t.process(&frame).unwrap().is_none());
        assert_eq!(*calls.borrow(), 2);
        assert!(rois.borrow().is_empty());
    }

    #[test]
    fn low_presence_drops_the_face() {
        let frame = RgbImage::new(640, 480);
        let (mut t, calls, _) = tracker(vec![detection()], vec![0.2]);
        assert!(t.process(&frame).unwrap().is_none());
        // Nothing is tracked, so the next frame detects again.
        assert!(t.process(&frame).unwrap().is_none());
        assert_eq!(*calls.borrow(), 2);
    }

    #[test]
    fn projection_maps_crop_to_frame() {
        let roi = Roi { cx: 300.0, cy: 200.0, width: 128.0, height: 128.0, angle: 0.0 };
        let out = project_landmarks(&[[128.0, 128.0, 0.0], [0.0, 0.0, 25.6], [256.0, 256.0, -25.6]], &roi);
        assert_eq!(out[0], [300.0, 200.0, 0.0]);
        assert_eq!(out[1], [236.0, 136.0, 12.8]);
        assert_eq!(out[2], [364.0, 264.0, -12.8]);
    }
}
