//! MediaPipe palm detection, the hand counterpart of BlazeFace.
//!
//! It uses the same anchor scheme and decoding as the face detector (see
//! `detector.rs`), with a 192x192 input, 2016 anchors and seven keypoints per
//! palm. It finds palms rather than whole hands because a palm is a rigid,
//! roughly square shape, while fingers move. The hand crop is then grown out
//! from the palm, the way MediaPipe's palm_detection_detection_to_roi graph
//! does it.

use std::f32::consts::FRAC_PI_2;

use super::detector::{self, Anchor, Detection, Letterbox};
use super::roi::{Roi, rotation};

/// Palm detector input side in pixels.
pub const INPUT_SIZE: usize = 192;
pub const NUM_ANCHORS: usize = 2016;
pub const NUM_KEYPOINTS: usize = 7;
pub const NUM_COORDS: usize = 4 + 2 * NUM_KEYPOINTS;

/// Keypoint order in palm detector output.
pub const WRIST: usize = 0;
/// The knuckle at the base of the middle finger.
pub const MIDDLE_MCP: usize = 2;

/// The hand crop starts at the palm box, moves half its size toward the
/// fingers, and grows 2.6x so the whole hand fits.
const ROI_SHIFT_Y: f32 = -0.5;
const ROI_SCALE: f32 = 2.6;

pub type PalmDetection = Detection<NUM_KEYPOINTS>;

/// Generates the 2016 anchors for the palm model.
pub fn anchors() -> Vec<Anchor> {
    detector::ssd_anchors(INPUT_SIZE, &detector::ANCHOR_LAYERS)
}

/// Turns raw model output into merged palms in frame coordinates.
///
/// `regressors` is 2016 x 18 and `scores` is 2016 raw logits.
pub fn decode(regressors: &[f32], scores: &[f32], anchors: &[Anchor], letterbox: Letterbox) -> Vec<PalmDetection> {
    detector::decode(regressors, scores, anchors, INPUT_SIZE, letterbox)
}

/// Builds the hand crop region from a palm detection. The crop is turned so
/// the line from the wrist to the middle finger points straight up.
pub fn roi_from_palm(d: &PalmDetection, width: usize, height: usize) -> Roi {
    let (w, h) = (width as f32, height as f32);
    let px = |p: [f32; 2]| [p[0] * w, p[1] * h];
    let angle = rotation(px(d.keypoints[WRIST]), px(d.keypoints[MIDDLE_MCP]), FRAC_PI_2);
    let palm = Roi {
        cx: (d.xmin + d.xmax) / 2.0 * w,
        cy: (d.ymin + d.ymax) / 2.0 * h,
        width: (d.xmax - d.xmin) * w,
        height: (d.ymax - d.ymin) * h,
        angle,
    };
    palm.transform(0.0, ROI_SHIFT_Y, ROI_SCALE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchor_layout() {
        let a = anchors();
        assert_eq!(a.len(), NUM_ANCHORS);
        assert_eq!(a[0], Anchor { cx: 0.5 / 24.0, cy: 0.5 / 24.0 });
        assert_eq!(a[1], a[0]);
        // 24 x 24 cells at stride 8 with two anchors each come first.
        assert_eq!(a[1152], Anchor { cx: 0.5 / 12.0, cy: 0.5 / 12.0 });
        assert_eq!(a[2015], Anchor { cx: 11.5 / 12.0, cy: 11.5 / 12.0 });
    }

    #[test]
    fn decodes_seven_keypoints() {
        let a = anchors();
        let mut r = vec![0.0; NUM_ANCHORS * NUM_COORDS];
        let mut s = vec![-50.0; NUM_ANCHORS];
        let i = 1500;
        // A 96 px box (half the input) with the middle finger knuckle 19.2 px
        // (0.1 of the input) above the anchor and the last keypoint 9.6 px right.
        r[i * NUM_COORDS + 2] = 96.0;
        r[i * NUM_COORDS + 3] = 96.0;
        r[i * NUM_COORDS + 4 + 2 * MIDDLE_MCP + 1] = -19.2;
        r[i * NUM_COORDS + 4 + 2 * 6] = 9.6;
        s[i] = 3.0;
        let d = decode(&r, &s, &a, Letterbox { pad_x: 0.0, pad_y: 0.0 });
        assert_eq!(d.len(), 1);
        let d = &d[0];
        assert!((d.xmin - (a[i].cx - 0.25)).abs() < 1e-5 && (d.ymax - (a[i].cy + 0.25)).abs() < 1e-5);
        assert!((d.keypoints[MIDDLE_MCP][1] - (a[i].cy - 0.1)).abs() < 1e-5);
        assert!((d.keypoints[6][0] - (a[i].cx + 0.05)).abs() < 1e-5);
        assert!((d.score - detector::sigmoid(3.0)).abs() < 1e-6);
    }

    fn palm(wrist: [f32; 2], middle: [f32; 2]) -> PalmDetection {
        let mut keypoints = [[0.5, 0.5]; NUM_KEYPOINTS];
        keypoints[WRIST] = wrist;
        keypoints[MIDDLE_MCP] = middle;
        Detection { score: 0.9, xmin: 0.4, ymin: 0.4, xmax: 0.6, ymax: 0.6, keypoints }
    }

    #[test]
    fn upright_palm_grows_toward_the_fingers() {
        // On a 1000 x 1000 frame the palm box is 200 px around (500, 500).
        let roi = roi_from_palm(&palm([0.5, 0.6], [0.5, 0.45]), 1000, 1000);
        assert!(roi.angle.abs() < 1e-6);
        assert!((roi.cx - 500.0).abs() < 1e-3 && (roi.cy - 400.0).abs() < 1e-3, "{roi:?}");
        assert!((roi.width - 520.0).abs() < 1e-3 && roi.width == roi.height);
    }

    #[test]
    fn sideways_palm_turns_the_crop() {
        // Fingers point to image +x, so the crop turns a quarter turn and the
        // shift moves the center right instead of up.
        let roi = roi_from_palm(&palm([0.4, 0.5], [0.55, 0.5]), 1000, 1000);
        assert!((roi.angle - FRAC_PI_2).abs() < 1e-5, "{}", roi.angle);
        assert!((roi.cx - 600.0).abs() < 1e-3 && (roi.cy - 500.0).abs() < 1e-3, "{roi:?}");
    }
}
