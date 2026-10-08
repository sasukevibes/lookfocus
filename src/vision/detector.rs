//! BlazeFace short-range face detection: anchors, decoding and merging.
//!
//! The model looks at a 128x128 letterboxed copy of the frame and scores 896
//! fixed anchor boxes. Each anchor also regresses a box and six keypoints. The
//! numbers here come from MediaPipe's face_detection_short_range graph:
//! strides 8, 16, 16, 16, two anchors per cell, scales fixed to 1, and a
//! weighted non-maximum suppression at IoU 0.3.

pub const INPUT_SIZE: usize = 128;
pub const NUM_ANCHORS: usize = 896;
pub const NUM_COORDS: usize = 16;
pub const MIN_SCORE: f32 = 0.5;
const NMS_IOU: f32 = 0.3;
const SCORE_CLIP: f32 = 100.0;

/// Keypoint order in BlazeFace output.
pub const RIGHT_EYE: usize = 0;
pub const LEFT_EYE: usize = 1;

/// An anchor center in normalized input coordinates. All anchors have size 1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Anchor {
    pub cx: f32,
    pub cy: f32,
}

/// Generates the 896 SSD anchors for the short-range model.
pub fn anchors() -> Vec<Anchor> {
    // Layers that share a stride are merged, each adding two anchors per cell
    // (one for the layer's scale and one interpolated), so stride 8 gives two
    // anchors per cell and the three stride-16 layers give six.
    let mut out = Vec::with_capacity(NUM_ANCHORS);
    for (stride, per_cell) in [(8usize, 2usize), (16, 6)] {
        let cells = INPUT_SIZE / stride;
        for y in 0..cells {
            for x in 0..cells {
                for _ in 0..per_cell {
                    out.push(Anchor { cx: (x as f32 + 0.5) / cells as f32, cy: (y as f32 + 0.5) / cells as f32 });
                }
            }
        }
    }
    out
}

/// A detected face in normalized coordinates of the original frame (0 to 1).
#[derive(Clone, Debug, PartialEq)]
pub struct Detection {
    pub score: f32,
    pub xmin: f32,
    pub ymin: f32,
    pub xmax: f32,
    pub ymax: f32,
    pub keypoints: [[f32; 2]; 6],
}

impl Detection {
    fn iou(&self, other: &Detection) -> f32 {
        let ix = (self.xmax.min(other.xmax) - self.xmin.max(other.xmin)).max(0.0);
        let iy = (self.ymax.min(other.ymax) - self.ymin.max(other.ymin)).max(0.0);
        let inter = ix * iy;
        let union = self.area() + other.area() - inter;
        if union <= 0.0 { 0.0 } else { inter / union }
    }

    fn area(&self) -> f32 {
        (self.xmax - self.xmin).max(0.0) * (self.ymax - self.ymin).max(0.0)
    }
}

/// How a frame was fitted into the square model input: the content scale and
/// the padding on each side, all as fractions of the input size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Letterbox {
    pub pad_x: f32,
    pub pad_y: f32,
}

impl Letterbox {
    pub fn for_frame(width: usize, height: usize) -> Self {
        let (w, h) = (width as f32, height as f32);
        if w >= h {
            Self { pad_x: 0.0, pad_y: (1.0 - h / w) / 2.0 }
        } else {
            Self { pad_x: (1.0 - w / h) / 2.0, pad_y: 0.0 }
        }
    }

    /// Maps a normalized model-input point back to the normalized frame.
    fn unpad(&self, x: f32, y: f32) -> [f32; 2] {
        [(x - self.pad_x) / (1.0 - 2.0 * self.pad_x), (y - self.pad_y) / (1.0 - 2.0 * self.pad_y)]
    }
}

/// Turns raw model output into merged detections in frame coordinates.
///
/// `regressors` is 896 x 16 and `scores` is 896 raw logits.
pub fn decode(regressors: &[f32], scores: &[f32], anchors: &[Anchor], letterbox: Letterbox) -> Vec<Detection> {
    let size = INPUT_SIZE as f32;
    let mut raw = Vec::new();
    for (i, a) in anchors.iter().enumerate() {
        let score = sigmoid(scores[i].clamp(-SCORE_CLIP, SCORE_CLIP));
        if score < MIN_SCORE {
            continue;
        }
        let r = &regressors[i * NUM_COORDS..(i + 1) * NUM_COORDS];
        let cx = r[0] / size + a.cx;
        let cy = r[1] / size + a.cy;
        let w = r[2] / size;
        let h = r[3] / size;
        let [xmin, ymin] = letterbox.unpad(cx - w / 2.0, cy - h / 2.0);
        let [xmax, ymax] = letterbox.unpad(cx + w / 2.0, cy + h / 2.0);
        let mut keypoints = [[0.0; 2]; 6];
        for (k, kp) in keypoints.iter_mut().enumerate() {
            *kp = letterbox.unpad(r[4 + 2 * k] / size + a.cx, r[5 + 2 * k] / size + a.cy);
        }
        raw.push(Detection { score, xmin, ymin, xmax, ymax, keypoints });
    }
    weighted_nms(raw)
}

/// MediaPipe's weighted non-maximum suppression: overlapping detections are
/// averaged by score into the strongest one instead of being dropped.
pub fn weighted_nms(mut dets: Vec<Detection>) -> Vec<Detection> {
    dets.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut out = Vec::new();
    while !dets.is_empty() {
        let top = dets[0].clone();
        let (cluster, rest): (Vec<_>, Vec<_>) = dets.into_iter().partition(|d| d.iou(&top) > NMS_IOU);
        dets = rest;
        let total: f32 = cluster.iter().map(|d| d.score).sum();
        let avg = |f: &dyn Fn(&Detection) -> f32| cluster.iter().map(|d| f(d) * d.score).sum::<f32>() / total;
        let mut merged = top.clone();
        merged.xmin = avg(&|d| d.xmin);
        merged.ymin = avg(&|d| d.ymin);
        merged.xmax = avg(&|d| d.xmax);
        merged.ymax = avg(&|d| d.ymax);
        for k in 0..6 {
            for c in 0..2 {
                merged.keypoints[k][c] = avg(&|d| d.keypoints[k][c]);
            }
        }
        out.push(merged);
    }
    out
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchor_layout() {
        let a = anchors();
        assert_eq!(a.len(), NUM_ANCHORS);
        assert_eq!(a[0], Anchor { cx: 0.5 / 16.0, cy: 0.5 / 16.0 });
        assert_eq!(a[1], a[0]);
        assert_eq!(a[2].cx, 1.5 / 16.0);
        // The first stride-16 anchor follows the 512 stride-8 ones.
        assert_eq!(a[512], Anchor { cx: 0.5 / 8.0, cy: 0.5 / 8.0 });
        assert_eq!(a[895], Anchor { cx: 7.5 / 8.0, cy: 7.5 / 8.0 });
    }

    fn one_hit(index: usize, reg: [f32; NUM_COORDS], logit: f32) -> (Vec<f32>, Vec<f32>) {
        let mut r = vec![0.0; NUM_ANCHORS * NUM_COORDS];
        let mut s = vec![-50.0; NUM_ANCHORS];
        r[index * NUM_COORDS..(index + 1) * NUM_COORDS].copy_from_slice(&reg);
        s[index] = logit;
        (r, s)
    }

    #[test]
    fn decodes_box_and_keypoints() {
        let a = anchors();
        // Anchor 600 sits at a stride-16 cell. Offset the box 12.8 px right
        // (0.1 of the input) and make it 64 px (half the input) square.
        let mut reg = [0.0; NUM_COORDS];
        reg[0] = 12.8;
        reg[2] = 64.0;
        reg[3] = 64.0;
        reg[4] = -16.0; // right eye 16 px left of the anchor
        reg[6] = 16.0; // left eye 16 px right
        let (r, s) = one_hit(600, reg, 4.0);
        let none = Letterbox { pad_x: 0.0, pad_y: 0.0 };
        let d = decode(&r, &s, &a, none);
        assert_eq!(d.len(), 1);
        let d = &d[0];
        let cx = a[600].cx + 0.1;
        assert!((d.xmin - (cx - 0.25)).abs() < 1e-5 && (d.xmax - (cx + 0.25)).abs() < 1e-5);
        assert!((d.ymin - (a[600].cy - 0.25)).abs() < 1e-5);
        assert!((d.keypoints[RIGHT_EYE][0] - (a[600].cx - 0.125)).abs() < 1e-5);
        assert!((d.keypoints[LEFT_EYE][0] - (a[600].cx + 0.125)).abs() < 1e-5);
        assert!((d.score - sigmoid(4.0)).abs() < 1e-6);
    }

    #[test]
    fn drops_low_scores() {
        let (r, s) = one_hit(10, [0.0; NUM_COORDS], -1.0);
        assert!(decode(&r, &s, &anchors(), Letterbox { pad_x: 0.0, pad_y: 0.0 }).is_empty());
    }

    #[test]
    fn letterbox_for_a_wide_frame() {
        // 640x480 fills the width and leaves 1/8 of the height empty top and bottom.
        let lb = Letterbox::for_frame(640, 480);
        assert_eq!(lb.pad_x, 0.0);
        assert!((lb.pad_y - 0.125).abs() < 1e-6);
        assert_eq!(lb.unpad(0.5, 0.125), [0.5, 0.0]);
        assert_eq!(lb.unpad(0.5, 0.875), [0.5, 1.0]);
        let tall = Letterbox::for_frame(480, 640);
        assert!((tall.pad_x - 0.125).abs() < 1e-6 && tall.pad_y == 0.0);
    }

    #[test]
    fn nms_merges_overlaps_and_keeps_separate_faces() {
        let det = |score: f32, x: f32| Detection {
            score,
            xmin: x,
            ymin: 0.2,
            xmax: x + 0.2,
            ymax: 0.4,
            keypoints: [[x, 0.3]; 6],
        };
        let out = weighted_nms(vec![det(0.9, 0.10), det(0.6, 0.12), det(0.8, 0.70)]);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].score, 0.9);
        // Score-weighted average of 0.10 and 0.12.
        let expect = (0.9 * 0.10 + 0.6 * 0.12) / 1.5;
        assert!((out[0].xmin - expect).abs() < 1e-6);
        assert!((out[1].xmin - 0.70).abs() < 1e-6);
    }
}
