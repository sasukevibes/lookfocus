//! Rotated square regions of interest (ROIs) around the face.
//!
//! This follows MediaPipe's face landmarker: the region is rotated so the eyes
//! sit level, made square on its longer side, and grown by 1.5x so the whole
//! face fits. The same mapping crops the model input and maps the model's
//! landmarks back to the full image, so the two always agree.

use crate::image::RgbImage;

/// MediaPipe grows both the detection box and the landmark box by this much.
pub const ROI_SCALE: f32 = 1.5;

/// A rotated rectangle in image pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Roi {
    pub cx: f32,
    pub cy: f32,
    pub width: f32,
    pub height: f32,
    /// Radians. Positive turns the crop's x axis clockwise in the image (toward
    /// +y, since image y points down).
    pub angle: f32,
}

impl Roi {
    /// Builds the ROI from an axis-aligned box and two reference points (the
    /// eyes), rotating so the line from `a` to `b` becomes horizontal.
    pub fn from_box_and_eyes(x0: f32, y0: f32, x1: f32, y1: f32, a: [f32; 2], b: [f32; 2]) -> Self {
        let angle = normalize_radians(-(-(b[1] - a[1])).atan2(b[0] - a[0]));
        let side = (x1 - x0).max(y1 - y0) * ROI_SCALE;
        Self { cx: (x0 + x1) / 2.0, cy: (y0 + y1) / 2.0, width: side, height: side, angle }
    }

    /// Maps a point given in the crop's normalized coordinates (0 to 1 across
    /// the crop) to image pixels.
    pub fn to_image(&self, u: f32, v: f32) -> [f32; 2] {
        let (sin, cos) = self.angle.sin_cos();
        let x = (u - 0.5) * self.width;
        let y = (v - 0.5) * self.height;
        [self.cx + cos * x - sin * y, self.cy + sin * x + cos * y]
    }

    /// Fills `out` (size x size x 3, row major, RGB) with the crop, scaling
    /// pixel values with `scale` and `offset`: value = pixel * scale + offset.
    pub fn crop_into(&self, img: &RgbImage, size: usize, scale: f32, offset: f32, out: &mut [f32]) {
        assert_eq!(out.len(), size * size * 3);
        let border = offset; // Outside the image counts as black.
        for row in 0..size {
            for col in 0..size {
                let [x, y] = self.to_image((col as f32 + 0.5) / size as f32, (row as f32 + 0.5) / size as f32);
                let o = (row * size + col) * 3;
                if x < -1.0 || y < -1.0 || x > img.width as f32 + 1.0 || y > img.height as f32 + 1.0 {
                    out[o..o + 3].fill(border);
                } else {
                    let px = img.sample(x, y);
                    for c in 0..3 {
                        out[o + c] = px[c] * scale + offset;
                    }
                }
            }
        }
    }
}

pub fn normalize_radians(a: f32) -> f32 {
    use std::f32::consts::PI;
    a - 2.0 * PI * ((a + PI) / (2.0 * PI)).floor()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_4;

    #[test]
    fn level_eyes_give_an_upright_square() {
        let roi = Roi::from_box_and_eyes(100.0, 50.0, 200.0, 130.0, [120.0, 80.0], [180.0, 80.0]);
        assert_eq!((roi.cx, roi.cy), (150.0, 90.0));
        assert_eq!((roi.width, roi.height), (150.0, 150.0));
        assert!(roi.angle.abs() < 1e-6);
    }

    #[test]
    fn tilted_eyes_rotate_the_crop() {
        // The second eye is lower in the image by 45 degrees.
        let roi = Roi::from_box_and_eyes(0.0, 0.0, 10.0, 10.0, [0.0, 0.0], [10.0, 10.0]);
        assert!((roi.angle - FRAC_PI_4).abs() < 1e-5, "{}", roi.angle);
        // The crop's x axis then runs along the eye line.
        let a = roi.to_image(0.0, 0.5);
        let b = roi.to_image(1.0, 0.5);
        assert!(((b[1] - a[1]) - (b[0] - a[0])).abs() < 1e-3);
    }

    #[test]
    fn to_image_maps_corners_and_center() {
        let roi = Roi { cx: 50.0, cy: 40.0, width: 20.0, height: 10.0, angle: 0.0 };
        assert_eq!(roi.to_image(0.5, 0.5), [50.0, 40.0]);
        assert_eq!(roi.to_image(0.0, 0.0), [40.0, 35.0]);
        assert_eq!(roi.to_image(1.0, 1.0), [60.0, 45.0]);
    }

    #[test]
    fn crop_reproduces_the_image_at_identity() {
        let img = RgbImage::from_fn(8, 8, |x, y| [(x * 30) as u8, (y * 30) as u8, 7]);
        let roi = Roi { cx: 4.0, cy: 4.0, width: 8.0, height: 8.0, angle: 0.0 };
        let mut out = vec![0.0; 8 * 8 * 3];
        roi.crop_into(&img, 8, 1.0, 0.0, &mut out);
        for y in 0..8 {
            for x in 0..8 {
                let o = (y * 8 + x) * 3;
                assert_eq!(&out[o..o + 3], &[(x * 30) as f32, (y * 30) as f32, 7.0]);
            }
        }
    }

    #[test]
    fn crop_applies_scale_offset_and_border() {
        let img = RgbImage::from_fn(4, 4, |_, _| [255, 255, 255]);
        // A crop twice as large as the image: the middle is white, edges black.
        // Crop pixel centers land at x = 2 * col - 5, so cols 3 and 4 are inside.
        let roi = Roi { cx: 2.0, cy: 2.0, width: 16.0, height: 16.0, angle: 0.0 };
        let mut out = vec![0.0; 8 * 8 * 3];
        roi.crop_into(&img, 8, 2.0 / 255.0, -1.0, &mut out);
        assert_eq!(out[0], -1.0);
        let inside = (3 * 8 + 3) * 3;
        assert!((out[inside] - 1.0).abs() < 1e-5, "{}", out[inside]);
    }

    #[test]
    fn rotated_crop_round_trips() {
        let roi = Roi { cx: 300.0, cy: 200.0, width: 120.0, height: 120.0, angle: 0.3 };
        // A point at crop coordinates (0.2, 0.7) maps out and back.
        let p = roi.to_image(0.2, 0.7);
        let (sin, cos) = roi.angle.sin_cos();
        let (dx, dy) = (p[0] - roi.cx, p[1] - roi.cy);
        let u = (cos * dx + sin * dy) / roi.width + 0.5;
        let v = (-sin * dx + cos * dy) / roi.height + 0.5;
        assert!((u - 0.2).abs() < 1e-5 && (v - 0.7).abs() < 1e-5);
    }

    #[test]
    fn normalize_wraps_into_range() {
        use std::f32::consts::PI;
        assert!((normalize_radians(3.0 * PI) - PI).abs() < 1e-4 || (normalize_radians(3.0 * PI) + PI).abs() < 1e-4);
        assert!((normalize_radians(0.5) - 0.5).abs() < 1e-6);
        assert!((normalize_radians(-0.5 - 2.0 * PI) + 0.5).abs() < 1e-5);
    }
}
