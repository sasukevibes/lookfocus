//! A minimal RGB image type and the pixel conversions the pipeline needs.
//!
//! Frames only ever live in memory. Nothing in this crate writes them to disk.

/// An 8-bit RGB image stored row by row.
#[derive(Clone, Debug, PartialEq)]
pub struct RgbImage {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl RgbImage {
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height, data: vec![0; width * height * 3] }
    }

    /// Builds an image from a function of (x, y) that returns RGB.
    pub fn from_fn(width: usize, height: usize, f: impl Fn(usize, usize) -> [u8; 3]) -> Self {
        let mut img = Self::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let i = (y * width + x) * 3;
                img.data[i..i + 3].copy_from_slice(&f(x, y));
            }
        }
        img
    }

    /// Converts a packed YUYV (YUY2) buffer to RGB using BT.601 limited range,
    /// which is what UVC webcams send.
    pub fn from_yuyv(width: usize, height: usize, yuyv: &[u8]) -> Self {
        let mut img = Self::new(width, height);
        let pairs = (width * height / 2).min(yuyv.len() / 4);
        for p in 0..pairs {
            let s = &yuyv[p * 4..p * 4 + 4];
            let (y0, u, y1, v) = (s[0], s[1], s[2], s[3]);
            let o = p * 6;
            img.data[o..o + 3].copy_from_slice(&yuv_to_rgb(y0, u, v));
            img.data[o + 3..o + 6].copy_from_slice(&yuv_to_rgb(y1, u, v));
        }
        img
    }

    /// Mean brightness (0 to 255) of an axis-aligned region, clamped to the
    /// image. Used for the "lighting is poor" check.
    pub fn mean_luma(&self, x0: f32, y0: f32, x1: f32, y1: f32) -> f32 {
        let xa = x0.max(0.0) as usize;
        let ya = y0.max(0.0) as usize;
        let xb = (x1.max(0.0) as usize).min(self.width);
        let yb = (y1.max(0.0) as usize).min(self.height);
        if xb <= xa || yb <= ya {
            return 0.0;
        }
        let mut sum = 0u64;
        for y in ya..yb {
            for x in xa..xb {
                let i = (y * self.width + x) * 3;
                let (r, g, b) = (self.data[i] as u64, self.data[i + 1] as u64, self.data[i + 2] as u64);
                sum += (299 * r + 587 * g + 114 * b) / 1000;
            }
        }
        sum as f32 / ((xb - xa) * (yb - ya)) as f32
    }

    /// Bilinear sample at a sub-pixel position. Pixel centers sit at
    /// half-integer coordinates. Positions outside the image read as black.
    pub fn sample(&self, x: f32, y: f32) -> [f32; 3] {
        let fx = x - 0.5;
        let fy = y - 0.5;
        let x0 = fx.floor();
        let y0 = fy.floor();
        let ax = fx - x0;
        let ay = fy - y0;
        let (x0, y0) = (x0 as i64, y0 as i64);
        let mut out = [0.0f32; 3];
        for (dy, wy) in [(0, 1.0 - ay), (1, ay)] {
            for (dx, wx) in [(0, 1.0 - ax), (1, ax)] {
                let w = wx * wy;
                if w == 0.0 {
                    continue;
                }
                if let Some(px) = self.pixel(x0 + dx, y0 + dy) {
                    for c in 0..3 {
                        out[c] += w * px[c] as f32;
                    }
                }
            }
        }
        out
    }

    fn pixel(&self, x: i64, y: i64) -> Option<&[u8]> {
        if x < 0 || y < 0 || x >= self.width as i64 || y >= self.height as i64 {
            return None;
        }
        let i = (y as usize * self.width + x as usize) * 3;
        Some(&self.data[i..i + 3])
    }
}

fn yuv_to_rgb(y: u8, u: u8, v: u8) -> [u8; 3] {
    let c = (y as f32 - 16.0) * 1.164;
    let d = u as f32 - 128.0;
    let e = v as f32 - 128.0;
    let clamp = |x: f32| x.round().clamp(0.0, 255.0) as u8;
    [clamp(c + 1.596 * e), clamp(c - 0.392 * d - 0.813 * e), clamp(c + 2.017 * d)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yuyv_grey_and_colors() {
        // Mid grey: Y=126, U=V=128 gives roughly 128 on every channel.
        let img = RgbImage::from_yuyv(2, 1, &[126, 128, 126, 128]);
        for c in &img.data {
            assert!((*c as i32 - 128).abs() <= 1, "{:?}", img.data);
        }
        // Limited-range black and white.
        let img = RgbImage::from_yuyv(2, 1, &[16, 128, 235, 128]);
        assert_eq!(&img.data[0..3], &[0, 0, 0]);
        assert_eq!(&img.data[3..6], &[255, 255, 255]);
        // Strong V pushes red up and green down.
        let img = RgbImage::from_yuyv(2, 1, &[81, 90, 81, 240]);
        assert!(img.data[0] > 200 && img.data[1] < 60, "{:?}", &img.data[0..3]);
    }

    #[test]
    fn sample_hits_pixel_centers_and_interpolates() {
        let img = RgbImage::from_fn(2, 1, |x, _| if x == 0 { [0, 0, 0] } else { [200, 100, 50] });
        assert_eq!(img.sample(0.5, 0.5), [0.0, 0.0, 0.0]);
        assert_eq!(img.sample(1.5, 0.5), [200.0, 100.0, 50.0]);
        let mid = img.sample(1.0, 0.5);
        assert!((mid[0] - 100.0).abs() < 1e-4 && (mid[2] - 25.0).abs() < 1e-4);
        // Outside the image fades to black.
        assert_eq!(img.sample(-5.0, -5.0), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn mean_luma_of_region() {
        let img = RgbImage::from_fn(4, 4, |x, _| if x < 2 { [0, 0, 0] } else { [255, 255, 255] });
        assert_eq!(img.mean_luma(0.0, 0.0, 2.0, 4.0), 0.0);
        assert_eq!(img.mean_luma(2.0, 0.0, 4.0, 4.0), 255.0);
        assert!((img.mean_luma(0.0, 0.0, 4.0, 4.0) - 127.5).abs() < 0.01);
        assert_eq!(img.mean_luma(10.0, 10.0, 20.0, 20.0), 0.0);
    }
}
