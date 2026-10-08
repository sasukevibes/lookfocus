//! Camera capture.
//!
//! `V4lCamera` reads YUYV frames straight from a V4L2 device. YUYV needs no
//! JPEG decoding, which keeps CPU use low. Webcams usually only offer 30 fps,
//! so frames are read at the camera's rate and skipped to reach the target
//! rate. In dim light many webcams lengthen exposure and slow down on their
//! own (20 fps on the development laptop), so the skipping adapts to the rate
//! frames really arrive at.
//!
//! Frames are converted to RGB in memory and dropped after use. Nothing is
//! ever written to disk.

use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use thiserror::Error;
use v4l::buffer::Type;
use v4l::io::mmap::Stream as MmapStream;
use v4l::io::traits::CaptureStream;
use v4l::video::Capture;
use v4l::{Device, Format, FourCC};

use crate::image::RgbImage;

/// One captured frame.
#[derive(Clone, Debug)]
pub struct Frame {
    pub image: RgbImage,
    pub time: Instant,
}

pub trait FrameSource {
    /// Blocks until the next frame at the source's target rate.
    fn next_frame(&mut self) -> Result<Frame, CameraError>;

    /// Changes the target rate. Sources without rate control ignore it.
    fn set_fps(&mut self, _fps: f32) {}
}

#[derive(Debug, Error)]
pub enum CameraError {
    #[error("camera {0} not found")]
    NotFound(PathBuf),
    #[error("camera {0} is busy. Another app (a video call?) is using it")]
    Busy(PathBuf),
    #[error("no permission to open camera {0}")]
    PermissionDenied(PathBuf),
    #[error("camera {path} does not offer YUYV at {width}x{height} (it offered {offered})")]
    UnsupportedFormat { path: PathBuf, width: u32, height: u32, offered: String },
    #[error("camera {path}: {source}")]
    Io { path: PathBuf, source: io::Error },
    #[error("camera stream ended")]
    Ended,
}

impl CameraError {
    fn from_io(path: &Path, e: io::Error) -> Self {
        let path = path.to_path_buf();
        match e.raw_os_error() {
            Some(libc_codes::ENOENT) | Some(libc_codes::ENODEV) | Some(libc_codes::ENXIO) => Self::NotFound(path),
            Some(libc_codes::EBUSY) => Self::Busy(path),
            Some(libc_codes::EACCES) | Some(libc_codes::EPERM) => Self::PermissionDenied(path),
            _ => Self::Io { path, source: e },
        }
    }
}

/// The Linux errno values we care about. They are fixed by the kernel ABI.
mod libc_codes {
    pub const EPERM: i32 = 1;
    pub const ENOENT: i32 = 2;
    pub const ENXIO: i32 = 6;
    pub const EACCES: i32 = 13;
    pub const EBUSY: i32 = 16;
    pub const ENODEV: i32 = 19;
}

pub struct V4lCamera {
    // The stream must be dropped before the device, and Rust drops fields in
    // declaration order.
    stream: MmapStream<'static>,
    _device: Device,
    path: PathBuf,
    width: usize,
    height: usize,
    stride: usize,
    picker: FramePicker,
}

impl V4lCamera {
    pub fn open(path: &Path, width: u32, height: u32, fps: f32) -> Result<Self, CameraError> {
        let err = |e| CameraError::from_io(path, e);
        let device = Device::with_path(path).map_err(err)?;
        let want = Format::new(width, height, FourCC::new(b"YUYV"));
        let got = device.set_format(&want).map_err(err)?;
        if got.fourcc != want.fourcc {
            return Err(CameraError::UnsupportedFormat {
                path: path.into(),
                width,
                height,
                offered: got.fourcc.to_string(),
            });
        }
        let stream = MmapStream::with_buffers(&device, Type::VideoCapture, 4).map_err(err)?;
        Ok(Self {
            stream,
            _device: device,
            path: path.into(),
            width: got.width as usize,
            height: got.height as usize,
            stride: (got.stride as usize).max(got.width as usize * 2),
            picker: FramePicker::new(fps),
        })
    }

    pub fn size(&self) -> (usize, usize) {
        (self.width, self.height)
    }
}

impl FrameSource for V4lCamera {
    fn next_frame(&mut self) -> Result<Frame, CameraError> {
        loop {
            let (buf, _) = self.stream.next().map_err(|e| CameraError::from_io(&self.path, e))?;
            let now = Instant::now();
            if !self.picker.offer(now) {
                continue;
            }
            let row = self.width * 2;
            let image = if self.stride == row {
                RgbImage::from_yuyv(self.width, self.height, buf)
            } else {
                let packed: Vec<u8> =
                    buf.chunks(self.stride).take(self.height).flat_map(|r| &r[..row.min(r.len())]).copied().collect();
                RgbImage::from_yuyv(self.width, self.height, &packed)
            };
            return Ok(Frame { image, time: now });
        }
    }

    fn set_fps(&mut self, fps: f32) {
        self.picker.set_fps(fps);
    }
}

/// Decides which camera frames to keep to get close to a target rate.
///
/// A frame is kept when it is the one closest to the next due time, judged
/// from how far apart frames have recently been arriving. With a 30 fps
/// camera and a 15 fps target it keeps every other frame. With a camera that
/// has slowed to 20 fps it keeps every frame, which is closer to 15 than 10.
#[derive(Clone, Debug)]
pub struct FramePicker {
    period: Duration,
    interval: Option<Duration>,
    last_arrival: Option<Instant>,
    last_taken: Option<Instant>,
}

impl FramePicker {
    pub fn new(fps: f32) -> Self {
        Self {
            period: Duration::from_secs_f32(1.0 / fps.max(0.1)),
            interval: None,
            last_arrival: None,
            last_taken: None,
        }
    }

    pub fn set_fps(&mut self, fps: f32) {
        self.period = Duration::from_secs_f32(1.0 / fps.max(0.1));
    }

    /// Reports a frame arriving at `now`. Returns true if it should be used.
    pub fn offer(&mut self, now: Instant) -> bool {
        if let Some(prev) = self.last_arrival {
            let dt = now.duration_since(prev);
            self.interval = Some(match self.interval {
                Some(i) => i.mul_f32(0.8) + dt.mul_f32(0.2),
                None => dt,
            });
        }
        self.last_arrival = Some(now);
        let keep = match self.last_taken {
            None => true,
            Some(taken) => now.duration_since(taken) + self.interval.unwrap_or_default() / 2 >= self.period,
        };
        if keep {
            self.last_taken = Some(now);
        }
        keep
    }
}

/// A frame source that replays prepared images, for tests.
pub struct MockCamera {
    frames: std::vec::IntoIter<RgbImage>,
    start: Instant,
    step: Duration,
    count: u32,
}

impl MockCamera {
    pub fn new(frames: Vec<RgbImage>, fps: f32) -> Self {
        let step = Duration::from_nanos((1e9 / fps as f64).round() as u64);
        Self { frames: frames.into_iter(), start: Instant::now(), step, count: 0 }
    }
}

impl FrameSource for MockCamera {
    fn next_frame(&mut self) -> Result<Frame, CameraError> {
        let image = self.frames.next().ok_or(CameraError::Ended)?;
        let time = self.start + self.step * self.count;
        self.count += 1;
        Ok(Frame { image, time })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errno_mapping() {
        let p = Path::new("/dev/video9");
        let e = |code| CameraError::from_io(p, io::Error::from_raw_os_error(code));
        assert!(matches!(e(16), CameraError::Busy(_)));
        assert!(matches!(e(2), CameraError::NotFound(_)));
        assert!(matches!(e(19), CameraError::NotFound(_)));
        assert!(matches!(e(13), CameraError::PermissionDenied(_)));
        assert!(matches!(e(5), CameraError::Io { .. }));
    }

    #[test]
    fn missing_device_is_not_found() {
        let r = V4lCamera::open(Path::new("/dev/lookfocus-no-such-camera"), 640, 480, 15.0);
        assert!(matches!(r, Err(CameraError::NotFound(_))), "{:?}", r.err());
    }

    fn kept_rate(camera_fps: f32, target_fps: f32) -> f32 {
        let mut p = FramePicker::new(target_fps);
        let start = Instant::now();
        let step = Duration::from_secs_f32(1.0 / camera_fps);
        let n = (camera_fps * 10.0) as u32;
        let kept = (0..n).filter(|&i| p.offer(start + step * i)).count();
        kept as f32 / 10.0
    }

    #[test]
    fn picker_halves_a_30fps_camera() {
        let r = kept_rate(30.0, 15.0);
        assert!((r - 15.0).abs() <= 0.2, "{r}");
    }

    #[test]
    fn picker_keeps_every_frame_of_a_slowed_camera() {
        // 20 fps in: keeping every frame (20) is closer to 15 than every
        // other frame (10).
        let r = kept_rate(20.0, 15.0);
        assert!(r >= 19.5, "{r}");
        // And a 10 fps camera is never throttled further.
        assert!(kept_rate(10.0, 15.0) >= 9.8);
    }

    #[test]
    fn picker_handles_a_60fps_camera() {
        let r = kept_rate(60.0, 15.0);
        assert!((r - 15.0).abs() <= 0.2, "{r}");
    }

    #[test]
    fn mock_replays_frames_with_steady_timestamps() {
        let mut cam = MockCamera::new(vec![RgbImage::new(4, 4), RgbImage::new(4, 4)], 10.0);
        let a = cam.next_frame().unwrap();
        let b = cam.next_frame().unwrap();
        assert_eq!(b.time - a.time, Duration::from_millis(100));
        assert!(matches!(cam.next_frame(), Err(CameraError::Ended)));
    }
}
