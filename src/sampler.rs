//! Camera frame in, head pose sample out.
//!
//! This joins the camera, the face tracker and the pose fit into one step so
//! the CLI commands and the daemon all read poses the same way. When hand
//! tracking is switched on and the hand models are installed, it also tracks a
//! hand and names its gesture. Hand tracking is off until asked for, and the
//! hand models are loaded only then, so they cost nothing for people who do
//! not use gestures.

use std::time::{Duration, Instant};

use anyhow::Result;

use crate::camera::FrameSource;
use crate::gesture::{self, Gesture};
use crate::image::RgbImage;
use crate::pose::{self, HeadPose};
use crate::vision::{FaceDetector, FaceMesh, FaceTracker, HandSource};

/// What one frame produced.
#[derive(Clone, Debug)]
pub struct Sample {
    pub time: Instant,
    pub face: Option<FaceSample>,
    /// The tracked hand. Always None when the hand models are not installed.
    pub hand: Option<HandSample>,
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

#[derive(Clone, Debug, PartialEq)]
pub struct HandSample {
    /// The gesture the hand shows, if it is one we know.
    pub gesture: Option<Gesture>,
    pub confidence: f32,
    /// The model's handedness score, 0 to 1. Which side it means is not
    /// checked yet (see NOTES.md).
    pub handedness: f32,
}

/// Anything that produces pose samples: the live sampler, or scripted poses
/// in tests.
pub trait PoseSource {
    fn sample(&mut self) -> Result<Sample>;

    /// Changes the target sample rate, where the source supports it.
    fn set_fps(&mut self, _fps: f32) {}

    /// Turns hand tracking on or off, where the source supports it. Samples
    /// carry no hand while it is off.
    fn set_hands(&mut self, _on: bool) {}
}

/// Loads the hand tracker when hand tracking is first switched on. Returns
/// None if the hand models are missing or fail to load.
type HandLoader = Box<dyn FnMut() -> Option<Box<dyn HandSource>>>;

pub struct Sampler<C, D, M> {
    camera: C,
    tracker: FaceTracker<D, M>,
    /// The loaded hand tracker. None when hand tracking is off, when the hand
    /// models are missing, or after a hand model error.
    hands: Option<Box<dyn HandSource>>,
    /// Whether hand tracking is switched on. Stays true after a hand model
    /// error, so the tracker is not loaded again and again.
    hands_on: bool,
    /// Where `set_hands(true)` gets a tracker from. None for a sampler given
    /// its tracker up front.
    hand_loader: Option<HandLoader>,
}

impl<C: FrameSource, D: FaceDetector, M: FaceMesh> Sampler<C, D, M> {
    pub fn new(camera: C, tracker: FaceTracker<D, M>) -> Self {
        Self { camera, tracker, hands: None, hands_on: false, hand_loader: None }
    }

    /// Adds hand tracking that is on from the start. None leaves hand samples
    /// empty, which is what a missing set of hand models gives.
    pub fn with_hands(mut self, hands: Option<impl HandSource + 'static>) -> Self {
        self.hands = hands.map(|h| Box::new(h) as Box<dyn HandSource>);
        self.hands_on = true;
        self
    }

    /// Adds hand tracking that starts off. `set_hands(true)` calls `load` to
    /// get the tracker, and `set_hands(false)` drops it again.
    pub fn with_hand_loader<H: HandSource + 'static>(mut self, mut load: impl FnMut() -> Option<H> + 'static) -> Self {
        self.hand_loader = Some(Box::new(move || load().map(|h| Box::new(h) as Box<dyn HandSource>)));
        self
    }

    /// Forgets the tracked face and hand, for example after a pause.
    pub fn reset(&mut self) {
        self.tracker.reset();
        if let Some(hands) = &mut self.hands {
            hands.reset();
        }
    }

    /// Tracks the hand and names its gesture. A hand model error turns hand
    /// tracking off for this sampler instead of failing the sample, so face
    /// tracking keeps working whatever the hand models do.
    fn track_hand(&mut self, image: &RgbImage) -> Option<HandSample> {
        if !self.hands_on {
            return None;
        }
        let hands = self.hands.as_mut()?;
        match hands.process(image) {
            Ok(hand) => hand.map(|h| HandSample {
                gesture: gesture::classify(&h.landmarks),
                confidence: h.confidence,
                handedness: h.handedness,
            }),
            Err(e) => {
                log::warn!("hand tracking failed and is now off: {e:#}");
                self.hands = None;
                None
            }
        }
    }
}

impl<C: FrameSource, D: FaceDetector, M: FaceMesh> PoseSource for Sampler<C, D, M> {
    fn set_fps(&mut self, fps: f32) {
        self.camera.set_fps(fps);
    }

    fn set_hands(&mut self, on: bool) {
        if on == self.hands_on {
            return;
        }
        self.hands_on = on;
        if on {
            if self.hands.is_none() {
                self.hands = self.hand_loader.as_mut().and_then(|load| load());
            }
        } else if self.hand_loader.is_some() {
            // Free the models. Switching back on loads them again.
            self.hands = None;
        } else if let Some(hands) = &mut self.hands {
            hands.reset();
        }
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
        let hand = self.track_hand(&frame.image);
        Ok(Sample { time: frame.time, face, hand, capture: t1 - t0, inference: t1.elapsed() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::MockCamera;
    use crate::vision::detector::Detection;
    use crate::vision::hand::{NUM_HAND_LANDMARKS, WRIST};
    use crate::vision::{Hand, MeshOutput, Roi};
    use anyhow::bail;
    use std::cell::Cell;
    use std::rc::Rc;

    /// Face models that never find a face, since these tests are about hands.
    struct NoFace;

    impl FaceDetector for NoFace {
        fn detect(&mut self, _: &RgbImage) -> Result<Vec<Detection>> {
            Ok(Vec::new())
        }
    }

    impl FaceMesh for NoFace {
        fn run(&mut self, _: &RgbImage, _: &Roi) -> Result<MeshOutput> {
            bail!("no face to read")
        }
    }

    /// A hand tracker that returns the given landmarks every frame, or fails.
    struct FakeHands {
        landmarks: Option<Vec<[f32; 3]>>,
    }

    impl HandSource for FakeHands {
        fn process(&mut self, _: &RgbImage) -> Result<Option<Hand>> {
            let Some(landmarks) = self.landmarks.clone() else { bail!("hand model broke") };
            let roi = Roi { cx: 0.0, cy: 0.0, width: 1.0, height: 1.0, angle: 0.0 };
            Ok(Some(Hand { landmarks, confidence: 0.9, handedness: 0.2, roi, tracked: true }))
        }

        fn reset(&mut self) {}
    }

    /// An upright hand with every finger pointing straight up from the wrist,
    /// tips farthest out: an open palm.
    fn open_palm() -> Vec<[f32; 3]> {
        let mut p = vec![[0.0, 0.0, 0.0]; NUM_HAND_LANDMARKS];
        p[WRIST] = [100.0, 200.0, 0.0];
        for (f, x) in [50.0, 80.0, 95.0, 110.0, 125.0].into_iter().enumerate() {
            for (j, y) in [170.0, 150.0, 120.0, 90.0].into_iter().enumerate() {
                p[1 + 4 * f + j] = [x, y, 0.0];
            }
        }
        p
    }

    fn sampler() -> Sampler<MockCamera, NoFace, NoFace> {
        let frames = vec![RgbImage::new(64, 48); 6];
        Sampler::new(MockCamera::new(frames, 15.0), FaceTracker::new(NoFace, NoFace))
    }

    #[test]
    fn reports_the_gesture_from_the_hand_tracker() {
        let mut s = sampler().with_hands(Some(FakeHands { landmarks: Some(open_palm()) }));
        let sample = s.sample().unwrap();
        assert!(sample.face.is_none());
        let hand = sample.hand.unwrap();
        assert_eq!(hand.gesture, Some(Gesture::OpenPalm));
        assert_eq!((hand.confidence, hand.handedness), (0.9, 0.2));
    }

    #[test]
    fn a_hand_without_a_known_gesture_has_none() {
        let mut fist = open_palm();
        for f in 1..5 {
            fist[4 * f + 4] = fist[4 * f + 1]; // each tip back down at its knuckle
        }
        let mut s = sampler().with_hands(Some(FakeHands { landmarks: Some(fist) }));
        assert_eq!(s.sample().unwrap().hand.unwrap().gesture, None);
    }

    #[test]
    fn no_hand_models_means_no_hand() {
        let mut s = sampler().with_hands(None::<FakeHands>);
        assert!(s.sample().unwrap().hand.is_none());
        let mut s = sampler();
        assert!(s.sample().unwrap().hand.is_none());
    }

    #[test]
    fn hands_load_only_when_switched_on_and_are_dropped_when_switched_off() {
        let loads = Rc::new(Cell::new(0));
        let counter = loads.clone();
        let mut s = sampler().with_hand_loader(move || {
            counter.set(counter.get() + 1);
            Some(FakeHands { landmarks: Some(open_palm()) })
        });
        // Off until asked: nothing loaded, no hand in the sample.
        assert!(s.sample().unwrap().hand.is_none());
        assert_eq!(loads.get(), 0);
        s.set_hands(true);
        s.set_hands(true);
        assert_eq!(loads.get(), 1);
        assert_eq!(s.sample().unwrap().hand.unwrap().gesture, Some(Gesture::OpenPalm));
        s.set_hands(false);
        assert!(s.hands.is_none(), "the models are freed");
        assert!(s.sample().unwrap().hand.is_none());
        s.set_hands(true);
        assert_eq!(loads.get(), 2);
        assert!(s.sample().unwrap().hand.is_some());
    }

    #[test]
    fn missing_hand_models_leave_hands_empty_without_retrying() {
        let loads = Rc::new(Cell::new(0));
        let counter = loads.clone();
        let mut s = sampler().with_hand_loader(move || {
            counter.set(counter.get() + 1);
            None::<FakeHands>
        });
        s.set_hands(true);
        s.set_hands(true);
        assert!(s.sample().unwrap().hand.is_none());
        assert_eq!(loads.get(), 1);
    }

    #[test]
    fn a_hand_model_error_is_not_followed_by_a_reload() {
        let loads = Rc::new(Cell::new(0));
        let counter = loads.clone();
        let mut s = sampler().with_hand_loader(move || {
            counter.set(counter.get() + 1);
            Some(FakeHands { landmarks: None })
        });
        s.set_hands(true);
        assert!(s.sample().unwrap().hand.is_none());
        s.set_hands(true);
        assert!(s.sample().unwrap().hand.is_none());
        assert_eq!(loads.get(), 1);
    }

    #[test]
    fn a_failing_hand_model_turns_hands_off_but_keeps_sampling() {
        let mut s = sampler().with_hands(Some(FakeHands { landmarks: None }));
        assert!(s.sample().unwrap().hand.is_none());
        assert!(s.hands.is_none());
        assert!(s.sample().is_ok());
    }
}
