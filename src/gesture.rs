//! Hand gestures from the 21 hand landmarks, by geometry alone.
//!
//! A finger counts as extended when its tip is farther from the wrist than its
//! middle joint (PIP). A curled finger folds its tip back toward the palm, so
//! the tip ends up closer to the wrist than the joint. The thumb bends across
//! the palm instead of toward the wrist, so it has its own rule: it is
//! extended when its tip is farther from the wrist than its last joint (IP),
//! and also farther from the base of the little finger than that joint. A
//! thumb folded over the palm moves its tip toward the little finger and
//! fails the second test.
//!
//! These rules compare distances on the same hand, so they do not depend on
//! how large the hand looks, which way it points, or which hand it is.
//! Distances use only x and y. Landmark depth is the least reliable of the
//! three, and an open palm faces the camera, which puts its fingers across the
//! image anyway.
//!
//! Only the open palm (all five fingers extended) is defined so far.

use serde::{Deserialize, Serialize};

use crate::vision::hand::{
    INDEX_PIP, INDEX_TIP, MIDDLE_PIP, MIDDLE_TIP, NUM_HAND_LANDMARKS, PINKY_MCP, PINKY_PIP, PINKY_TIP, RING_PIP,
    RING_TIP, THUMB_IP, THUMB_TIP, WRIST,
};

/// Config files name gestures in snake_case, for example `open_palm`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Gesture {
    /// All five fingers extended.
    OpenPalm,
}

/// Which fingers are extended.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Fingers {
    pub thumb: bool,
    pub index: bool,
    pub middle: bool,
    pub ring: bool,
    pub pinky: bool,
}

impl Fingers {
    /// Reads finger states from hand landmarks in MediaPipe order. Returns
    /// None if there are fewer than 21 landmarks.
    pub fn from_landmarks(points: &[[f32; 3]]) -> Option<Self> {
        if points.len() < NUM_HAND_LANDMARKS {
            return None;
        }
        let dist = |a: usize, b: usize| (points[a][0] - points[b][0]).hypot(points[a][1] - points[b][1]);
        let finger = |pip: usize, tip: usize| dist(tip, WRIST) > dist(pip, WRIST);
        Some(Self {
            thumb: finger(THUMB_IP, THUMB_TIP) && dist(THUMB_TIP, PINKY_MCP) > dist(THUMB_IP, PINKY_MCP),
            index: finger(INDEX_PIP, INDEX_TIP),
            middle: finger(MIDDLE_PIP, MIDDLE_TIP),
            ring: finger(RING_PIP, RING_TIP),
            pinky: finger(PINKY_PIP, PINKY_TIP),
        })
    }

    pub fn count(&self) -> usize {
        [self.thumb, self.index, self.middle, self.ring, self.pinky].iter().filter(|&&f| f).count()
    }
}

/// Names the gesture the landmarks show, if it is one we know.
pub fn classify(points: &[[f32; 3]]) -> Option<Gesture> {
    let fingers = Fingers::from_landmarks(points)?;
    (fingers.count() == 5).then_some(Gesture::OpenPalm)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An upright open hand in image pixels: wrist at the bottom, fingers up,
    /// thumb out to the left.
    fn open_hand() -> Vec<[f32; 3]> {
        let mut p = vec![[0.0, 0.0, 0.0]; NUM_HAND_LANDMARKS];
        p[WRIST] = [100.0, 200.0, 0.0];
        // Thumb: CMC, MCP, IP, tip.
        p[1] = [80.0, 185.0, 0.0];
        p[2] = [65.0, 170.0, 0.0];
        p[3] = [55.0, 155.0, 0.0];
        p[4] = [45.0, 140.0, 0.0];
        // Index to pinky: MCP, PIP, DIP, tip, each column straight up.
        for (f, x) in [80.0, 95.0, 110.0, 125.0].into_iter().enumerate() {
            for (j, y) in [150.0, 120.0, 100.0, 80.0].into_iter().enumerate() {
                p[5 + 4 * f + j] = [x, y, 0.0];
            }
        }
        p
    }

    /// Curls the four fingers so each tip folds back below its PIP joint.
    fn curl_fingers(p: &mut [[f32; 3]]) {
        for f in 0..4 {
            let x = p[5 + 4 * f][0];
            p[6 + 4 * f] = [x, 130.0, 0.0];
            p[7 + 4 * f] = [x, 140.0, 0.0];
            p[8 + 4 * f] = [x, 160.0, 0.0];
        }
    }

    /// Folds the thumb across the palm, tip near the middle finger's base.
    fn fold_thumb(p: &mut [[f32; 3]]) {
        p[3] = [75.0, 160.0, 0.0];
        p[4] = [100.0, 155.0, 0.0];
    }

    /// Rotates by `angle` about the origin, scales and moves the points.
    fn place(p: &[[f32; 3]], angle: f32, scale: f32, dx: f32) -> Vec<[f32; 3]> {
        let (sin, cos) = angle.sin_cos();
        p.iter().map(|q| [scale * (cos * q[0] - sin * q[1]) + dx, scale * (sin * q[0] + cos * q[1]), q[2]]).collect()
    }

    #[test]
    fn open_hand_is_an_open_palm() {
        let p = open_hand();
        assert_eq!(Fingers::from_landmarks(&p).unwrap().count(), 5);
        assert_eq!(classify(&p), Some(Gesture::OpenPalm));
    }

    #[test]
    fn fist_has_no_fingers_extended() {
        let mut p = open_hand();
        curl_fingers(&mut p);
        fold_thumb(&mut p);
        assert_eq!(Fingers::from_landmarks(&p), Some(Fingers::default()));
        assert_eq!(classify(&p), None);
    }

    #[test]
    fn folded_thumb_is_not_an_open_palm() {
        let mut p = open_hand();
        fold_thumb(&mut p);
        let f = Fingers::from_landmarks(&p).unwrap();
        assert!(!f.thumb && f.index && f.middle && f.ring && f.pinky);
        assert_eq!(classify(&p), None);
    }

    #[test]
    fn one_curled_finger_is_not_an_open_palm() {
        let mut p = open_hand();
        p[RING_PIP] = [110.0, 130.0, 0.0];
        p[RING_TIP] = [110.0, 160.0, 0.0];
        let f = Fingers::from_landmarks(&p).unwrap();
        assert!(!f.ring && f.count() == 4);
        assert_eq!(classify(&p), None);
    }

    #[test]
    fn rotation_size_and_side_do_not_matter() {
        let open = open_hand();
        let mut fist = open_hand();
        curl_fingers(&mut fist);
        fold_thumb(&mut fist);
        for angle in [0.5f32, 2.0, -1.2, 3.0] {
            for scale in [0.4f32, 2.5] {
                assert_eq!(classify(&place(&open, angle, scale, 300.0)), Some(Gesture::OpenPalm));
                assert_eq!(classify(&place(&fist, angle, scale, 300.0)), None);
            }
        }
        // The other hand: mirror left to right.
        let mirrored: Vec<_> = open.iter().map(|q| [640.0 - q[0], q[1], q[2]]).collect();
        assert_eq!(classify(&mirrored), Some(Gesture::OpenPalm));
    }

    #[test]
    fn gesture_names_are_snake_case() {
        assert_eq!(serde_json::to_string(&Gesture::OpenPalm).unwrap(), "\"open_palm\"");
        assert_eq!(serde_json::from_str::<Gesture>("\"open_palm\"").unwrap(), Gesture::OpenPalm);
        assert!(serde_json::from_str::<Gesture>("\"OpenPalm\"").is_err());
    }

    #[test]
    fn too_few_landmarks() {
        assert_eq!(classify(&open_hand()[..20]), None);
        assert_eq!(Fingers::from_landmarks(&[]), None);
    }
}
