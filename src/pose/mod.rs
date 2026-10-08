//! Head pose from face landmarks.
//!
//! The 3D landmarks are matched against MediaPipe's canonical face with a
//! weighted rigid fit (Procrustes, solved with the Kabsch method). The fitted
//! rotation tells us which way the face points.
//!
//! The fit treats the projection as orthographic. That makes the absolute
//! angles a little different from a full perspective solve, but they are
//! consistent from frame to frame, and calibration only compares poses with
//! each other.

mod canonical;

use nalgebra::{Matrix3, Vector3};

pub use canonical::PROCRUSTES_POINTS;

/// Head orientation in degrees.
///
/// - `yaw` is positive when the face turns toward the person's own left, which
///   is toward the right side of an unmirrored camera image.
/// - `pitch` is positive when the face tilts up.
/// - `roll` is positive when the head tilts toward the person's own right
///   shoulder, which looks counterclockwise in the camera image.
///
/// All three are zero when facing the camera squarely.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeadPose {
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
}

/// Estimates head pose from 3D landmarks in image coordinates: x to the right
/// and y down in pixels, z in the same units with smaller values closer to the
/// camera. This is the layout the face mesh model produces after its crop is
/// mapped back to the full image.
///
/// Returns `None` if there are too few landmarks or the fit is degenerate.
pub fn estimate(landmarks: &[[f32; 3]]) -> Option<HeadPose> {
    let r = fit_rotation(landmarks)?;
    Some(angles(&r))
}

/// Rotation that best maps the canonical face onto the landmarks, in a camera
/// frame with x right, y up and z toward the viewer.
pub fn fit_rotation(landmarks: &[[f32; 3]]) -> Option<Matrix3<f32>> {
    let mut src = Vec::with_capacity(PROCRUSTES_POINTS.len());
    let mut dst = Vec::with_capacity(PROCRUSTES_POINTS.len());
    let mut weights = Vec::with_capacity(PROCRUSTES_POINTS.len());
    for &(index, weight, c) in PROCRUSTES_POINTS.iter() {
        let p = landmarks.get(index)?;
        src.push(Vector3::new(c[0], c[1], c[2]));
        // Flip y and z so the image frame matches the canonical frame.
        dst.push(Vector3::new(p[0], -p[1], -p[2]));
        weights.push(weight);
    }
    kabsch(&src, &dst, &weights)
}

/// Weighted Kabsch: the rotation R minimizing sum w_i |R (s_i - s0) - (d_i - d0)|^2.
fn kabsch(src: &[Vector3<f32>], dst: &[Vector3<f32>], w: &[f32]) -> Option<Matrix3<f32>> {
    let total: f32 = w.iter().sum();
    if src.len() < 3 || total <= 0.0 {
        return None;
    }
    let mean = |pts: &[Vector3<f32>]| pts.iter().zip(w).map(|(p, &wi)| p * wi).sum::<Vector3<f32>>() / total;
    let (s0, d0) = (mean(src), mean(dst));
    let mut h = Matrix3::zeros();
    for ((s, d), &wi) in src.iter().zip(dst).zip(w) {
        h += (s - s0) * (d - d0).transpose() * wi;
    }
    let svd = h.svd(true, true);
    let (u, v_t) = (svd.u?, svd.v_t?);
    if svd.singular_values[1] < 1e-6 * svd.singular_values[0].max(1e-12) {
        return None;
    }
    let v = v_t.transpose();
    let mut d = Matrix3::identity();
    if (v * u.transpose()).determinant() < 0.0 {
        d[(2, 2)] = -1.0;
    }
    Some(v * d * u.transpose())
}

/// Reads yaw, pitch and roll out of a rotation matrix. The face's forward
/// direction is the third column, and its "up" is the second.
pub fn angles(r: &Matrix3<f32>) -> HeadPose {
    let fwd = r.column(2);
    let yaw = fwd[0].atan2(fwd[2]);
    let pitch = fwd[1].atan2(fwd[0].hypot(fwd[2]));
    let roll = r[(1, 0)].atan2(r[(0, 0)]);
    HeadPose { yaw: yaw.to_degrees(), pitch: pitch.to_degrees(), roll: roll.to_degrees() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Rotation3;

    /// Builds a rotation that turns the face by `yaw` (about the up axis) and
    /// then tilts it by `pitch`, matching the sign conventions of HeadPose.
    fn rotation(yaw: f32, pitch: f32, roll: f32) -> Matrix3<f32> {
        let ry = Rotation3::from_axis_angle(&Vector3::y_axis(), yaw.to_radians());
        let rx = Rotation3::from_axis_angle(&Vector3::x_axis(), -pitch.to_radians());
        let rz = Rotation3::from_axis_angle(&Vector3::z_axis(), roll.to_radians());
        (ry * rx * rz).into_inner()
    }

    /// Renders the canonical face into image coordinates the way the face mesh
    /// model reports landmarks: scaled to pixels, moved to a spot in the frame,
    /// with y and z flipped.
    fn synthetic_landmarks(r: &Matrix3<f32>, scale: f32, center: (f32, f32), noise: f32) -> Vec<[f32; 3]> {
        let mut out = vec![[0.0f32; 3]; 478];
        let mut seed = 12345u32;
        let mut rand = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        for &(index, _, c) in PROCRUSTES_POINTS.iter() {
            let p = r * Vector3::new(c[0], c[1], c[2]) * scale;
            out[index] = [center.0 + p.x + noise * rand(), center.1 - p.y + noise * rand(), -p.z + noise * rand()];
        }
        out
    }

    #[test]
    fn frontal_face_is_zero() {
        let lm = synthetic_landmarks(&Matrix3::identity(), 10.0, (320.0, 240.0), 0.0);
        let pose = estimate(&lm).unwrap();
        assert!(pose.yaw.abs() < 1e-3 && pose.pitch.abs() < 1e-3 && pose.roll.abs() < 1e-3, "{pose:?}");
    }

    #[test]
    fn recovers_known_angles() {
        for &(yaw, pitch, roll) in &[
            (30.0, 0.0, 0.0),
            (-45.0, 0.0, 0.0),
            (0.0, 20.0, 0.0),
            (0.0, -25.0, 0.0),
            (40.0, -10.0, 5.0),
            (-60.0, 15.0, -8.0),
        ] {
            let lm = synthetic_landmarks(&rotation(yaw, pitch, roll), 8.0, (200.0, 150.0), 0.0);
            let pose = estimate(&lm).unwrap();
            assert!((pose.yaw - yaw).abs() < 0.05, "yaw {yaw}: {pose:?}");
            assert!((pose.pitch - pitch).abs() < 0.05, "pitch {pitch}: {pose:?}");
            if roll == 0.0 {
                assert!(pose.roll.abs() < 0.05, "roll: {pose:?}");
            }
        }
    }

    #[test]
    fn sign_conventions_match_the_image() {
        // Turning toward the person's left moves the nose tip (landmark 4)
        // toward the right side of the image, and yaw is positive.
        let lm = synthetic_landmarks(&rotation(30.0, 0.0, 0.0), 10.0, (320.0, 240.0), 0.0);
        assert!(lm[4][0] > 320.0);
        assert!(estimate(&lm).unwrap().yaw > 0.0);
        // Tilting up moves the nose tip up in the image (smaller y).
        let lm = synthetic_landmarks(&rotation(0.0, 20.0, 0.0), 10.0, (320.0, 240.0), 0.0);
        assert!(lm[4][1] < 240.0);
        assert!(estimate(&lm).unwrap().pitch > 0.0);
    }

    #[test]
    fn ignores_scale_and_position() {
        let r = rotation(25.0, -12.0, 0.0);
        let a = estimate(&synthetic_landmarks(&r, 4.0, (50.0, 60.0), 0.0)).unwrap();
        let b = estimate(&synthetic_landmarks(&r, 20.0, (500.0, 400.0), 0.0)).unwrap();
        assert!((a.yaw - b.yaw).abs() < 1e-2 && (a.pitch - b.pitch).abs() < 1e-2);
    }

    #[test]
    fn tolerates_landmark_noise() {
        // About one pixel of noise on a face roughly 150 pixels wide.
        let lm = synthetic_landmarks(&rotation(35.0, 5.0, 0.0), 10.0, (320.0, 240.0), 2.0);
        let pose = estimate(&lm).unwrap();
        assert!((pose.yaw - 35.0).abs() < 2.0 && (pose.pitch - 5.0).abs() < 2.0, "{pose:?}");
    }

    #[test]
    fn rejects_missing_or_degenerate_input() {
        assert!(estimate(&[[0.0; 3]; 10]).is_none());
        assert!(estimate(&[[1.0, 2.0, 3.0]; 478]).is_none());
    }
}
