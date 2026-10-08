//! Finding the model files and running them with ONNX Runtime.
//!
//! Both models are MediaPipe's own (Apache-2.0), converted from the TFLite
//! files in face_landmarker.task to ONNX. `scripts/fetch-models.sh` downloads
//! pinned copies and checks their SHA-256.

use std::env;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;

use crate::image::RgbImage;
use crate::vision::detector::{self, Anchor, Detection, Letterbox};
use crate::vision::{FaceDetector, FaceMesh, MESH_INPUT_SIZE, MeshOutput, NUM_LANDMARKS, Roi};

pub const DETECTOR_FILE: &str = "face_detector.onnx";
pub const MESH_FILE: &str = "face_landmarks.onnx";

/// Places to look for the model files, most specific first.
pub fn model_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(d) = env::var_os("LOOKFOCUS_MODEL_DIR") {
        dirs.push(PathBuf::from(d));
    }
    let data_home = env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")));
    if let Some(d) = data_home {
        dirs.push(d.join("lookfocus/models"));
    }
    dirs.push(PathBuf::from("/usr/share/lookfocus/models"));
    dirs
}

/// Returns the directory holding both model files, or an error that says
/// where we looked and how to fix it.
pub fn find_model_dir(explicit: Option<&Path>) -> Result<PathBuf> {
    let candidates = match explicit {
        Some(p) => vec![p.to_path_buf()],
        None => model_dirs(),
    };
    for dir in &candidates {
        if dir.join(DETECTOR_FILE).is_file() && dir.join(MESH_FILE).is_file() {
            return Ok(dir.clone());
        }
    }
    let looked: Vec<String> = candidates.iter().map(|d| format!("  {}", d.display())).collect();
    bail!(
        "model files not found. Looked for {DETECTOR_FILE} and {MESH_FILE} in:\n{}\n\
         Download them with: scripts/fetch-models.sh",
        looked.join("\n")
    )
}

/// Loads the system ONNX Runtime library. Call once before creating models.
///
/// Uses `ORT_DYLIB_PATH` if set, otherwise the library Arch's
/// `onnxruntime-cpu` package installs.
pub fn init_onnxruntime() -> Result<()> {
    let path = env::var_os("ORT_DYLIB_PATH").map(PathBuf::from).unwrap_or_else(|| {
        let system = PathBuf::from("/usr/lib/libonnxruntime.so");
        if system.exists() { system } else { PathBuf::from("libonnxruntime.so") }
    });
    let builder = ort::init_from(&path).map_err(|e| {
        anyhow!(
            "could not load ONNX Runtime from {}: {e}\n\
             On Arch, install it with: sudo pacman -S onnxruntime-cpu\n\
             Or point ORT_DYLIB_PATH at libonnxruntime.so.",
            path.display()
        )
    })?;
    builder.commit();
    Ok(())
}

fn session(path: &Path, threads: usize) -> Result<Session> {
    let build = || -> ort::Result<Session> {
        Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .with_inter_threads(1)?
            .commit_from_file(path)
    };
    build().with_context(|| format!("loading model {}", path.display()))
}

/// BlazeFace short-range detector.
pub struct OnnxDetector {
    session: Session,
    anchors: Vec<Anchor>,
    input: Vec<f32>,
}

impl OnnxDetector {
    pub fn load(dir: &Path, threads: usize) -> Result<Self> {
        let size = detector::INPUT_SIZE;
        Ok(Self {
            session: session(&dir.join(DETECTOR_FILE), threads)?,
            anchors: detector::anchors(),
            input: vec![0.0; size * size * 3],
        })
    }
}

impl FaceDetector for OnnxDetector {
    fn detect(&mut self, frame: &RgbImage) -> Result<Vec<Detection>> {
        let size = detector::INPUT_SIZE;
        // Letterbox: a square region as large as the longer side, centered,
        // so the frame keeps its aspect ratio and the rest is black.
        let side = frame.width.max(frame.height) as f32;
        let roi =
            Roi { cx: frame.width as f32 / 2.0, cy: frame.height as f32 / 2.0, width: side, height: side, angle: 0.0 };
        // The detector expects RGB in [-1, 1].
        roi.crop_into(frame, size, 2.0 / 255.0, -1.0, &mut self.input);
        let outputs =
            self.session.run(ort::inputs![TensorRef::from_array_view(([1usize, size, size, 3], &self.input[..]))?])?;
        let (_, regressors) = outputs["regressors"].try_extract_tensor::<f32>()?;
        let (_, scores) = outputs["classificators"].try_extract_tensor::<f32>()?;
        if regressors.len() != detector::NUM_ANCHORS * detector::NUM_COORDS || scores.len() != detector::NUM_ANCHORS {
            bail!("face detector returned unexpected output sizes");
        }
        Ok(detector::decode(regressors, scores, &self.anchors, Letterbox::for_frame(frame.width, frame.height)))
    }
}

/// MediaPipe Face Mesh V2 landmark model (478 points).
pub struct OnnxFaceMesh {
    session: Session,
    input: Vec<f32>,
}

impl OnnxFaceMesh {
    pub fn load(dir: &Path, threads: usize) -> Result<Self> {
        Ok(Self {
            session: session(&dir.join(MESH_FILE), threads)?,
            input: vec![0.0; MESH_INPUT_SIZE * MESH_INPUT_SIZE * 3],
        })
    }
}

impl FaceMesh for OnnxFaceMesh {
    fn run(&mut self, frame: &RgbImage, roi: &Roi) -> Result<MeshOutput> {
        let size = MESH_INPUT_SIZE;
        // The landmark model expects RGB in [0, 1].
        roi.crop_into(frame, size, 1.0 / 255.0, 0.0, &mut self.input);
        let outputs =
            self.session.run(ort::inputs![TensorRef::from_array_view(([1usize, size, size, 3], &self.input[..]))?])?;
        // "Identity" holds the landmarks and "Identity_1" the face presence
        // logit, as in the original TFLite model.
        let (_, flat) = outputs["Identity"].try_extract_tensor::<f32>()?;
        let (_, presence) = outputs["Identity_1"].try_extract_tensor::<f32>()?;
        if flat.len() != NUM_LANDMARKS * 3 || presence.is_empty() {
            bail!("face mesh returned unexpected output sizes");
        }
        let points = flat.as_chunks::<3>().0.to_vec();
        Ok(MeshOutput { points, presence: detector::sigmoid(presence[0]) })
    }
}
