//! Guided per-monitor recording through the Rust pipeline.
//!
//! This mirrors `prototype/proto.py record` and writes the same JSON layout, so
//! `uv run proto.py analyze <file>` can compare the Rust pose numbers with the
//! Python MediaPipe ones. It is a development check, not part of the CLI. The
//! real calibration flow comes in Phase 3.
//!
//! Only pose numbers are saved. Frames never leave memory.
//!
//! Run: cargo run --release --example record -- [rounds] [output.json]

use std::process::Command;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde_json::{Value, json};

use lookfocus::camera::V4lCamera;
use lookfocus::models::{self, OnnxDetector, OnnxFaceMesh};
use lookfocus::sampler::{PoseSource, Sampler};
use lookfocus::vision::FaceTracker;

const LEAD: Duration = Duration::from_secs(3);
const RECORD: Duration = Duration::from_secs(2);

/// The newest Hyprland instance, for shells that were started without
/// HYPRLAND_INSTANCE_SIGNATURE in their environment.
fn newest_instance() -> Option<String> {
    let runtime = std::env::var_os("XDG_RUNTIME_DIR")?;
    let dir = std::path::Path::new(&runtime).join("hypr");
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.file_name().into_string().ok()?)))
        .max()
        .map(|(_, name)| name)
}

fn monitors() -> Result<Vec<Value>> {
    let mut cmd = Command::new("hyprctl");
    cmd.args(["monitors", "-j"]);
    if std::env::var_os("HYPRLAND_INSTANCE_SIGNATURE").is_none()
        && let Some(sig) = newest_instance()
    {
        cmd.env("HYPRLAND_INSTANCE_SIGNATURE", sig);
    }
    let out = cmd.output().context("running hyprctl")?;
    let mut mons: Vec<Value> = serde_json::from_slice(&out.stdout)
        .with_context(|| format!("parsing hyprctl output: {}", String::from_utf8_lossy(&out.stdout).trim()))?;
    mons.sort_by_key(|m| (m["x"].as_i64().unwrap_or(0), m["y"].as_i64().unwrap_or(0)));
    Ok(mons
        .into_iter()
        .map(|m| json!({"name": m["name"], "x": m["x"], "y": m["y"], "w": m["width"], "h": m["height"], "description": m["description"]}))
        .collect())
}

/// A plain position word for each monitor, so the prompt says where to look.
/// Monitors arrive sorted left to right, then top to bottom.
fn positions(mons: &[Value]) -> Vec<String> {
    let coord = |m: &Value, k: &str| m[k].as_i64().unwrap_or(0);
    let n = mons.len();
    let row = mons.iter().all(|m| coord(m, "y") == coord(&mons[0], "y"));
    let column = mons.iter().all(|m| coord(m, "x") == coord(&mons[0], "x"));
    let line = |first: &str, middle: &str, last: &str| -> Vec<String> {
        (0..n)
            .map(|i| match i {
                0 => first.to_string(),
                i if i == n - 1 => last.to_string(),
                _ if n == 3 => middle.to_string(),
                i => format!("{middle} {i}"),
            })
            .collect()
    };
    if n == 1 {
        vec!["only screen".into()]
    } else if row {
        line("LEFT", "CENTER", "RIGHT")
    } else if column {
        let mut by_y: Vec<usize> = (0..n).collect();
        by_y.sort_by_key(|&i| coord(&mons[i], "y"));
        let words = line("TOP", "MIDDLE", "BOTTOM");
        let mut out = vec![String::new(); n];
        for (rank, &i) in by_y.iter().enumerate() {
            out[i] = words[rank].clone();
        }
        out
    } else {
        mons.iter().map(|m| format!("screen at {},{}", coord(m, "x"), coord(m, "y"))).collect()
    }
}

fn notify(text: &str) {
    println!("{text}");
    let _ = Command::new("notify-send").args(["-t", "2500", "lookfocus record", text]).status();
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let rounds: usize = args.next().map(|a| a.parse()).transpose()?.unwrap_or(3);
    let out_path = args.next().unwrap_or_else(|| {
        format!(
            "prototype/data/rust-session-{}.json",
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()
        )
    });

    let mons = monitors()?;
    let names: Vec<String> = mons.iter().map(|m| m["name"].as_str().unwrap_or("?").to_string()).collect();
    let places = positions(&mons);
    let label = |name: &String| {
        let i = names.iter().position(|n| n == name).unwrap_or(0);
        format!("{} screen ({name})", places[i])
    };
    println!("Monitors:");
    for name in &names {
        println!("  {}", label(name));
    }

    let dir = models::find_model_dir(None)?;
    models::init_onnxruntime()?;
    let tracker = FaceTracker::new(OnnxDetector::load(&dir, 1)?, OnnxFaceMesh::load(&dir, 1)?);
    let camera = V4lCamera::open(std::path::Path::new("/dev/video0"), 640, 480, 15.0)?;
    let mut sampler = Sampler::new(camera, tracker);
    let start = Instant::now();

    // Let auto exposure settle.
    while start.elapsed() < Duration::from_secs(3) {
        sampler.sample()?;
    }

    let mut steps = Vec::new();
    for r in 0..rounds {
        let order: Vec<&String> = if r % 2 == 0 { names.iter().collect() } else { names.iter().rev().collect() };
        for name in order {
            notify(&format!("Round {}: look at the {}", r + 1, label(name)));
            let lead_end = Instant::now() + LEAD;
            while Instant::now() < lead_end {
                sampler.sample()?;
            }
            let mut samples = Vec::new();
            let rec_end = Instant::now() + RECORD;
            while Instant::now() < rec_end {
                let s = sampler.sample()?;
                let t = s.time.duration_since(start).as_secs_f32();
                samples.push(match s.face {
                    Some(f) => json!({"t": t, "face": true, "yaw": f.pose.yaw, "pitch": f.pose.pitch, "roll": f.pose.roll, "luma": f.luma}),
                    None => json!({"t": t, "face": false, "luma": 0.0}),
                });
            }
            let got = samples.iter().filter(|s| s["face"] == true).count();
            println!("  {got}/{} frames had a face", samples.len());
            steps.push(json!({"round": r, "monitor": name, "samples": samples}));
        }
    }
    notify("Done. You can look anywhere now.");

    let session = json!({
        "created": format!("{:?}", std::time::SystemTime::now()),
        "source": "rust",
        "camera_monitor": names.last(),
        "fps": 15,
        "monitors": mons,
        "steps": steps,
    });
    std::fs::write(&out_path, serde_json::to_string_pretty(&session)?)?;
    println!("Saved pose numbers (no images) to {out_path}");
    Ok(())
}
