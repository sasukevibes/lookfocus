use std::io::{BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};

use lookfocus::calibrate::{self, Plan, Prompter};
use lookfocus::camera::V4lCamera;
use lookfocus::classify::Classifier;
use lookfocus::config::{CALIBRATION_FILE, Calibration, SETTINGS_FILE, Settings, config_dir};
use lookfocus::control::{self, Status};
use lookfocus::daemon::Daemon;
use lookfocus::events::Event;
use lookfocus::filter::PoseFilter;
use lookfocus::gesture::Gesture;
use lookfocus::hypr::{Compositor, Hyprland};
use lookfocus::models::{self, OnnxDetector, OnnxFaceMesh};
use lookfocus::notify;
use lookfocus::sampler::{HandSample, PoseSource, Sampler};
use lookfocus::state::{State, state_path};
use lookfocus::switcher::{Decision, SwitchParams, Switcher, auto_hysteresis};
use lookfocus::vision::FaceTracker;

/// Exit code for problems a restart cannot fix (no calibration, no models).
/// The systemd unit does not restart on it.
const EXIT_CONFIG: i32 = 78;
const SERVICE: &str = "lookfocus.service";

#[derive(Parser)]
#[command(name = "lookfocus", version, about = "Focus the Hyprland monitor you are facing")]
struct Cli {
    #[command(flatten)]
    input: InputArgs,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Record where you look for each monitor. Run this first.
    Calibrate(CalibrateArgs),
    /// Run calibration again, for example after moving the camera or a monitor.
    Recalibrate(CalibrateArgs),
    /// Track your head and switch monitors (the service runs this).
    Run,
    /// Pause if tracking, resume if paused. Starts the service if it is not running.
    Toggle,
    /// Pause tracking and release the camera.
    Pause,
    /// Resume tracking. Starts the service if it is not running.
    Resume,
    /// Show what lookfocus is doing.
    Status {
        /// Print JSON (for the bar widget and scripts).
        #[arg(long)]
        json: bool,
    },
    /// Switch adaptive centroids, which learn from mouse use.
    Adaptive {
        #[arg(value_enum)]
        action: AdaptiveAction,
    },
    /// Switch hand gestures, which run a command when you hold a hand up.
    Gestures {
        #[arg(value_enum)]
        action: GesturesAction,
    },
    /// Stream events as JSON lines (experimental).
    Watch,
    /// Show live head pose, the monitor it points at, and what would happen.
    Debug {
        /// Stop after this many seconds (default: run until Ctrl+C).
        #[arg(long)]
        seconds: Option<f32>,
        /// Print one JSON object per frame instead of a status line.
        #[arg(long)]
        json: bool,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum AdaptiveAction {
    On,
    Off,
    Toggle,
    /// Forget what has been learned.
    Reset,
}

impl AdaptiveAction {
    fn word(self) -> &'static str {
        match self {
            AdaptiveAction::On => "on",
            AdaptiveAction::Off => "off",
            AdaptiveAction::Toggle => "toggle",
            AdaptiveAction::Reset => "reset",
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum GesturesAction {
    On,
    Off,
    Toggle,
}

impl GesturesAction {
    fn word(self) -> &'static str {
        match self {
            GesturesAction::On => "on",
            GesturesAction::Off => "off",
            GesturesAction::Toggle => "toggle",
        }
    }
}

#[derive(Args)]
struct CalibrateArgs {
    /// How many times to visit each monitor.
    #[arg(long, default_value_t = 2)]
    rounds: usize,
    /// The monitor your camera is mounted on. Asked interactively if omitted.
    #[arg(long)]
    camera_monitor: Option<String>,
}

/// Overrides for values in config.toml.
#[derive(Args)]
struct InputArgs {
    /// V4L2 camera device.
    #[arg(long, global = true)]
    device: Option<PathBuf>,
    /// Directory holding the ONNX model files.
    #[arg(long, global = true)]
    models: Option<PathBuf>,
}

type LiveSampler = Sampler<V4lCamera, OnnxDetector, OnnxFaceMesh>;

fn load_settings(input: &InputArgs) -> Result<Settings> {
    let mut s = Settings::load(&config_dir().join(SETTINGS_FILE))?;
    if let Some(d) = &input.device {
        s.camera.device = d.clone();
    }
    if let Some(m) = &input.models {
        s.models_dir = Some(m.clone());
    }
    Ok(s)
}

/// Finds the models and loads ONNX Runtime. Done once per process.
fn prepare_models(settings: &Settings) -> Result<PathBuf> {
    let dir = models::find_model_dir(settings.models_dir.as_deref())?;
    models::init_onnxruntime()?;
    Ok(dir)
}

fn open_sampler(settings: &Settings, model_dir: &Path) -> Result<LiveSampler> {
    let threads = settings.camera.threads;
    let tracker = FaceTracker::new(OnnxDetector::load(model_dir, threads)?, OnnxFaceMesh::load(model_dir, threads)?);
    let camera = V4lCamera::open(&settings.camera.device, 640, 480, settings.camera.fps)?;
    // The hand models load only when hand tracking is switched on.
    let hand_dir = model_dir.to_path_buf();
    Ok(Sampler::new(camera, tracker).with_hand_loader(move || models::load_hand_tracker(&hand_dir, threads)))
}

fn calibration_path() -> PathBuf {
    config_dir().join(CALIBRATION_FILE)
}

fn load_calibration() -> Result<Calibration> {
    let path = calibration_path();
    Calibration::load(&path)?
        .with_context(|| format!("no calibration at {}. Run `lookfocus calibrate` first.", path.display()))
}

// ---------------------------------------------------------------- control

fn daemon_request(command: &str) -> Result<Status> {
    let reply = control::request(&control::socket_path(), command)?;
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&reply)
        && let Some(e) = v.get("error")
    {
        bail!("{}", e.as_str().unwrap_or("error"));
    }
    serde_json::from_str(&reply).with_context(|| format!("unexpected reply: {reply}"))
}

fn daemon_running() -> bool {
    control::request(&control::socket_path(), "status").is_ok()
}

/// Starts the systemd service and waits for its socket.
fn start_service() -> Result<()> {
    let ok = std::process::Command::new("systemctl").args(["--user", "start", SERVICE]).status()?.success();
    if !ok {
        bail!("lookfocus is not running and `systemctl --user start {SERVICE}` failed. Is the service installed?");
    }
    let until = Instant::now() + Duration::from_secs(10);
    while Instant::now() < until {
        if daemon_running() {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    bail!("the service started but is not answering. See `journalctl --user -u {SERVICE}`.")
}

fn print_status(s: &Status) {
    let mut line = match s.state.as_str() {
        "tracking" => format!("tracking, focused on {}", s.monitor.as_deref().unwrap_or("?")),
        other => other.replace('_', " "),
    };
    if s.running && s.state == "tracking" {
        if let Some(z) = &s.zone {
            line.push_str(&format!(", facing {z}"));
        }
        if s.mouse_hold {
            line.push_str(", held by mouse use");
        }
    }
    println!("{line}");
    if let Some(r) = &s.reason {
        println!("  {r}");
    }
    if s.running {
        println!("  camera {}", if s.camera { "open" } else { "released" });
    }
    let drift: Vec<String> = s.drift.iter().filter(|(_, d)| *d > 0.05).map(|(m, d)| format!("{m} {d:.1}°")).collect();
    println!(
        "  adaptive centroids {}{}",
        if s.adaptive { "on" } else { "off" },
        if drift.is_empty() { String::new() } else { format!(" (learned: {})", drift.join(", ")) }
    );
    let seeing = s.gesture.map(|g| format!(" (seeing {g})")).unwrap_or_default();
    println!("  gestures {}{seeing}", if s.gestures { "on" } else { "off" });
}

/// Status when the daemon is not running, so the bar can still show
/// something useful.
fn stopped_status(settings: &Settings) -> Status {
    let calibrated = calibration_path().exists();
    let state = State::load(&state_path());
    Status {
        running: false,
        state: if calibrated { "stopped" } else { "not_calibrated" }.into(),
        reason: Some(if calibrated {
            format!("Not running. Start it with `systemctl --user start {SERVICE}`.")
        } else {
            "Not calibrated yet. Run `lookfocus calibrate`.".into()
        }),
        monitor: None,
        zone: None,
        face: false,
        camera: false,
        adaptive: state.adaptive.unwrap_or(settings.adaptive.enabled),
        gestures: state.gestures.unwrap_or(settings.gestures.enabled),
        gesture: None,
        mouse_hold: false,
        drift: Vec::new(),
        version: env!("CARGO_PKG_VERSION").into(),
    }
}

fn status(settings: &Settings, json: bool) -> Result<()> {
    let s = daemon_request("status").unwrap_or_else(|_| stopped_status(settings));
    if json {
        println!("{}", serde_json::to_string(&s)?)
    } else {
        print_status(&s)
    }
    Ok(())
}

fn simple_command(command: &str, start_if_stopped: bool) -> Result<()> {
    if !daemon_running() {
        if !start_if_stopped {
            bail!("lookfocus is not running");
        }
        start_service()?;
        if command == "toggle" {
            // Starting it is the toggle: it now tracks.
            print_status(&daemon_request("status")?);
            return Ok(());
        }
    }
    print_status(&daemon_request(command)?);
    Ok(())
}

fn adaptive(settings: &Settings, action: AdaptiveAction) -> Result<()> {
    if daemon_running() {
        print_status(&daemon_request(&format!("adaptive {}", action.word()))?);
        return Ok(());
    }
    // Not running: change the saved switch so the next start uses it.
    let path = state_path();
    let mut state = State::load(&path);
    let current = state.adaptive.unwrap_or(settings.adaptive.enabled);
    match action {
        AdaptiveAction::On => state.adaptive = Some(true),
        AdaptiveAction::Off => state.adaptive = Some(false),
        AdaptiveAction::Toggle => state.adaptive = Some(!current),
        AdaptiveAction::Reset => state.learned = None,
    }
    state.save(&path)?;
    println!(
        "adaptive centroids {} (applies when lookfocus starts)",
        if state.adaptive.unwrap_or(current) { "on" } else { "off" }
    );
    Ok(())
}

fn gestures(settings: &Settings, action: GesturesAction) -> Result<()> {
    if daemon_running() {
        print_status(&daemon_request(&format!("gestures {}", action.word()))?);
        return Ok(());
    }
    // Not running: change the saved switch so the next start uses it.
    let path = state_path();
    let mut state = State::load(&path);
    let current = state.gestures.unwrap_or(settings.gestures.enabled);
    let on = match action {
        GesturesAction::On => true,
        GesturesAction::Off => false,
        GesturesAction::Toggle => !current,
    };
    state.gestures = Some(on);
    state.save(&path)?;
    println!("gestures {} (applies when lookfocus starts)", if on { "on" } else { "off" });
    Ok(())
}

fn watch() -> Result<()> {
    let mut out = std::io::stdout().lock();
    control::watch(&control::socket_path(), |line| {
        writeln!(out, "{line}")?;
        out.flush()?;
        Ok(())
    })
}

// ---------------------------------------------------------------- calibrate

struct CliPrompter;

impl Prompter for CliPrompter {
    fn instruct(&mut self, text: &str) {
        println!("{text}");
        notify::send("lookfocus calibration", text, 2500);
    }
    fn report(&mut self, text: &str) {
        println!("  {text}");
    }
}

fn ask_camera_monitor(plan: &Plan, default: usize) -> Result<usize> {
    let labels = calibrate::position_labels(&plan.monitors);
    println!("Which monitor is your camera on?");
    for (i, m) in plan.monitors.iter().enumerate() {
        let mark = if i == default { "  (default)" } else { "" };
        println!("  {}) {} screen ({}){mark}", i + 1, labels[i], m.name);
    }
    print!("Number [{}]: ", default + 1);
    std::io::stdout().flush()?;
    let mut line = String::new();
    std::io::stdin().lock().read_line(&mut line)?;
    let line = line.trim();
    if line.is_empty() {
        return Ok(default);
    }
    match line.parse::<usize>() {
        Ok(n) if (1..=plan.monitors.len()).contains(&n) => Ok(n - 1),
        _ => bail!("expected a number from 1 to {}", plan.monitors.len()),
    }
}

/// What a running daemon was doing before another command borrowed the camera.
struct Borrowed {
    tracking: bool,
    gestures: bool,
}

/// Pauses a running daemon so calibration can use the camera. Returns what it
/// was doing before, so it can be given back afterwards.
fn borrow_camera_from_daemon() -> Result<Option<Borrowed>> {
    let Ok(before) = daemon_request("status") else { return Ok(None) };
    let borrowed = Borrowed { tracking: before.state != "paused", gestures: before.gestures };
    daemon_request("pause")?;
    if borrowed.gestures {
        // Gestures keep the camera open while paused, so they stop for now too.
        daemon_request("gestures off")?;
    }
    let until = Instant::now() + Duration::from_secs(5);
    while Instant::now() < until {
        if !daemon_request("status")?.camera {
            println!("Paused the running lookfocus to use the camera.");
            return Ok(Some(borrowed));
        }
        std::thread::sleep(Duration::from_millis(150));
    }
    // Do not leave it stopped.
    give_camera_back(&borrowed)?;
    bail!("the running lookfocus did not release the camera")
}

/// Undoes `borrow_camera_from_daemon`.
fn give_camera_back(borrowed: &Borrowed) -> Result<()> {
    if borrowed.tracking {
        daemon_request("resume")?;
    }
    if borrowed.gestures {
        daemon_request("gestures on")?;
    }
    Ok(())
}

fn calibrate(settings: &Settings, args: CalibrateArgs, previous_hint: Option<String>) -> Result<()> {
    let hypr = Hyprland::connect()?;
    let plan = Plan::new(hypr.monitors()?, args.rounds.max(1));
    if plan.monitors.len() < 2 {
        bail!("only one monitor is connected, so there is nothing to switch between");
    }
    let find = |name: &str| plan.monitors.iter().position(|m| m.name == name);
    let hint = match args.camera_monitor.as_deref() {
        Some(name) => Some(find(name).with_context(|| format!("no monitor named {name}"))?),
        None => {
            // A remembered hint becomes the default for the question.
            let default = previous_hint
                .as_deref()
                .and_then(find)
                .unwrap_or_else(|| calibrate::guess_camera_monitor(&plan.monitors));
            if std::io::stdin().is_terminal() { Some(ask_camera_monitor(&plan, default)?) } else { Some(default) }
        }
    };

    let model_dir = prepare_models(settings)?;
    let resume_after = borrow_camera_from_daemon()?;

    println!();
    println!("Calibration takes about {} seconds.", 3 + (plan.monitors.len() * plan.rounds + 1) * 5);
    println!("Sit the way you normally do. A notification names each screen in turn.");
    println!("Look at its center the way you would while working, without exaggerating");
    println!("the head turn, and hold still until the next one.");
    println!();

    let result = (|| {
        let mut sampler = open_sampler(settings, &model_dir)?;
        let visits = calibrate::record(&mut sampler, &plan, &mut CliPrompter)?;
        drop(sampler); // Release the camera before anything else.
        calibrate::summarize(&plan, &visits, hint, now_iso())
    })();

    let saved = result.and_then(|(cal, warnings)| {
        println!();
        println!("  {:8} {:>7} {:>7} {:>8} {:>8} {:>6}", "monitor", "yaw", "pitch", "jitter", "visits", "face");
        for m in &cal.monitors {
            println!(
                "  {:8} {:+7.1} {:+7.1} {:8.1} {:8.1} {:5.0}%",
                m.name,
                m.yaw,
                m.pitch,
                m.hold_spread,
                m.visit_spread,
                m.face_rate * 100.0
            );
        }
        match &cal.look_down {
            Some(ld) => println!("  looking down: pitch {:+.1}, counts below {:+.1}", ld.pitch, ld.threshold),
            None => println!("  looking down: not separable from the screens"),
        }
        if !warnings.is_empty() {
            println!();
            for w in &warnings {
                println!("Note: {w}");
            }
        }
        let path = calibration_path();
        cal.save(&path)?;
        println!();
        println!("Saved to {}.", path.display());
        Ok(())
    });

    // Hand the camera back, whether or not calibration worked.
    if let Some(borrowed) = resume_after {
        let reloaded = if saved.is_ok() {
            daemon_request("reload").map(|_| println!("The running lookfocus now uses the new calibration."))
        } else {
            Ok(())
        };
        // Give the camera back even if the reload failed.
        let returned = give_camera_back(&borrowed);
        reloaded?;
        returned?;
    } else if saved.is_ok() {
        println!("Start tracking with `systemctl --user start {SERVICE}` or `lookfocus run`.");
    }
    saved
}

/// Local time as an ISO 8601 string, without a date crate.
fn now_iso() -> String {
    std::process::Command::new("date")
        .arg("--iso-8601=seconds")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| {
            let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
            format!("unix-{secs}")
        })
}

// ---------------------------------------------------------------- run

fn config_error(e: anyhow::Error) -> ! {
    eprintln!("Error: {e:#}");
    std::process::exit(EXIT_CONFIG);
}

fn run(settings: Settings) -> Result<()> {
    // Problems a restart cannot fix exit with EXIT_CONFIG.
    let calibration = load_calibration().unwrap_or_else(|e| config_error(e));
    let model_dir = prepare_models(&settings).unwrap_or_else(|e| config_error(e));
    let listener = control::bind(&control::socket_path())?;
    let hypr = Hyprland::connect()?;

    let open_settings = settings.clone();
    let opener = Box::new(move || open_sampler(&open_settings, &model_dir));
    let mut daemon = Daemon::new(opener, hypr, calibration, &settings, Some(state_path()), Box::new(Instant::now))?;
    daemon.calibration_path = Some(calibration_path());

    // Desktop notifications for things that need attention.
    let events = daemon.events.subscribe();
    std::thread::spawn(move || {
        for e in events {
            match e {
                Event::Paused { reason } if reason != "paused" => notify::send("lookfocus paused", &reason, 8000),
                Event::CameraUnavailable { reason } => notify::send("lookfocus cannot use the camera", &reason, 8000),
                _ => {}
            }
        }
    });
    daemon.start();

    // Hyprland events arrive on their own thread. If Hyprland restarts, the
    // stream ends and we reconnect.
    let (tx, hypr_events) = mpsc::channel();
    std::thread::spawn(move || {
        loop {
            match Hyprland::connect().and_then(|h| h.events()) {
                Ok(stream) => {
                    for e in stream {
                        if tx.send(e).is_err() {
                            return;
                        }
                    }
                    log::warn!("Hyprland event stream closed, reconnecting");
                }
                Err(e) => log::warn!("cannot read Hyprland events: {e}"),
            }
            std::thread::sleep(Duration::from_secs(2));
        }
    });

    log::info!("lookfocus {} running", env!("CARGO_PKG_VERSION"));
    daemon.run(hypr_events, control::serve(listener))
}

// ---------------------------------------------------------------- debug

fn debug(settings: &Settings, seconds: Option<f32>, json: bool) -> Result<()> {
    let calibration = Calibration::load(&calibration_path())?;
    let mut switcher = calibration.as_ref().map(|c| {
        let classifier = Classifier::new(c.centroids());
        let hysteresis = settings.switching.hysteresis_deg.unwrap_or_else(|| auto_hysteresis(&classifier));
        Switcher::new(
            classifier,
            SwitchParams {
                dwell: Duration::from_millis(settings.switching.dwell_ms),
                hysteresis,
                settle_speed: settings.switching.settle_speed,
            },
        )
    });
    if switcher.is_none() && !json {
        eprintln!("No calibration yet, so only the pose is shown. Run `lookfocus calibrate` to see zones.");
    }

    let model_dir = prepare_models(settings)?;
    let resume_after = borrow_camera_from_daemon()?;
    let result = debug_loop(settings, &model_dir, switcher.as_mut(), seconds, json);
    if let Some(borrowed) = resume_after {
        give_camera_back(&borrowed)?;
    }
    result
}

/// The hand in a sample as a few words: none, a gesture with the model's
/// confidence, or a hand showing no known gesture.
fn hand_text(hand: Option<&HandSample>) -> String {
    match hand {
        None => "none".into(),
        Some(h) => format!("{} {:.2}", h.gesture.map_or("unknown", Gesture::name), h.confidence),
    }
}

fn debug_loop(
    settings: &Settings,
    model_dir: &Path,
    mut switcher: Option<&mut Switcher>,
    seconds: Option<f32>,
    json: bool,
) -> Result<()> {
    let mut sampler = open_sampler(settings, model_dir)?;
    // Show hands whatever the gestures setting is, to check they work.
    sampler.set_hands(true);
    let mut filter = PoseFilter::new(settings.filter);
    let start = Instant::now();
    let end = seconds.map(|s| start + Duration::from_secs_f32(s));
    let tty = std::io::stdout().is_terminal() && !json;
    let mut out = std::io::stdout().lock();
    let mut last = start;
    let mut fps = 0.0f32;
    let ms = |d: Duration| d.as_secs_f32() * 1000.0;
    while end.is_none_or(|e| Instant::now() < e) {
        let s = sampler.sample()?;
        let t = s.time.duration_since(start).as_secs_f32();
        let dt = s.time.duration_since(last).as_secs_f32();
        last = s.time;
        if dt > 0.0 {
            fps = if fps == 0.0 { 1.0 / dt } else { 0.9 * fps + 0.1 / dt };
        }
        let Some(f) = &s.face else {
            filter.reset();
            if let Some(sw) = switcher.as_deref_mut() {
                sw.hold();
            }
            if json {
                let v = serde_json::json!({
                    "t": t, "face": false, "hand": s.hand.is_some(), "gesture": s.hand.as_ref().and_then(|h| h.gesture),
                    "inference_ms": ms(s.inference),
                });
                writeln!(out, "{v}")?;
            } else {
                let line = format!("no face  hand {}  {fps:4.1} fps", hand_text(s.hand.as_ref()));
                if tty { write!(out, "\r{line:<120}")? } else { writeln!(out, "{line}")? }
            }
            out.flush()?;
            continue;
        };
        let (yaw, pitch) = filter.filter(t, f.pose.yaw, f.pose.pitch);
        let speed = filter.speed();
        // What the daemon would do, without doing it.
        let (zone, margin, decision) = match switcher.as_deref_mut() {
            Some(sw) => {
                let c = sw.classifier().classify(yaw, pitch);
                let zone = c.as_ref().map(|c| sw.classifier().centroids()[c.best].monitor.clone());
                let margin = c.map(|c| c.margin());
                let d = sw.update(s.time, yaw, pitch, speed);
                let name = |i: usize| sw.classifier().centroids()[i].monitor.clone();
                let text = match d {
                    Decision::Stay => "stay".to_string(),
                    Decision::Pending { target, held } => format!("pending {} {:.0} ms", name(target), ms(held)),
                    Decision::Switch { to, .. } => format!("SWITCH -> {}", name(to)),
                };
                (zone, margin, Some(text))
            }
            None => (None, None, None),
        };
        if json {
            let v = serde_json::json!({
                "t": t, "face": true, "yaw": f.pose.yaw, "pitch": f.pose.pitch, "yaw_smooth": yaw,
                "pitch_smooth": pitch, "speed": speed, "zone": zone, "margin": margin, "decision": decision,
                "luma": f.luma, "hand": s.hand.is_some(), "gesture": s.hand.as_ref().and_then(|h| h.gesture),
                "inference_ms": ms(s.inference),
            });
            writeln!(out, "{v}")?;
        } else {
            let mut line = format!("yaw {yaw:+6.1}  pitch {pitch:+6.1}  speed {speed:4.0}°/s");
            if let (Some(z), Some(m), Some(d)) = (&zone, margin, &decision) {
                line.push_str(&format!("  zone {z:<6} margin {m:4.1}  {d}"));
            }
            line.push_str(&format!("  hand {}", hand_text(s.hand.as_ref())));
            line.push_str(&format!("  {fps:4.1} fps"));
            if tty { write!(out, "\r{line:<120}")? } else { writeln!(out, "{line}")? }
        }
        out.flush()?;
    }
    if tty {
        writeln!(out)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).format_timestamp_secs().init();
    let cli = Cli::parse();
    let settings = load_settings(&cli.input)?;
    match cli.command {
        Command::Calibrate(args) => calibrate(&settings, args, None),
        Command::Recalibrate(args) => {
            let previous = Calibration::load(&calibration_path())?.and_then(|c| c.camera_monitor);
            calibrate(&settings, args, previous)
        }
        Command::Run => run(settings),
        Command::Toggle => simple_command("toggle", true),
        Command::Pause => simple_command("pause", false),
        Command::Resume => simple_command("resume", true),
        Command::Status { json } => status(&settings, json),
        Command::Adaptive { action } => adaptive(&settings, action),
        Command::Gestures { action } => gestures(&settings, action),
        Command::Watch => watch(),
        Command::Debug { seconds, json } => debug(&settings, seconds, json),
    }
}
