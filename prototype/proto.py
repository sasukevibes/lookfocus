"""Throwaway Phase 1 prototype for lookfocus.

It answers one question before the Rust build starts: from where the camera
sits, are the head poses for each monitor far enough apart to tell them apart?

Frames are processed in memory and never written to disk. Only pose numbers
(yaw, pitch, roll, face brightness) are saved.

Commands:
  live      print live yaw and pitch
  record    guided per-monitor recording, then analysis
  analyze   re-run the analysis on a saved recording
  classify  live nearest-centroid prediction using a saved recording
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# MediaPipe 1.1.0 imports sounddevice while building a vision task. On this
# machine that opens PortAudio, then ALSA's JACK plugin, then PipeWire, and the
# process gets killed. We never need audio, so make that import fail cleanly.
# MediaPipe already catches the ImportError.
sys.modules["sounddevice"] = None
# Hide MediaPipe's startup log lines so the output stays readable.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

HERE = Path(__file__).resolve().parent
MODEL = HERE / "models" / "face_landmarker.task"
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
DATA = HERE / "data"

# Yaw beyond this, measured from looking straight at the camera, is where
# landmark quality is expected to drop off.
EXTREME_YAW_DEG = 45.0


# ---------------------------------------------------------------- capture


class Tracker:
    """Opens the camera and turns frames into (yaw, pitch, roll) samples."""

    def __init__(self, device: int, fps: float):
        if not MODEL.exists():
            sys.exit(f"Model missing. Download it with:\n  curl -L -o {MODEL} {MODEL_URL}")
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            sys.exit(f"Could not open /dev/video{device}. Is another app using the camera?")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.period = 1.0 / fps
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.t0 = time.monotonic()
        self.last = 0.0

    def close(self):
        self.landmarker.close()
        self.cap.release()

    def next(self) -> dict:
        """Returns one sample at roughly the target rate.

        The camera only offers 30 fps modes, so frames that arrive before the
        next slot are read and dropped.
        """
        while True:
            ok, frame = self.cap.read()
            if not ok:
                sys.exit("Camera stopped returning frames.")
            now = time.monotonic()
            if now - self.last >= self.period * 0.9:
                self.last = now
                break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(image, int((now - self.t0) * 1000))
        sample = {"t": round(now - self.t0, 3), "face": False}
        if not result.facial_transformation_matrixes:
            sample["luma"] = round(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()), 1)
            return sample
        yaw, pitch, roll = matrix_to_angles(np.asarray(result.facial_transformation_matrixes[0]))
        h, w = frame.shape[:2]
        lm = np.array([(p.x * w, p.y * h, p.z * w) for p in result.face_landmarks[0]])
        o_yaw, o_pitch = ortho_angles(lm)
        sample.update(
            face=True,
            yaw=round(yaw, 2),
            pitch=round(pitch, 2),
            roll=round(roll, 2),
            # The Rust pipeline's orthographic fit, run on MediaPipe's own
            # landmarks, so the two fits can be compared on the same frames.
            yaw_ortho=round(o_yaw, 2),
            pitch_ortho=round(o_pitch, 2),
            # Where the face sits in the frame, in pixels from the center.
            face_dx=round(float(lm[:, 0].mean() - w / 2), 1),
            face_dy=round(float(lm[:, 1].mean() - h / 2), 1),
            luma=round(face_luma(frame, result.face_landmarks[0]), 1),
        )
        return sample


def load_canonical():
    """Reads the 33 weighted canonical points from the Rust source."""
    import re
    text = (HERE.parent / "src" / "pose" / "canonical.rs").read_text()
    rows = re.findall(r"\((\d+), ([-\d.e]+), \[([-\d.e]+), ([-\d.e]+), ([-\d.e]+)\]\)", text)
    idx = np.array([int(r[0]) for r in rows])
    weights = np.array([float(r[1]) for r in rows])
    points = np.array([[float(r[2]), float(r[3]), float(r[4])] for r in rows])
    return idx, weights, points


CANON_IDX, CANON_W, CANON_PTS = load_canonical()


def ortho_angles(lm: np.ndarray) -> tuple[float, float]:
    """The same weighted Kabsch fit as src/pose/mod.rs, on image-space landmarks."""
    dst = lm[CANON_IDX] * np.array([1.0, -1.0, -1.0])
    src = CANON_PTS
    w = CANON_W[:, None]
    s0 = (src * w).sum(0) / w.sum()
    d0 = (dst * w).sum(0) / w.sum()
    hm = ((src - s0) * w).T @ (dst - d0)
    u, _, vt = np.linalg.svd(hm)
    v = vt.T
    d = np.eye(3)
    if np.linalg.det(v @ u.T) < 0:
        d[2, 2] = -1
    r = v @ d @ u.T
    fwd = r[:, 2]
    return math.degrees(math.atan2(fwd[0], fwd[2])), math.degrees(math.atan2(fwd[1], math.hypot(fwd[0], fwd[2])))


def matrix_to_angles(m: np.ndarray) -> tuple[float, float, float]:
    """Turns MediaPipe's facial transformation matrix into degrees.

    The matrix maps the canonical face model into camera space. Its third
    column is the direction the face points. Yaw is the left/right angle of
    that direction and pitch is the up/down angle. Zero means facing the
    camera.
    """
    r = m[:3, :3]
    fwd = r[:, 2]
    yaw = math.degrees(math.atan2(fwd[0], fwd[2]))
    pitch = math.degrees(math.atan2(fwd[1], math.hypot(fwd[0], fwd[2])))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return yaw, pitch, roll


def face_luma(frame: np.ndarray, landmarks) -> float:
    """Mean brightness inside the face's bounding box, 0 to 255."""
    h, w = frame.shape[:2]
    xs = [p.x for p in landmarks]
    ys = [p.y for p in landmarks]
    x0, x1 = max(0, int(min(xs) * w)), min(w, int(max(xs) * w))
    y0, y1 = max(0, int(min(ys) * h)), min(h, int(max(ys) * h))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).mean())


# ---------------------------------------------------------------- hyprland


def hypr_monitors() -> list[dict]:
    """Reads monitors from Hyprland, sorted left to right then top to bottom."""
    try:
        out = subprocess.run(["hyprctl", "monitors", "-j"], capture_output=True, text=True, check=True).stdout
        mons = json.loads(out)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as e:
        sys.exit(f"Could not read monitors from Hyprland: {e}")
    result = []
    for m in mons:
        w, h = m["width"] / m["scale"], m["height"] / m["scale"]
        if m["transform"] % 2 == 1:
            w, h = h, w
        result.append({"name": m["name"], "x": m["x"], "y": m["y"], "w": round(w), "h": round(h),
                       "description": m["description"]})
    return sorted(result, key=lambda m: (m["x"], m["y"]))


def describe(mon: dict, mons: list[dict]) -> str:
    """A plain position word so the prompt says where to look."""
    i = mons.index(mon)
    if len(mons) == 1:
        return "only screen"
    if all(m["y"] == mons[0]["y"] for m in mons):
        if i == 0:
            return "LEFT screen"
        if i == len(mons) - 1:
            return "RIGHT screen"
        return "CENTER screen" if len(mons) == 3 else f"screen {i + 1} from the left"
    return f"screen at {mon['x']},{mon['y']}"


def notify(text: str):
    print(text, flush=True)
    if shutil.which("notify-send"):
        subprocess.run(["notify-send", "-t", "2500", "lookfocus prototype", text], check=False)


# ---------------------------------------------------------------- commands


def cmd_live(args):
    tr = Tracker(args.device, args.fps)
    end = time.monotonic() + args.seconds if args.seconds else math.inf
    tty = sys.stdout.isatty()
    try:
        while time.monotonic() < end:
            s = tr.next()
            if s["face"]:
                line = f"yaw {s['yaw']:+7.1f}  pitch {s['pitch']:+7.1f}  roll {s['roll']:+6.1f}  luma {s['luma']:5.1f}"
            else:
                line = f"no face                                          luma {s['luma']:5.1f}"
            print(("\r" + line) if tty else line, end="" if tty else "\n", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        tr.close()
        print()


def cmd_record(args):
    mons = hypr_monitors()
    names = [m["name"] for m in mons]
    camera = args.camera_monitor or mons[-1]["name"]
    if camera not in names:
        sys.exit(f"--camera-monitor {camera} is not one of {names}")
    print("Monitors, left to right:")
    for m in mons:
        tag = "  <- camera" if m["name"] == camera else ""
        print(f"  {m['name']:8} {describe(m, mons):10} {m['w']}x{m['h']} at {m['x']},{m['y']}{tag}")
    print(f"\n{args.rounds} rounds. Each step: {args.lead}s to turn your head, then {args.seconds}s of recording.")
    print("Sit how you normally sit, turn your head naturally, and look at the center of the named monitor.\n")

    tr = Tracker(args.device, args.fps)
    session = {"created": datetime.now().isoformat(timespec="seconds"), "camera_monitor": camera,
               "fps": args.fps, "monitors": mons, "steps": []}
    try:
        # Warm up so auto exposure settles before the first step. On the
        # development laptop's camera this takes about two seconds.
        for _ in range(int(args.fps * 3)):
            tr.next()
        for r in range(args.rounds):
            order = mons if r % 2 == 0 else list(reversed(mons))
            for m in order:
                notify(f"Round {r + 1}: look at the {describe(m, mons)} ({m['name']})")
                lead_end = time.monotonic() + args.lead
                while time.monotonic() < lead_end:
                    tr.next()
                print("  recording...", end="", flush=True)
                samples = []
                rec_end = time.monotonic() + args.seconds
                while time.monotonic() < rec_end:
                    samples.append(tr.next())
                got = sum(s["face"] for s in samples)
                print(f" {got}/{len(samples)} frames had a face")
                session["steps"].append({"round": r, "monitor": m["name"], "samples": samples})
        notify("Done. You can look anywhere now.")
    except KeyboardInterrupt:
        print("\nStopped early. Saving what was recorded.")
    finally:
        tr.close()

    DATA.mkdir(exist_ok=True)
    path = DATA / f"session-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(session, indent=1))
    print(f"\nSaved pose numbers (no images) to {path.relative_to(HERE)}\n")
    analyze(session)


def cmd_analyze(args):
    analyze(json.loads(Path(args.file).read_text()))


def cmd_classify(args):
    session = json.loads(Path(args.file).read_text())
    cents = centroids(session)
    tr = Tracker(args.device, args.fps)
    tty = sys.stdout.isatty()
    end = time.monotonic() + args.seconds if args.seconds else math.inf
    log = []
    try:
        while time.monotonic() < end:
            s = tr.next()
            if not s["face"]:
                line = "no face"
            else:
                p = np.array([s["yaw"], s["pitch"]])
                d = {n: float(np.linalg.norm(p - c)) for n, c in cents.items()}
                best = min(d, key=d.get)
                s["pred"] = best
                s["dist"] = {n: round(v, 2) for n, v in d.items()}
                dists = "  ".join(f"{n} {v:5.1f}" for n, v in d.items())
                line = f"yaw {s['yaw']:+6.1f} pitch {s['pitch']:+6.1f}  ->  {best:8}  [{dists}]"
            log.append(s)
            print(("\r" + line.ljust(90)) if tty else line, end="" if tty else "\n", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        tr.close()
        print()

    DATA.mkdir(exist_ok=True)
    path = DATA / f"classify-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps({"calibration": str(args.file), "samples": log}, indent=1))
    print(f"\nSaved pose numbers (no images) to {path.relative_to(HERE)}\n")
    summarize_classify(log, cents)


def summarize_classify(log: list[dict], cents: dict, dwell_s: float = 0.3):
    """Summarizes a free-use classify run: where time went and how often the
    prediction would have switched, with and without a dwell."""
    faces = [s for s in log if s["face"]]
    if not faces:
        print("No face found during the run.")
        return
    dur = log[-1]["t"] - log[0]["t"] if len(log) > 1 else 0
    print(f"{len(log)} samples over {dur:.0f}s, face found in {100 * len(faces) / len(log):.0f}%")

    print("\nTime per predicted monitor:")
    for n in cents:
        k = sum(s["pred"] == n for s in faces)
        print(f"  {n:8} {100 * k / len(faces):5.1f}%")

    # Margin: how much closer the winner is than the runner-up. Small margins
    # mean the pose sat between two monitors.
    margins = []
    for s in faces:
        d = sorted(s["dist"].values())
        margins.append(d[1] - d[0] if len(d) > 1 else math.inf)
    margins = np.array(margins)
    print("\nMargin between best and second-best monitor (degrees):")
    print(f"  median {np.median(margins):.1f}, 10th percentile {np.percentile(margins, 10):.1f}")
    print(f"  {100 * np.mean(margins < 3):.0f}% of frames under 3 degrees (ambiguous)")

    raw = sum(1 for a, b in zip(faces, faces[1:]) if a["pred"] != b["pred"])
    # Replay with a dwell: a new monitor only wins after it has held for dwell_s.
    current, cand, since, dwell_switches, flickers = faces[0]["pred"], None, None, 0, 0
    for s in faces[1:]:
        if s["pred"] == current:
            if cand is not None:
                flickers += 1
            cand = None
        elif s["pred"] != cand:
            cand, since = s["pred"], s["t"]
        elif s["t"] - since >= dwell_s:
            current, cand = cand, None
            dwell_switches += 1
    print(f"\nPrediction changes: {raw} raw, {dwell_switches} after a {dwell_s * 1000:.0f}ms dwell "
          f"({flickers} brief excursions filtered out)")

    yaws = np.array([s["yaw"] for s in faces])
    lo, hi = min(c[0] for c in cents.values()), max(c[0] for c in cents.values())
    print(f"\nYaw range in use: {np.percentile(yaws, 5):+.1f} to {np.percentile(yaws, 95):+.1f} "
          f"(5th to 95th percentile). Calibrated centroids span {lo:+.1f} to {hi:+.1f}.")


# ---------------------------------------------------------------- analysis


def face_points(steps, monitor=None, round_=None) -> np.ndarray:
    pts = [
        (s["yaw"], s["pitch"])
        for st in steps
        if (monitor is None or st["monitor"] == monitor) and (round_ is None or st["round"] == round_)
        for s in st["samples"]
        if s["face"]
    ]
    return np.array(pts).reshape(-1, 2)


def centroids(session, round_=None) -> dict:
    out = {}
    for m in session["monitors"]:
        pts = face_points(session["steps"], m["name"], round_)
        if len(pts):
            out[m["name"]] = np.median(pts, axis=0)
    return out


def robust_spread(pts: np.ndarray) -> np.ndarray:
    """Per-axis spread as a standard deviation estimate that ignores outliers."""
    med = np.median(pts, axis=0)
    return 1.4826 * np.median(np.abs(pts - med), axis=0)


def smooth(pts: np.ndarray, n: int) -> np.ndarray:
    """Simple moving average, a stand-in for the One Euro filter."""
    if len(pts) < n:
        return pts
    k = np.ones(n) / n
    return np.column_stack([np.convolve(pts[:, i], k, mode="valid") for i in range(2)])


def analyze(session):
    mons = [m["name"] for m in session["monitors"]]
    steps = session["steps"]
    rounds = sorted({st["round"] for st in steps})
    camera = session["camera_monitor"]

    print("Per monitor (degrees). Spread is a robust standard deviation.")
    print(f"  {'monitor':8} {'face%':>6} {'yaw':>7} {'pitch':>7} {'yaw sd':>7} {'pit sd':>7}  per-round yaw/pitch")
    stats = {}
    for name in mons:
        all_s = [s for st in steps if st["monitor"] == name for s in st["samples"]]
        pts = face_points(steps, name)
        rate = 100 * len(pts) / max(1, len(all_s))
        if not len(pts):
            print(f"  {name:8} {rate:5.0f}%   no face detected")
            continue
        med, sd = np.median(pts, axis=0), robust_spread(pts)
        per_round = []
        for r in rounds:
            rp = face_points(steps, name, r)
            per_round.append(f"{np.median(rp[:, 0]):+.1f}/{np.median(rp[:, 1]):+.1f}" if len(rp) else "none")
        stats[name] = {"med": med, "sd": sd, "rate": rate, "n": len(pts)}
        print(f"  {name:8} {rate:5.0f}% {med[0]:+7.1f} {med[1]:+7.1f} {sd[0]:7.2f} {sd[1]:7.2f}  {'  '.join(per_round)}")

    with_ortho = [s for st in steps for s in st["samples"] if s.get("face") and "yaw_ortho" in s]
    if with_ortho:
        print("\nMediaPipe perspective fit vs the Rust orthographic fit, same frames (degrees):")
        print(f"  {'monitor':8} {'yaw':>7} {'yaw_o':>7} {'pitch':>7} {'pitch_o':>8} {'face_dx':>8} {'face_dy':>8}")
        for name in mons:
            rows = [s for st in steps if st["monitor"] == name for s in st["samples"] if s.get("face") and "yaw_ortho" in s]
            if rows:
                med = lambda k: float(np.median([r[k] for r in rows]))
                print(f"  {name:8} {med('yaw'):+7.1f} {med('yaw_ortho'):+7.1f} {med('pitch'):+7.1f} "
                      f"{med('pitch_ortho'):+8.1f} {med('face_dx'):+8.0f} {med('face_dy'):+8.0f}")

    if camera in stats:
        ref = stats[camera]["med"]
        print(f"\nYaw relative to the camera monitor ({camera}):")
        for name, s in stats.items():
            rel = s["med"][0] - ref[0]
            flag = "  <- beyond {:.0f}°, landmarks may be unreliable".format(EXTREME_YAW_DEG) if abs(rel) > EXTREME_YAW_DEG else ""
            print(f"  {name:8} {rel:+6.1f}°{flag}")

    lumas = [s["luma"] for st in steps for s in st["samples"] if s["face"]]
    if lumas:
        print(f"\nFace brightness: median {np.median(lumas):.0f}/255 (below about 60 is dim)")

    names = list(stats)
    if len(names) >= 2:
        print("\nSeparation between monitor pairs.")
        print("  ratio = centroid distance / (spread A + spread B) along the line between them.")
        print("  Above 3 is comfortable, 1.5 to 3 is workable with smoothing, below 1.5 is a problem.")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = stats[names[i]], stats[names[j]]
                diff = b["med"] - a["med"]
                dist = float(np.linalg.norm(diff))
                u = diff / dist if dist else np.array([1.0, 0.0])
                sa = float(np.linalg.norm(a["sd"] * u))
                sb = float(np.linalg.norm(b["sd"] * u))
                ratio = dist / (sa + sb) if sa + sb else math.inf
                print(f"  {names[i]:6} vs {names[j]:6} distance {dist:5.1f}°  ratio {ratio:5.1f}")

    # Nearest centroid, trained on one round and tested on the others, so the
    # number reflects coming back to a monitor rather than staring at it.
    if len(rounds) >= 2 and len(names) >= 2:
        print("\nNearest-centroid accuracy, trained on one round and tested on the rest:")
        for smooth_n, label in [(1, "raw frames"), (5, "5-frame average")]:
            correct = total = 0
            for train in rounds:
                cents = centroids(session, train)
                for st in steps:
                    if st["round"] == train or st["monitor"] not in cents:
                        continue
                    pts = smooth(face_points([st]), smooth_n)
                    for p in pts:
                        best = min(cents, key=lambda n: np.linalg.norm(p - cents[n]))
                        correct += best == st["monitor"]
                        total += 1
            if total:
                print(f"  {label:16} {100 * correct / total:5.1f}%  ({correct}/{total})")


# ---------------------------------------------------------------- main


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", type=int, default=0, help="V4L2 device number (default 0)")
    p.add_argument("--fps", type=float, default=15)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("live")
    s.add_argument("--seconds", type=float, default=0, help="stop after this long (default: until Ctrl+C)")
    s.set_defaults(func=cmd_live)

    s = sub.add_parser("record")
    s.add_argument("--rounds", type=int, default=3)
    s.add_argument("--seconds", type=float, default=2.0, help="recording time per monitor")
    s.add_argument("--lead", type=float, default=3.0, help="time to turn your head before recording")
    s.add_argument("--camera-monitor", help="monitor the camera is on (default: rightmost)")
    s.set_defaults(func=cmd_record)

    s = sub.add_parser("analyze")
    s.add_argument("file")
    s.set_defaults(func=cmd_analyze)

    s = sub.add_parser("classify")
    s.add_argument("file")
    s.add_argument("--seconds", type=float, default=0)
    s.set_defaults(func=cmd_classify)

    args = p.parse_args()
    if "HYPRLAND_INSTANCE_SIGNATURE" not in os.environ and args.cmd in ("record",):
        runtime = Path(os.environ.get("XDG_RUNTIME_DIR", "/run/user/1000")) / "hypr"
        sigs = sorted(runtime.glob("*"), key=lambda q: q.stat().st_mtime) if runtime.exists() else []
        if sigs:
            os.environ["HYPRLAND_INSTANCE_SIGNATURE"] = sigs[-1].name
    args.func(args)


if __name__ == "__main__":
    main()
