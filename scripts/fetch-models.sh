#!/usr/bin/env bash
# Downloads the ONNX models lookfocus needs and checks their SHA-256.
#
# All are MediaPipe models (Apache-2.0) converted to ONNX.
#
# - The face models come from face_landmarker.task. They are attached to this
#   project's `models-v1` GitHub release, with the pinned Hugging Face
#   revisions they came from as a fallback. The conversion was checked against
#   Google's original TFLite files: outputs match to within 0.0005 (see
#   NOTES.md).
# - The hand models are OpenCV Zoo's conversions of MediaPipe's palm detector
#   and hand landmark model, from pinned Hugging Face revisions. They are not
#   on the release yet, so they come from Hugging Face directly.
#
# Usage: scripts/fetch-models.sh [target directory]
# Default target: ${XDG_DATA_HOME:-$HOME/.local/share}/lookfocus/models

set -euo pipefail

dest="${1:-${XDG_DATA_HOME:-$HOME/.local/share}/lookfocus/models}"

release="https://github.com/sasukevibes/lookfocus/releases/download/models-v1"
# Each entry: file name, original URL, SHA-256, and whether the release
# carries a copy (yes or no).
models=(
  "face_detector.onnx|https://huggingface.co/fernandotonon/QtMeshEditor-blazeface-onnx/resolve/50f2c66ffbdf84beae8c267df2b49e5c5a5162e9/face_detector.onnx|02a04d5d37c3558dc4d5274f7f8f0f0f01ac94e46c5ffb2cee82395d47e23181|yes"
  "face_landmarks.onnx|https://huggingface.co/fernandotonon/QtMeshEditor-facemesh-onnx/resolve/32714088ed9830a98df6ee653ad406a3b83ef014/face_landmarks.onnx|d16e5a55e6a480284d468ee32469692464049518c311662ab3956681de31e3e9|yes"
  "hand_detector.onnx|https://huggingface.co/opencv/palm_detection_mediapipe/resolve/233e619dcea1759bf6de707b9b904fe30881ea55/palm_detection_mediapipe_2023feb.onnx|78ff51c38496b7fc8b8ebdb6cc8c1abb02fa6c38427c6848254cdaba57fcce7c|no"
  "hand_landmarks.onnx|https://huggingface.co/opencv/handpose_estimation_mediapipe/resolve/4b2a0b446e5cf2f11fb6b2c7251091c035d2c1f7/handpose_estimation_mediapipe_2023feb.onnx|db0898ae717b76b075d9bf563af315b29562e11f8df5027a1ef07b02bef6d81c|no"
)

mkdir -p "$dest"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

for entry in "${models[@]}"; do
  IFS='|' read -r name source sha mirrored <<<"$entry"
  if [[ -f "$dest/$name" ]] && echo "$sha  $dest/$name" | sha256sum --check --status; then
    echo "$name is already present"
    continue
  fi
  echo "Downloading $name"
  if [[ "$mirrored" != yes ]]; then
    curl --fail --location --silent --show-error --output "$tmp/$name" "$source"
  elif ! curl --fail --location --silent --show-error --output "$tmp/$name" "$release/$name"; then
    echo "  release download failed, trying the original source"
    curl --fail --location --silent --show-error --output "$tmp/$name" "$source"
  fi
  if ! echo "$sha  $tmp/$name" | sha256sum --check --status; then
    echo "Checksum mismatch for $name. Not installing it." >&2
    exit 1
  fi
  mv "$tmp/$name" "$dest/$name"
done

echo "Models are in $dest"
