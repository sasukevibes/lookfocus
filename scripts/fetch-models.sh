#!/usr/bin/env bash
# Downloads the two ONNX models lookfocus needs and checks their SHA-256.
#
# Both are MediaPipe models (Apache-2.0) from face_landmarker.task, converted
# to ONNX. They are attached to this project's `models-v1` GitHub release, with
# the pinned Hugging Face revisions they came from as a fallback. The
# conversion was checked against Google's original TFLite files: outputs match
# to within 0.0005 (see NOTES.md).
#
# Usage: scripts/fetch-models.sh [target directory]
# Default target: ${XDG_DATA_HOME:-$HOME/.local/share}/lookfocus/models

set -euo pipefail

dest="${1:-${XDG_DATA_HOME:-$HOME/.local/share}/lookfocus/models}"

release="https://github.com/sasukevibes/lookfocus/releases/download/models-v1"
models=(
  "face_detector.onnx|https://huggingface.co/fernandotonon/QtMeshEditor-blazeface-onnx/resolve/50f2c66ffbdf84beae8c267df2b49e5c5a5162e9/face_detector.onnx|02a04d5d37c3558dc4d5274f7f8f0f0f01ac94e46c5ffb2cee82395d47e23181"
  "face_landmarks.onnx|https://huggingface.co/fernandotonon/QtMeshEditor-facemesh-onnx/resolve/32714088ed9830a98df6ee653ad406a3b83ef014/face_landmarks.onnx|d16e5a55e6a480284d468ee32469692464049518c311662ab3956681de31e3e9"
)

mkdir -p "$dest"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

for entry in "${models[@]}"; do
  IFS='|' read -r name fallback sha <<<"$entry"
  if [[ -f "$dest/$name" ]] && echo "$sha  $dest/$name" | sha256sum --check --status; then
    echo "$name is already present"
    continue
  fi
  echo "Downloading $name"
  if ! curl --fail --location --silent --show-error --output "$tmp/$name" "$release/$name"; then
    echo "  release download failed, trying the original source"
    curl --fail --location --silent --show-error --output "$tmp/$name" "$fallback"
  fi
  if ! echo "$sha  $tmp/$name" | sha256sum --check --status; then
    echo "Checksum mismatch for $name. Not installing it." >&2
    exit 1
  fi
  mv "$tmp/$name" "$dest/$name"
done

echo "Models are in $dest"
