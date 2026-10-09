#!/usr/bin/env bash
# Installs lookfocus for the current user, without a package:
#
#   - the binary to ~/.local/bin/lookfocus
#   - the ONNX models (face and hand) to ~/.local/share/lookfocus/models
#   - a systemd user service, enabled for the graphical session
#   - the Omarchy bar widget to ~/.config/omarchy/plugins/lookfocus (if
#     Omarchy is installed)
#
# It does not calibrate, add keybinds or change the bar layout. The README
# covers those. Run scripts/uninstall-local.sh to undo.

set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
bin_dir="$HOME/.local/bin"
unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
plugin_dir="${XDG_CONFIG_HOME:-$HOME/.config}/omarchy/plugins/sasukevibes.lookfocus"

if [[ ! -e /usr/lib/libonnxruntime.so ]] && ! ldconfig -p 2>/dev/null | grep libonnxruntime.so >/dev/null; then
  echo "ONNX Runtime is missing. On Arch: sudo pacman -S onnxruntime-cpu" >&2
  exit 1
fi

echo "Building"
cargo build --release --manifest-path "$repo/Cargo.toml"

echo "Installing the binary to $bin_dir"
install -Dm755 "$repo/target/release/lookfocus" "$bin_dir/lookfocus"

"$repo/scripts/fetch-models.sh"

echo "Installing the systemd user service"
mkdir -p "$unit_dir"
sed "s|^ExecStart=/usr/bin/lookfocus|ExecStart=$bin_dir/lookfocus|" \
  "$repo/packaging/systemd/lookfocus.service" > "$unit_dir/lookfocus.service"
systemctl --user daemon-reload
systemctl --user enable lookfocus.service

if [[ -d "${XDG_CONFIG_HOME:-$HOME/.config}/omarchy" ]]; then
  echo "Installing the Omarchy bar widget to $plugin_dir"
  mkdir -p "$plugin_dir"
  install -m644 "$repo/omarchy/sasukevibes.lookfocus/manifest.json" "$repo/omarchy/sasukevibes.lookfocus/BarWidget.qml" "$plugin_dir/"
fi

cat <<MSG

Installed. Next:
  1. lookfocus calibrate
  2. systemctl --user start lookfocus
MSG
