#!/usr/bin/env bash
# Removes what scripts/install-local.sh installed. Your calibration and
# settings in ~/.config/lookfocus are kept; delete that folder too if you
# want everything gone.

set -euo pipefail

unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

systemctl --user disable --now lookfocus.service 2>/dev/null || true
rm -f "$unit_dir/lookfocus.service"
systemctl --user daemon-reload
rm -f "$HOME/.local/bin/lookfocus"
rm -rf "${XDG_CONFIG_HOME:-$HOME/.config}/omarchy/plugins/sasukevibes.lookfocus"
rm -rf "${XDG_DATA_HOME:-$HOME/.local/share}/lookfocus"
rm -rf "${XDG_STATE_HOME:-$HOME/.local/state}/lookfocus"
echo "Removed. Settings and calibration remain in ${XDG_CONFIG_HOME:-$HOME/.config}/lookfocus."
