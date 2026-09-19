#!/usr/bin/env bash
# Build + run on NixOS. winit/wgpu dlopen() these at runtime, and a
# nix-built binary doesn't search any system lib dir, so point it at them.
set -euo pipefail
cd "$(dirname "$0")"

libs=$(nix build --no-link --print-out-paths \
  nixpkgs#vulkan-loader nixpkgs#wayland nixpkgs#libxkbcommon nixpkgs#libGL \
  | sed 's|$|/lib|' | paste -sd:)
export LD_LIBRARY_PATH="$libs:/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cargo build --release
exec ./target/release/fractals "$@"
