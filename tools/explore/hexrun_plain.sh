#!/usr/bin/env bash
# Hexagon hunt on plain-set spiral regions (no inherited rings).
out=${1:-/home/diatom/Pictures/fractals/explore/hex3}
mkdir -p "$out"
run() {
  uv run -q --with numpy --with pillow --with scipy --with gmpy2 --with scikit-image \
    python hexmorph.py "$1" "$2" 0 "$out" --width-k "$3" --n 5 --seed "$4" > "$out/log_$4.txt" 2>&1
}
run -0.1560 1.0320 4e-3 1 &
run -0.5600 0.6400 4e-3 2 &
run -0.0883 0.6545 4e-3 3 &
wait
cat "$out"/log_*.txt
