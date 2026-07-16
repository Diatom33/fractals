# Fractal Explorer

GPU-accelerated fractal renderer with an interactive GUI and deep-zoom support. Built with Rust, wgpu compute shaders (WGSL), and egui.

## Fractal Types

| Type | Formula | Description |
|------|---------|-------------|
| **Mandelbrot** | z = z² + c | Classic escape-time fractal. c = pixel, z0 = 0 |
| **Julia** | z = z² + c | Julia set. z0 = pixel, c = adjustable constant |
| **Burning Ship** | z = (\|Re(z)\| + i\|Im(z)\|)² + c | Absolute value before squaring |
| **Multibrot** | z = z^d + c | Generalized Mandelbrot with configurable power d |
| **Newton** | z = z - a·f(z)/f'(z) | Newton's method for z^n - 1, colored by basin |
| **Nova Julia** | z = z - a·f(z)/f'(z) + c | Nova fractal with fixed c, z0 = pixel |
| **Nova Mandelbrot** | z = z - a·f(z)/f'(z) + c | Nova fractal with c = pixel, z0 = critical point |
| **Tricorn** | z = conj(z)² + c | Anti-holomorphic Mandelbrot |
| **Celtic** | \|Re(z²)\| + i·Im(z²) + c | Abs on the real part after squaring |
| **Perpendicular** | Re(z²) - 2i·\|Re(z)\|·Im(z) + c | Perpendicular Mandelbrot variant |
| **Buffalo** | \|Re(z²)\| - i·\|Im(z²)\| + c | Abs on both parts after squaring |
| **Nebulabrot** | Buddhabrot RGB | Orbit-density histogram, separate iteration caps per channel |

## Deep Zoom

The view center is stored in arbitrary precision (`rug`/GMP); zoom depth is
effectively unlimited from the math side:

- **Double-single coordinates** in the standard escape shader (~48-bit mantissa)
  carry the first ~1e-7 of zoom.
- Below `pixel_step < 1e-7`, all z²+c variants switch to **perturbation
  rendering**: a reference orbit is iterated on the CPU at full precision, and
  the GPU iterates only the per-pixel delta.
- Plain Mandelbrot additionally builds a **BLA (bivariate linear approximation)
  tree** so the GPU can skip runs of iterations (disabled above 250k
  iterations to bound memory; per-step perturbation still works there).
- Pauldelbrot glitch detection + reference rebasing corrects perturbation
  artifacts.

## Coloring

Sixteen palettes, most with one or two live parameter sliders. Escape-time
fractals use smooth iteration coloring; Newton/Nova use root-basin coloring.

Classic HSV, Oklab, Smooth Gradient, Monochrome, Thin Film, Midnight Aurora,
Storm, Canopy, Bioluminescence, STEVE, Inverted Pair — plus five palettes
built on richer per-pixel signals:

- **Obsidian** — relief-lit volcanic glass: the escape-direction derivative
  becomes a surface normal (movable light azimuth), the boundary smolders
  through as a zoom-invariant ember rim (distance estimation).
- **Noctilucent** — night-shining clouds: orbit stripe-averages form silvery
  wisps over a twilight sky.
- **Lichtenberg** — fossilized lightning: stripe filaments branch off the
  white-hot boundary trunk in amber.
- **Corona** — the boundary as a neon plasma filament of constant screen
  width at any zoom depth, hue drifting along the discharge.
- **Slot Canyon** — sandstone strata carved by stripe fields, lit by a low
  warm keylight with crevice occlusion and reflected ember glow.

The new palettes also color the set **interior** (orbit-average field) instead
of leaving it black, and their signals — extended-range derivative (arg + log2
magnitude, tracked through perturbation and BLA), distance estimation, stripe
averages, interior field — are available for building more.

Two anti-aliasing modes:

- **Accumulate** — jittered sub-pixel samples averaged in linear color space
  (6x6 = 36 or 8x8 = 64 samples).
- **Median** — per-pixel median of 9 jittered samples' iteration values;
  better at suppressing single-sample fireflies near the boundary.

Palettes that sample neighboring pixels for screen-space gradients (Storm,
Bioluminescence, STEVE) automatically disable supersampling.

## Requirements

- Rust toolchain + C compiler and `m4`/`make` (the `rug` crate builds GMP/MPFR from source)
- GPU with Vulkan, Metal, or DX12 support
- Linux: Wayland and/or X11 dev libraries

## Build & Run

```bash
cargo run --release   # debug builds are unusably slow for GPU work
```

## CLI Export

```bash
# Basic export (1920x1080 PNG)
cargo run --release -- --export output.png --type mandelbrot

# All options
cargo run --release -- --export out.png \
    --type mandelbrot \          # see table above; case/space-insensitive
    --width 3840 --height 2160 \
    --iter 50000 \               # max iterations (10..=1000000)
    --ss 3 \                     # supersampling: 1=off, 2=6x6, 3=8x8
    --palette steve \            # classic|oklab|smooth|mono|thinfilm|aurora|storm|canopy|biolum|
                                 # steve|ip|obsidian|noctilucent|lichtenberg|corona|slotcanyon
    --median | --no-median \     # AA filter (default: median)
    --bounds -2.5,1.0,-1.25,1.25 # x_min,x_max,y_min,y_max

# Deep zoom via arbitrary-precision center (use "Copy CLI args" in the GUI)
cargo run --release -- --export deep.png --type mandelbrot \
    --center-re "-1.7492046334625463..." --center-im "0.000047..." \
    --zoom 1e-50 --iter 100000

# Nebulabrot (separate pipeline)
cargo run --release -- --export nebula.png --type nebulabrot \
    --width 1920 --height 1080 --nebula-samples 500000000 --nebula-iters 5000,500,50

# Zoom video: geometric zoom from --zoom-start down to --zoom-end, PNG frames
# + zoom.mp4 (if ffmpeg is on PATH). Find a spot in the GUI, "Copy CLI args",
# then add --zoom-video and frame settings. The perturbation reference orbit
# is computed once and shared by every frame.
cargo run --release -- --zoom-video out_dir \
    --type mandelbrot --center-re "..." --center-im "..." \
    --zoom-end 1e-30 --zoom-start 4.0 --frames 900 --fps 60 \
    --width 1920 --height 1080 --iter 50000 --palette steve

# Newton / Nova / Multibrot extras
#   --degree N   polynomial degree for z^n - 1 (2..=8)
#   --power P    Multibrot exponent (2.0..=8.0)
```

## Controls

### Mouse
- **Scroll**: Zoom toward cursor (with instant zoom preview)
- **Click + Drag**: Pan (texture follows during drag, re-renders on release)
- **Double-click**: Center view on the clicked point (undoable with Backspace)

### Keyboard
- **R**: Reset view to defaults
- **Backspace**: Undo last navigation
- **Arrow keys**: Pan 10% of view
- **+/-**: Zoom in/out (centered)
- **Ctrl+Q**: Quit

### Side Panel
- Fractal type, max iterations (log slider to 1M), supersampling, AA filter,
  palette + palette parameters, per-type parameters (Julia c, power d,
  degree n, relaxation a), Nebulabrot iteration/sample controls
- Precise center coordinates with **Copy CLI args** for reproducing a view
- PNG export of the current view, plus threaded high-res export (up to
  15360x8640) and Nebulabrot export with progress/cancel

## Architecture

```
src/
├── main.rs          # Entry point, CLI parsing, Nebulabrot CLI export
├── app.rs           # FractalApp: egui UI, mouse/keyboard, render scheduling
├── gpu.rs           # GpuState: wgpu pipelines/buffers, interactive render loop
├── export.rs        # Headless high-res export (own wgpu device, any size)
├── nebula.rs        # Threaded Nebulabrot export with progress reporting
├── fractals.rs      # FractalType/params, reference orbits, BLA tree build
└── shaders/
    ├── escape.wgsl          # Escape-time iteration (double-single coords)
    ├── escape_perturb.wgsl  # Perturbation + BLA deep-zoom iteration
    ├── newton.wgsl          # Newton / Nova iteration
    ├── colorize.wgsl        # All palettes; accumulation supersampling
    ├── finalize.wgsl        # Weight-normalize accumulated samples, pack RGBA
    ├── median_finalize.wgsl # Median-of-samples AA + coloring in one pass
    ├── nebula_sample.wgsl   # Buddhabrot orbit sampling into RGB histograms
    └── nebula_finalize.wgsl # Histogram → exposure-normalized RGBA
```

### GPU Pipeline (escape/Newton fractals)

Per sub-pixel sample: iterate (escape/perturb/newton shader) → colorize into an
accumulation buffer. Then a finalize pass (accumulate: divide by total weight;
median: median of per-sample iteration counts, then color) packs RGBA, which is
read back to the CPU and uploaded as an egui texture.

### Key Design Decisions

- **Complex numbers in WGSL**: `vec2<f32>` with custom `cmul`/`csqr`/`cdiv`/`cpow_int`.
- **View state**: arbitrary-precision center (`rug::Float`) + f64 half-ranges;
  GPU gets Dekker-split f32 hi/lo center + per-pixel step.
- **Newton roots**: analytic nth roots of unity — no readback or clustering.
- **Convergence criterion**: escape checks `|z|² > 256`; Newton/Nova check step
  size `|z_new - z|² < tol²`, not residual (Nova fixed points have f(z) ≠ 0).
- **Width alignment**: buffer *stride* is rounded up to 64 pixels for wgpu's
  256-byte row alignment; padding is stripped before display/export.
- **Render-on-demand**: re-render only on param-hash change; SS=1 during
  interaction with a deferred full-quality pass 150 ms after input settles.
